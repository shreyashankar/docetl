"""Pipeline execution engine with pull-based DAG evaluation."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from collections import defaultdict

# Avoid circular import — Pipeline is only needed for isinstance checks
# and from_dict calls, so import lazily or use TYPE_CHECKING.
from typing import TYPE_CHECKING, Any

import pyrate_limiter
from dotenv import load_dotenv
from pyrate_limiter import BucketFullException, LimiterDelayException
from rich.panel import Panel

from docetl.agents import get_agent_tool_names
from docetl.console import get_console
from docetl.containers import build_operation_graph
from docetl.dataset import DataLoader, create_parsing_tool_map
from docetl.display import format_execution_summary, format_query_plan
from docetl.operations import get_operation
from docetl.operations.utils import APIWrapper
from docetl.optimizer import Optimizer
from docetl.ratelimiter import create_bucket_factory
from docetl.utils import decrypt, load_config, op_ref_name

if TYPE_CHECKING:
    from docetl.api import Pipeline as PipelineType
    from docetl.operations.base import BaseOperation

load_dotenv()


def _create_router(console, fallback_models: list, router_type: str) -> Any | None:
    if not fallback_models:
        return None
    try:
        from litellm import Router
    except ImportError:
        console.log(
            f"[yellow]Warning: LiteLLM Router not available. Fallback {router_type} models will be ignored.[/yellow]"
        )
        return None

    model_list = []
    names = []
    for cfg in fallback_models:
        if isinstance(cfg, dict):
            name, params = cfg.get("model_name"), cfg.get("litellm_params", {})
        elif isinstance(cfg, str):
            name, params = cfg, {}
        else:
            continue
        if not name:
            continue
        model_list.append(
            {
                "model_name": name,
                "litellm_params": {**params, "model": name},
            }
        )
        names.append(name)

    if not model_list:
        return None
    try:
        kwargs = {"model_list": model_list}
        if len(names) > 1:
            kwargs["fallbacks"] = [
                {names[i]: names[i + 1 :]} for i in range(len(names) - 1)
            ]
        router = Router(**kwargs)
        console.log(
            f"[green]Created LiteLLM {router_type} Router with "
            f"{len(model_list)} fallback model(s): {', '.join(names)}[/green]"
        )
        return router
    except Exception as e:
        console.log(
            f"[yellow]Warning: Failed to create LiteLLM {router_type} Router: {e}. "
            f"Fallback models will be ignored.[/yellow]"
        )
        return None


def save_output(data: list[dict], path: str, console) -> None:
    """Write *data* to *path*, dispatching on extension (.json / .parquet /
    CSV fallback). Shared by DSLRunner.save and Frame.write_*."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    elif path.lower().endswith(".parquet"):
        import pandas as pd

        pd.DataFrame(data).to_parquet(path, index=False)
    else:  # CSV
        import csv

        fieldnames: dict[str, None] = {}  # union of row keys, in first-seen order
        for row in data:
            fieldnames.update(dict.fromkeys(row))
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in fieldnames} for row in data)
    console.log(f"[green]✓[/green] Saved to [dim]{path}[/dim]\n")


class DSLRunner:

    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        if not yaml_file.endswith(".yaml") and not yaml_file.endswith(".yml"):
            raise ValueError(
                "Invalid file type. Please provide a YAML file ending with '.yaml' or '.yml'."
            )
        base_name = yaml_file.rsplit(".", 1)[0]
        suffix = yaml_file.split("/")[-1].split(".")[0]
        config = load_config(yaml_file)
        return cls(
            config,
            base_name=base_name,
            yaml_file_suffix=suffix,
            yaml_file=yaml_file,
            **kwargs,
        )

    def __init__(
        self, config: "dict | PipelineType", max_threads: int | None = None, **kwargs
    ):
        self._set_config(config)
        self.base_name = kwargs.pop("base_name", None)
        self.yaml_file = kwargs.pop("yaml_file", None)
        self.yaml_file_suffix = kwargs.pop("yaml_file_suffix", None) or (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.console = kwargs.pop("console", None) or get_console()
        for rewrite in self.applied_rewrites:
            self.console.log(f"[dim]Plan rewrite — {rewrite}[/dim]")
        if getattr(self, "_pipeline_rederived", False):
            self.console.log(
                "[dim]Plan rewrites changed the config, so the passed-in "
                "Pipeline object was re-derived from it; custom op types "
                "outside the typed registry lose their class in "
                "runner.pipeline (execution uses the raw config and is "
                "unaffected). Set plan_rewrites=False on the Pipeline to "
                "keep the original object.[/dim]"
            )
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.status = None

        self._setup_api_keys()
        self._setup_rate_limiter()
        self._setup_routers()
        self.api = APIWrapper(self)

        self.total_cost = 0
        self.total_token_usage = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0}
        )
        self.progress_tracker = None
        self._tui_active = False

        self.datasets = {}
        self.intermediate_dir = self.pipeline.output.intermediate_dir
        # Created once and filled in place by compute_operation_hashes, so the
        # CheckpointStore's reference stays valid when hashes are recomputed
        # (e.g. on optimizer resume).
        self.step_op_hashes: defaultdict[str, dict[str, str]] = defaultdict(dict)
        self._setup_parsing_tools()
        self._setup_retrievers()
        build_operation_graph(self)
        self._compute_hashes()
        self._setup_checkpoints()

        self._from_df_accessors = kwargs.get("from_df_accessors", False)
        if not self._from_df_accessors:
            self.syntax_check()

    def _set_config(self, config: "dict | PipelineType") -> None:
        from docetl.api import Pipeline as PipelineCls

        if isinstance(config, PipelineCls):
            pipeline, cfg = config, config._to_dict()
        else:
            pipeline, cfg = None, config

        # Lift the config into a logical plan once.  Rewrite rules run
        # against the plan; if any fire the config is re-derived via
        # lower().  The plan is kept on the runner so build_operation_graph
        # can walk it instead of re-parsing the raw config dict.
        self.applied_rewrites = []
        from docetl.plan import configured_rules, lift, lower
        from docetl.plan.rewrite import apply_rules, could_fire

        rules = configured_rules(cfg)
        self._plan = lift(cfg)
        if (
            rules
            and could_fire(cfg, rules)
            and not any(i.level == "error" for i in self._plan.issues)
        ):
            self.applied_rewrites = apply_rules(self._plan, rules=rules)
            if self.applied_rewrites:
                cfg = lower(self._plan)

        self.config = cfg
        self._pipeline_rederived = pipeline is not None and bool(self.applied_rewrites)
        if pipeline is not None and not self.applied_rewrites:
            # Keep the caller's typed Pipeline: from_dict degrades op
            # types missing from its registry, so don't re-derive it
            # unless the config actually changed.
            self.pipeline: PipelineCls = pipeline
        else:
            self.pipeline = PipelineCls.from_dict(cfg)
        self._raw_ops_list = self.config.get("operations", [])
        self.default_model = self.config.get("default_model", "gpt-4o-mini")

    def reload(self, config: "dict | PipelineType") -> None:
        """Replace the pipeline config and rebuild all derived state.

        The one place that keeps ``config``, the typed ``pipeline``, the op
        map, the operation graph, and checkpoint hashes in sync — use this
        instead of reassigning any of them by hand. Loaded datasets are
        preserved.
        """
        self._set_config(config)
        self.intermediate_dir = self.pipeline.output.intermediate_dir
        self._setup_parsing_tools()
        self._setup_retrievers()
        build_operation_graph(self)
        self._compute_hashes()

    def _compute_hashes(self) -> None:
        """Compute checkpoint-invalidation hashes for every operation."""
        self.step_op_hashes.clear()
        if not self.intermediate_dir:
            return

        datasets = self.config.get("datasets", {})
        system_prompt = self.pipeline.other_config.get("system_prompt", {})
        step_final_hash: dict[str, str] = {}

        def effective(op_cfg: dict) -> dict:
            if "model" in op_cfg:
                return op_cfg
            return {**op_cfg, "model": self.default_model}

        def input_token(name: str | None) -> dict | None:
            if name is None:
                return None
            if name in step_final_hash:
                return {"step": name, "hash": step_final_hash[name]}
            return {"dataset": name, "config": datasets.get(name)}

        for step in self.pipeline.steps:
            digest = hashlib.sha256()
            for element in ({"system_prompt": system_prompt}, input_token(step.input)):
                digest.update(json.dumps(element, sort_keys=True, default=str).encode())

            for entry in step.operations:
                name = op_ref_name(entry)
                if isinstance(entry, str):
                    element = effective(self._op_map[name])
                else:
                    join_cfg = entry[name]
                    element = {
                        "equijoin": effective(self._op_map[name]),
                        "left": input_token(join_cfg.get("left")),
                        "right": input_token(join_cfg.get("right")),
                    }
                digest.update(json.dumps(element, sort_keys=True, default=str).encode())
                self.step_op_hashes[step.name][name] = digest.copy().hexdigest()

            step_final_hash[step.name] = digest.hexdigest()

    def _setup_api_keys(self) -> None:
        encrypted_llm_api_keys = self.config.get("llm_api_keys", {})
        if encrypted_llm_api_keys:
            self.llm_api_keys = {
                key: decrypt(value, os.environ.get("DOCETL_ENCRYPTION_KEY", ""))
                for key, value in encrypted_llm_api_keys.items()
            }
        else:
            self.llm_api_keys = {}
        self._original_env = os.environ.copy()
        for key, value in self.llm_api_keys.items():
            os.environ[key] = value

    def _setup_rate_limiter(self) -> None:
        bucket_factory = create_bucket_factory(self.config.get("rate_limits", {}))
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=200)
        self.is_cancelled = False

    def _setup_routers(self) -> None:
        self.fallback_models_config = self.config.get("fallback_models", [])
        self.router = _create_router(
            self.console, self.fallback_models_config, "completion"
        )
        self.embedding_router = _create_router(
            self.console, self.config.get("fallback_embedding_models", []), "embedding"
        )
        self._router_cache: dict[str, Any] = {}

    def _setup_checkpoints(self) -> None:
        self._checkpoints = None

    @property
    def checkpoints(self):
        """The checkpoint store for the current ``intermediate_dir``.

        Resolved lazily so that setting ``intermediate_dir`` after
        construction (e.g. in tests, or before a partial-results run)
        behaves the same as passing it in the config.
        """
        if not self.intermediate_dir:
            return None
        if (
            self._checkpoints is None
            or self._checkpoints.base_dir != self.intermediate_dir
        ):
            from docetl.checkpoint import CheckpointStore

            # Hashes are skipped when no intermediate_dir is configured at
            # init, so fill them in now that checkpointing is active.
            if not self.step_op_hashes:
                self._compute_hashes()
            self._checkpoints = CheckpointStore(
                self.intermediate_dir,
                self.step_op_hashes,
                bypass=self.config.get("bypass_cache", False),
            )
        return self._checkpoints

    def reset_env(self):
        os.environ = self._original_env

    def blocking_acquire(self, key: str, weight: int, wait_time=0.5):
        while True:
            try:
                self.rate_limiter.try_acquire(key, weight=weight)
                return
            except LimiterDelayException as e:
                time_to_wait = e.meta_info["actual_delay"] / 1000
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))
            except BucketFullException as e:
                time_to_wait = e.meta_info["remaining_time"]
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))

    def _setup_parsing_tools(self) -> None:
        self.parsing_tool_map = create_parsing_tool_map(
            self.config.get("parsing_tools", None)
        )

    def _setup_retrievers(self) -> None:
        from docetl.retrievers.lancedb import LanceDBRetriever

        self.retrievers: dict[str, Any] = {}
        retrievers_cfg = self.config.get("retrievers", {}) or {}
        for name, rconf in retrievers_cfg.items():
            if not isinstance(rconf, dict):
                raise ValueError(f"Invalid retriever '{name}' configuration")
            if rconf.get("type") != "lancedb":
                raise ValueError(
                    f"Unsupported retriever type '{rconf.get('type')}' for '{name}'. Only 'lancedb' is supported."
                )
            required = ["dataset", "index_dir", "index_types"]
            for key in required:
                if key not in rconf:
                    raise ValueError(
                        f"Retriever '{name}' missing required key '{key}'."
                    )
            # Copy before applying defaults — rconf may be a dict shared with
            # the caller (e.g. a Frame Retriever's config).
            rconf = dict(rconf)
            rconf.setdefault("query", {"top_k": 5})
            rconf.setdefault("build_index", "if_missing")

            self.retrievers[name] = LanceDBRetriever(self, name, rconf)

    def get_output_path(self, require=False):
        output_path = self.pipeline.output.path or None
        valid_exts = (".json", ".csv", ".parquet")
        if output_path:
            if not any(output_path.lower().endswith(ext) for ext in valid_exts):
                raise ValueError(
                    f"Output path '{output_path}' must end with one of {valid_exts}."
                )
        elif require:
            raise ValueError(
                "No output path specified. Provide a path ending with "
                f"one of {valid_exts} in the pipeline output configuration."
            )

        return output_path

    def syntax_check(self):
        self.console.log("[yellow]Checking operations...[/yellow]")

        self.get_output_path()

        for node in self._plan.nodes():
            op_cls = get_operation(node.op_type)
            # Prefer the live op config used at execution time. Plan lift()
            # deep-copies configs into node.op_config, so confirming a
            # non-Jinja prompt against that copy would not stick — the user
            # would be prompted again when _run_operation instantiates the
            # op with find_operation()'s config.
            try:
                op_config = self.find_operation(node.name)
            except ValueError:
                op_config = node.op_config
            try:
                op_cls(
                    self,
                    op_config,
                    self.default_model,
                    self.max_threads,
                    self.console,
                    self.status,
                ).syntax_check()
                self.console.log(
                    f"[green]✓[/green] Operation '{node.name}' ({node.op_type})",
                    end="",
                )
            except Exception as e:
                raise ValueError(
                    f"Syntax check failed for operation '{node.name}': {str(e)}"
                )

        self.console.log("[green]✓ All operations passed syntax check[/green]")

    def pipeline_label(self) -> str:
        """Human-readable pipeline name for logs and the interactive UI."""
        if self.yaml_file:
            return os.path.basename(self.yaml_file)
        if self.base_name:
            return os.path.basename(self.base_name) + ".yaml"
        return "DocETL pipeline"

    def _cascade_model_roles(self) -> dict[str, str]:
        """Map model names to cascade roles for the execution-summary token table."""
        roles: dict[str, str] = {}
        default = self.config.get("default_model")
        for op in self.config.get("operations", []):
            cascade = op.get("cascade")
            if not cascade:
                continue
            if isinstance(cascade, dict):
                proxy = cascade.get("proxy_model")
            else:
                proxy = getattr(cascade, "proxy_model", None)
            if proxy:
                roles[proxy] = "cascade proxy"
            from docetl.operations.utils.cascade_runner import cascade_oracle_model

            oracle = cascade_oracle_model(op, default)
            if oracle:
                roles[oracle] = "oracle"
        return roles

    def print_query_plan(self, show_boundaries=False):
        if self._plan.root is None:
            self.console.log("\n[bold]Pipeline Steps:[/bold]")
            self.console.log(
                Panel("No operations in pipeline", title="Query Plan", width=100)
            )
            self.console.log()
            return

        step_colors, plan_text = format_query_plan(
            self._plan,
            default_model=self.config.get("default_model", "?"),
        )
        if self.applied_rewrites:
            self.console.log("\n[bold]Plan rewrites:[/bold]")
            for rewrite in self.applied_rewrites:
                self.console.log(f"  [dim]{rewrite}[/dim]")
        self.console.log("\n[bold]Pipeline Steps:[/bold]")
        for step_name, color in step_colors.items():
            self.console.log(f"[{color}]■[/{color}] {step_name}")
        self.console.log(Panel(plan_text, title="Query Plan", width=100))
        self.console.log()

    def find_operation(self, op_name: str) -> dict:
        try:
            return self._op_map[op_name]
        except KeyError:
            raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def list_pipeline_operations(
        self,
    ) -> list[tuple[str, str, str, str | None, list[str]]]:
        ops: list[tuple[str, str, str, str | None, list[str]]] = []
        for step in self._plan.steps:
            for node in step.nodes:
                model = node.op_config.get("model") or self.default_model
                agent_tools = get_agent_tool_names(node.op_config.get("agent"))
                ops.append(
                    (
                        step.name,
                        f"{step.name}/{node.name}",
                        node.op_type,
                        model,
                        agent_tools,
                    )
                )
        return ops

    def _should_use_tui(self) -> bool:
        import sys

        if self._tui_active:
            return False
        if not self.config.get("interactive_ui", False):
            return False
        if not (sys.stdout.isatty() and sys.stdin.isatty()):
            self.console.log(
                "[yellow]interactive_ui requested but stdout is not a TTY; "
                "falling back to standard output.[/yellow]"
            )
            return False
        return True

    def load_run_save(self) -> float:
        if self._should_use_tui():
            try:
                from docetl.tui.app import run_with_tui
            except ImportError:
                self.console.log(
                    "[yellow]interactive_ui is enabled but the 'textual' package "
                    "is not installed. Install it with `pip install docetl[tui]` "
                    "to use the interactive progress view. Falling back to "
                    "standard output.[/yellow]"
                )
            else:
                return run_with_tui(self)

        output_path = self.get_output_path(require=True)
        self.print_query_plan()

        execution_time = 0.0
        if self.last_op_container:
            output, execution_time = self.run()
            self.save(output)

        self._log_summary(execution_time, output_path)

        if self.progress_tracker is not None:
            self.progress_tracker.pipeline_done()

        return self.total_cost

    def run(self) -> tuple[list[dict], float]:
        """Load datasets and execute the operation DAG.

        Returns ``(output rows, execution seconds)``. Does not save or log
        the execution summary — callers decide both.
        """
        if not self.last_op_container:
            raise ValueError("Pipeline has no operations to execute.")
        start_time = time.time()
        self.load()
        self.console.rule("[bold]Pipeline Execution[/bold]")
        output, _, _ = self.last_op_container.next()
        return output, time.time() - start_time

    def _log_summary(self, execution_time: float, output_path: str | None) -> None:
        summary = format_execution_summary(
            self.total_cost,
            execution_time,
            self.total_token_usage,
            self.intermediate_dir,
            output_path,
            cascade_roles=self._cascade_model_roles(),
        )
        self.console.log(Panel(summary, title="Execution Summary"))

    def load(self) -> None:
        self.console.rule("[bold]Loading Datasets[/bold]")
        self.datasets = {}

        for name, ds in self.pipeline.datasets.items():
            parsing = ds.parsing or []
            self.datasets[name] = DataLoader(
                self,
                ds.type,
                ds.path,
                source="local",
                parsing=parsing,
                user_defined_parsing_tool_map=self.parsing_tool_map,
            )
            label = ds.path if ds.type == "file" else "in-memory data"
            self.console.log(f"[green]✓[/green] Loaded dataset '{name}' from {label}")

        self.datasets["__empty__"] = DataLoader(self, "memory", [{}])
        self.console.log()

    def save(self, data: list[dict]) -> None:
        self.get_output_path(require=True)

        out = self.pipeline.output
        if out.type != "file":
            raise ValueError(
                f"Unsupported output type: {out.type}. Supported types: file"
            )
        save_output(data, out.path, self.console)

    def clear_intermediate(self) -> None:
        if self.checkpoints:
            self.checkpoints.clear_all()
            return
        raise ValueError("Intermediate directory not set. Cannot clear intermediate.")

    def _prepare_optimizer_kwargs(self, **kwargs) -> dict:
        opt_cfg = self.pipeline.optimizer_config or {}
        kwargs.setdefault("litellm_kwargs", opt_cfg.get("litellm_kwargs", {}))
        kwargs.setdefault(
            "rewrite_agent_model", opt_cfg.get("rewrite_agent_model", "gpt-5.1")
        )
        kwargs.setdefault(
            "judge_agent_model", opt_cfg.get("judge_agent_model", "gpt-4o-mini")
        )
        return kwargs

    def should_optimize(
        self, step_name: str, op_name: str, **kwargs
    ) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
        self.load()
        builder = Optimizer(self, **self._prepare_optimizer_kwargs(**kwargs))
        self.optimizer = builder
        return builder.should_optimize(step_name, op_name)

    def optimize(
        self,
        save: bool = False,
        return_pipeline: bool = True,
        **kwargs,
    ) -> tuple[dict | "DSLRunner", float]:
        if not self.last_op_container:
            raise ValueError("No operations in pipeline. Cannot optimize.")

        self.load()
        save_path = kwargs.pop("save_path", None)

        builder = Optimizer(self, **self._prepare_optimizer_kwargs(**kwargs))
        self.optimizer = builder
        llm_api_cost = builder.optimize()
        operations_cost = self.total_cost
        self.total_cost += llm_api_cost

        self.console.log(
            f"[green italic]💰 Total cost: ${self.total_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  ├─ Operation execution cost: ${operations_cost:.4f}[/green italic]"
        )
        self.console.log(
            f"[green italic]  └─ Optimization cost: ${llm_api_cost:.4f}[/green italic]"
        )

        if save:
            if save_path:
                if not os.path.isabs(save_path):
                    save_path = os.path.join(os.getcwd(), save_path)
            else:
                save_path = f"{self.base_name}_opt.yaml"
            builder.save_optimized_config(save_path)
            self.optimized_config_path = save_path

        if return_pipeline:
            return (
                DSLRunner(builder.clean_optimized_config(), self.max_threads),
                self.total_cost,
            )

        return builder.clean_optimized_config(), self.total_cost

    def _make_operation(self, op_config: dict[str, Any]) -> "BaseOperation":
        """Construct an operation instance without executing it.

        Kept separate from :meth:`_run_operation` so callers can instantiate
        (and handle any interactive prompts) before starting a Rich status
        spinner that would otherwise fight the terminal UI.
        """
        operation_class = get_operation(op_config["type"])
        return operation_class(
            runner=self,
            config=op_config,
            default_model=self.default_model,
            max_threads=self.max_threads,
            console=self.console,
            status=self.status,
        )

    def _run_operation(
        self,
        op_config: dict[str, Any],
        input_data: list[dict[str, Any]] | dict[str, Any],
        return_instance: bool = False,
        is_build: bool = False,
        operation_instance: "BaseOperation | None" = None,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], BaseOperation]:
        if operation_instance is None:
            operation_instance = self._make_operation(op_config)
        else:
            # Keep the instance's status pointer in sync with the runner's
            # current spinner (set after instantiation in _execute_and_track).
            operation_instance.status = self.status
        if op_config["type"] == "equijoin":
            output_data, cost = operation_instance.execute(
                input_data["left_data"], input_data["right_data"]
            )
        elif op_config["type"] == "filter":
            output_data, cost = operation_instance.execute(input_data, is_build)
        else:
            output_data, cost = operation_instance.execute(input_data)

        self.total_cost += cost

        if return_instance:
            return output_data, operation_instance
        else:
            return output_data

    def _flush_partial_results(
        self, operation_name: str, batch_index: int, data: list[dict]
    ) -> None:
        if not self.checkpoints:
            return
        path = self.checkpoints.flush_batch(operation_name, batch_index, data)
        self.console.log(
            f"[green]✓[/green] [italic]Partial checkpoint saved for '{operation_name}', "
            f"batch {batch_index} at '{path}'[/italic]"
        )
