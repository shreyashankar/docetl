"""High-level API for defining, optimizing, and running DocETL pipelines."""

import inspect
import os
from typing import TYPE_CHECKING, Any, Callable, Type

import yaml
from rich import print

from docetl.runner import DSLRunner
from docetl.utils import op_ref_name

if TYPE_CHECKING:
    from docetl.moar.optimizer import MOARResult
from docetl.schemas import (
    ClusterOp,
    CodeFilterOp,
    CodeMapOp,
    CodeReduceOp,
    Dataset,
    EquijoinOp,
    ExtractOp,
    FilterOp,
    GatherOp,
    MapOp,
    OpType,
    ParallelMapOp,
    ParsingTool,
    PipelineOutput,
    PipelineStep,
    ReduceOp,
    ResolveOp,
    SampleOp,
    SplitOp,
    UnnestOp,
)

from pydantic import BaseModel


class Pipeline:
    """Typed pipeline object API.

    .. deprecated::
        Prefer the Frame API (``docetl.read_json(...).map(...)``) for new
        code. Kept for backward compatibility and internal use
        (``DSLRunner`` builds one from every config).
    """

    _OP_TYPE_REGISTRY: dict[str, type] = {
        "map": MapOp,
        "resolve": ResolveOp,
        "reduce": ReduceOp,
        "parallel_map": ParallelMapOp,
        "filter": FilterOp,
        "equijoin": EquijoinOp,
        "split": SplitOp,
        "gather": GatherOp,
        "unnest": UnnestOp,
        "cluster": ClusterOp,
        "sample": SampleOp,
        "code_map": CodeMapOp,
        "code_reduce": CodeReduceOp,
        "code_filter": CodeFilterOp,
        "extract": ExtractOp,
    }

    def __init__(
        self,
        name: str,
        datasets: dict[str, Dataset],
        operations: list[OpType],
        steps: list[PipelineStep],
        output: PipelineOutput,
        parsing_tools: list[ParsingTool | Callable] = [],
        default_model: str | None = None,
        rate_limits: dict[str, int] | None = None,
        optimizer_config: dict[str, Any] = {},
        **kwargs,
    ):
        self.name = name
        self.datasets = datasets
        self.operations = operations
        self.steps = steps
        self.output = output
        self.parsing_tools = [
            (
                tool
                if isinstance(tool, ParsingTool)
                else ParsingTool(
                    name=tool.__name__, function_code=inspect.getsource(tool)
                )
            )
            for tool in parsing_tools
        ]
        self.default_model = default_model
        self.rate_limits = rate_limits
        self.optimizer_config = optimizer_config

        self.other_config = kwargs

        self._output_schema: Type[BaseModel] | None = None

        self._load_env()

    @property
    def ops_by_name(self) -> dict[str, OpType]:
        return {op.name: op for op in self.operations}

    def get_step_for_op(self, op_name: str) -> PipelineStep:
        for step in self.steps:
            for entry in step.operations:
                name = op_ref_name(entry)
                if name == op_name:
                    return step
        raise KeyError(f"Operation {op_name!r} not found in any step")

    @classmethod
    def from_dict(cls, config: dict[str, Any], name: str | None = None) -> "Pipeline":
        datasets = {}
        for ds_name, ds_cfg in config.get("datasets", {}).items():
            datasets[ds_name] = Dataset(**ds_cfg)

        operations: list[OpType] = []
        for op_cfg in config.get("operations", []):
            op_type = op_cfg.get("type")
            schema_cls = cls._OP_TYPE_REGISTRY.get(op_type)
            filtered = {k: v for k, v in op_cfg.items() if v is not None}
            if schema_cls is not None:
                try:
                    operations.append(schema_cls(**filtered))
                except Exception:
                    # Keep the correct op type even when validation fails —
                    # syntax_check reports the validation error loudly on
                    # every run path, but typed inspection (ops_by_name,
                    # list_pipeline_operations) must not misreport the type.
                    operations.append(schema_cls.model_construct(**filtered))
            else:
                operations.append(MapOp.model_construct(**filtered))

        steps = []
        for step_cfg in config.get("pipeline", {}).get("steps", []):
            steps.append(
                PipelineStep(**{k: v for k, v in step_cfg.items() if v is not None})
            )

        # Copy before defaulting — the caller's config must not be mutated.
        output_cfg = {
            "type": "file",
            "path": "",
            **(config.get("pipeline", {}).get("output") or {}),
        }
        output = PipelineOutput(**output_cfg)

        parsing_tools = []
        for tool_cfg in config.get("parsing_tools", []) or []:
            if isinstance(tool_cfg, ParsingTool):
                parsing_tools.append(tool_cfg)
            elif isinstance(tool_cfg, dict):
                parsing_tools.append(ParsingTool(**tool_cfg))

        known_keys = {
            "datasets",
            "operations",
            "pipeline",
            "default_model",
            "parsing_tools",
            "rate_limits",
            "optimizer_config",
        }
        other = {k: v for k, v in config.items() if k not in known_keys}

        return cls(
            name=name or "pipeline",
            datasets=datasets,
            operations=operations,
            steps=steps,
            output=output,
            parsing_tools=parsing_tools,
            default_model=config.get("default_model"),
            rate_limits=config.get("rate_limits"),
            optimizer_config=config.get("optimizer_config", {}),
            **other,
        )

    def _load_env(self):
        from dotenv import load_dotenv

        env_file = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)

    def set_output_schema(self, model: Type[BaseModel]) -> "Pipeline":
        """Set pydantic output schema for validation.

        Args:
            model: A pydantic BaseModel subclass. The LLM output will be validated
                   and transformed using this model after pipeline execution.

        Returns:
            self, for chaining

        Raises:
            ValueError: If model is not a pydantic BaseModel subclass.
        """
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ValueError(
                "_output_schema must be a pydantic BaseModel subclass"
            )
        self._output_schema = model
        return self

    def get_output_schema(self) -> Type[BaseModel] | None:
        """Get the pydantic output schema, if set.

        Returns:
            The BaseModel subclass, or None if not set.
        """
        return getattr(self, "_output_schema", None)

    def optimize(
        self,
        method: str = "moar",
        # MOAR parameters
        eval_fn: Any = None,
        metric_key: str | None = None,
        models: list[str] | None = None,
        agent_model: str | None = None,
        max_iterations: int = 20,
        save_dir: str | None = None,
        exploration_weight: float = 1.414,
        dataset_path: str | None = None,
        # V1 parameters
        max_threads: int | None = None,
        resume: bool = False,
        save_path: str | None = None,
    ) -> "MOARResult | Pipeline":
        if method == "moar":
            return self._optimize_moar(
                eval_fn=eval_fn,
                metric_key=metric_key,
                models=models,
                agent_model=agent_model,
                max_iterations=max_iterations,
                save_dir=save_dir,
                exploration_weight=exploration_weight,
                dataset_path=dataset_path,
            )
        elif method == "v1":
            return self._optimize_v1(
                max_threads=max_threads,
                resume=resume,
                save_path=save_path,
            )
        else:
            raise ValueError(
                f"Unknown optimization method {method!r}. Use 'moar' or 'v1'."
            )

    def _optimize_moar(self, *, eval_fn, metric_key, **kwargs) -> "MOARResult":
        from docetl.moar.optimizer import run_moar

        return run_moar(self, eval_fn=eval_fn, metric_key=metric_key, **kwargs)

    def _optimize_v1(self, *, max_threads, resume, save_path) -> "Pipeline":
        runner = DSLRunner(
            self._to_dict(),
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
        )
        optimized_config, _ = runner.optimize(
            resume=resume,
            return_pipeline=False,
            save_path=save_path,
        )

        updated = Pipeline(
            name=self.name,
            datasets=self.datasets,
            operations=self.operations,
            steps=self.steps,
            output=self.output,
            default_model=self.default_model,
            parsing_tools=self.parsing_tools,
            optimizer_config=self.optimizer_config,
        )
        updated._update_from_dict(optimized_config)
        return updated

    def run(self, max_threads: int | None = None) -> float:
        runner = DSLRunner(
            self,
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
        )
        result = runner.load_run_save()
        return result

    def run_with_stats(self, max_threads: int | None = None) -> dict[str, Any]:
        runner = DSLRunner(
            self,
            base_name=os.path.join(os.getcwd(), self.name),
            yaml_file_suffix=self.name,
            max_threads=max_threads,
        )
        runner.load_run_save()
        return {
            "cost": runner.total_cost,
            "token_usage": dict(runner.total_token_usage),
        }

    def to_yaml(self, path: str) -> None:
        config = self._to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(config, f)

        print(f"[green]Pipeline saved to {path}[/green]")

    def _to_dict(self) -> dict[str, Any]:
        d = {
            "datasets": {
                name: (
                    dataset.model_dump()
                    if hasattr(dataset, "model_dump")
                    else dataset.dict()
                )
                for name, dataset in self.datasets.items()
            },
            "operations": [
                op.model_dump(exclude_none=True, exclude_unset=True)
                for op in self.operations
            ],
            "pipeline": {
                "steps": [
                    {k: v for k, v in step.model_dump().items() if v is not None}
                    for step in self.steps
                ],
                "output": self.output.model_dump(),
            },
            "default_model": self.default_model,
            "parsing_tools": (
                [tool.model_dump() for tool in self.parsing_tools]
                if self.parsing_tools
                else None
            ),
            "optimizer_config": self.optimizer_config,
            **self.other_config,
        }
        if self.rate_limits:
            d["rate_limits"] = self.rate_limits
        return d

    def _update_from_dict(self, config: dict[str, Any]):
        other = Pipeline.from_dict(config, name=self.name)
        self.datasets = other.datasets
        self.operations = other.operations
        self.steps = other.steps
        self.output = other.output
        self.default_model = other.default_model
        self.parsing_tools = other.parsing_tools
        self.optimizer_config = other.optimizer_config
        self.other_config = other.other_config


# Export the main classes and functions for easy import
__all__ = [
    "Pipeline",
    "Dataset",
    "MapOp",
    "ResolveOp",
    "ReduceOp",
    "ParallelMapOp",
    "FilterOp",
    "EquijoinOp",
    "SplitOp",
    "GatherOp",
    "UnnestOp",
    "CodeMapOp",
    "CodeReduceOp",
    "CodeFilterOp",
    "ExtractOp",
    "PipelineStep",
    "PipelineOutput",
    "ParsingTool",
]
