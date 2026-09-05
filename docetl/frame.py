"""PySpark-like API for docetl pipelines.

Usage::

    import docetl

    docetl.default_model = "gpt-4o-mini"

    results = (
        docetl.read_json("input.json")
        .map(prompt="Summarize: {{ input.text }}",
             output={"schema": {"summary": "string"}})
        .filter(prompt="Is this good? {{ input.summary }}",
                output={"schema": {"keep": "boolean"}})
        .collect()
    )
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from docetl import _config
from docetl.console import get_console
from docetl.utils import op_ref_name

if TYPE_CHECKING:
    import pandas as pd

    from docetl.plan import LogicalPlan

# Top-level pipeline config keys a Frame carries with it (set by from_yaml).
# These take precedence over the module-level ``docetl.<attr>`` globals.
_SETTING_KEYS = (
    "default_model",
    "rate_limits",
    "bypass_cache",
    "fallback_models",
    "fallback_embedding_models",
    "parsing_tools",
    "system_prompt",
    "plan_rewrites",
)


def _data_loader(
    ds: dict[str, Any],
    parsing_tools: list[dict] | None = None,
    apply_parsing: bool = True,
):
    """A DataLoader for a dataset config (the same loader execution uses),
    so file formats and parsing tools behave identically here."""
    from docetl.dataset import DataLoader, create_parsing_tool_map

    parsing = ds.get("parsing") if apply_parsing else None
    return DataLoader(
        None,
        ds.get("type", "memory"),
        ds.get("path", []),
        parsing=parsing,
        user_defined_parsing_tool_map=create_parsing_tool_map(parsing_tools),
    )


class Retriever:
    """A LanceDB retriever configuration. Pass to operations via ``retriever=``."""

    _counter = 0

    def __init__(
        self,
        dataset: str | None = None,
        index_dir: str | None = None,
        index_types: list[str] | None = None,
        *,
        data: str | list[dict] | None = None,
        fts: dict[str, str] | None = None,
        embedding: dict[str, str] | None = None,
        query: dict[str, Any] | None = None,
        build_index: str = "if_missing",
    ):
        if (dataset is None) == (data is None):
            raise ValueError(
                "Pass exactly one of 'data' (a file path or list of dicts to "
                "index) or 'dataset' (the name of a pipeline dataset or step)."
            )
        if index_dir is None or index_types is None:
            raise ValueError("'index_dir' and 'index_types' are required.")
        Retriever._counter += 1
        self._name = f"retriever_{Retriever._counter}"
        # Inline data is registered as a pipeline dataset under this name
        # when the retriever is attached to an operation.
        self._data = data
        self._config: dict[str, Any] = {
            "type": "lancedb",
            "dataset": dataset if dataset is not None else f"{self._name}_data",
            "index_dir": index_dir,
            "index_types": index_types,
            "build_index": build_index,
        }
        if fts is not None:
            self._config["fts"] = fts
        if embedding is not None:
            self._config["embedding"] = embedding
        if query is not None:
            self._config["query"] = query


class Frame:
    """A lazy pipeline frame—operations are recorded but not executed
    until a terminal action (``.collect()``, ``.write_json()``, etc.) is called."""

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        operations: list[dict[str, Any]] | None = None,
        steps: list[dict[str, Any]] | None = None,
        *,
        _last_step: str | None = None,
        _first_dataset: str | None = None,
        _op_counter: dict[str, int] | None = None,
        _retrievers: dict[str, dict[str, Any]] | None = None,
        _settings: dict[str, Any] | None = None,
    ):
        self._datasets = datasets
        self._operations = operations or []
        self._steps = steps or []
        self._last_step = _last_step
        self._first_dataset = _first_dataset or next(iter(datasets), None)
        self._op_counter = _op_counter or {}
        self._retrievers = _retrievers or {}
        # Per-frame pipeline settings (see _SETTING_KEYS). These travel with
        # the Frame and take precedence over the docetl.<attr> globals.
        self._settings = _settings or {}
        self._total_cost: float = 0.0
        self._token_usage: dict[str, dict[str, int]] = {}
        # Memoized (config key, output, cost, token usage) of the last
        # execution — terminal actions on an unchanged config reuse it.
        self._memo: tuple[str, list[dict], float, dict] | None = None
        # Per-dataset content digests and sampled-frame cache for show();
        # deliberately not carried through _copy (frames are immutable, but a
        # copy may swap in different datasets under the same names).
        self._ds_digests: dict[str, str] = {}
        self._sample_cache: dict[int, Frame] = {}

    def _copy(self, **overrides) -> Frame:
        kw = dict(
            datasets=self._datasets,
            operations=list(self._operations),
            steps=list(self._steps),
            _last_step=self._last_step,
            _first_dataset=self._first_dataset,
            _op_counter=dict(self._op_counter),
            _retrievers=dict(self._retrievers),
            _settings=dict(self._settings),
        )
        kw.update(overrides)
        return Frame(**kw)

    # ── auto-naming ────────────────────────────────────────────────

    def _auto_name(self, op_type: str) -> tuple[str, dict[str, int]]:
        counter = dict(self._op_counter)
        counter[op_type] = counter.get(op_type, 0) + 1
        return f"{op_type}_{counter[op_type]}", counter

    def _append_op(
        self, op_type: str, name: str | None, config: dict[str, Any]
    ) -> Frame:
        name_val, new_counter = (
            self._auto_name(op_type) if name is None else (name, self._op_counter)
        )

        new_retrievers = dict(self._retrievers)
        new_datasets = self._datasets
        retriever = config.get("retriever")
        if isinstance(retriever, Retriever):
            new_retrievers[retriever._name] = retriever._config
            config = {**config, "retriever": retriever._name}
            if retriever._data is not None:
                ds: dict[str, Any] = (
                    {"type": "file", "path": retriever._data}
                    if isinstance(retriever._data, str)
                    else {"type": "memory", "path": retriever._data}
                )
                new_datasets = {
                    **self._datasets,
                    retriever._config["dataset"]: ds,
                }

        op = {
            "name": name_val,
            "type": op_type,
            **{k: v for k, v in config.items() if v is not None},
        }

        step_input = self._last_step or self._first_dataset
        step_name = f"step_{name_val}"
        step: dict[str, Any] = {"name": step_name, "operations": [name_val]}
        if step_input:
            step["input"] = step_input

        new = self._copy(
            datasets=new_datasets,
            operations=self._operations + [op],
            steps=self._steps + [step],
            _last_step=step_name,
            _op_counter=dict(new_counter),
            _retrievers=new_retrievers,
        )
        return new

    def _append_equijoin(
        self, name: str | None, left: str, right: str, config: dict[str, Any]
    ) -> Frame:
        name_val, new_counter = (
            self._auto_name("equijoin") if name is None else (name, self._op_counter)
        )
        op = {
            "name": name_val,
            "type": "equijoin",
            **{k: v for k, v in config.items() if v is not None},
        }
        step_name = f"step_{name_val}"
        step: dict[str, Any] = {
            "name": step_name,
            "operations": [{name_val: {"left": left, "right": right}}],
        }

        return self._copy(
            operations=self._operations + [op],
            steps=self._steps + [step],
            _last_step=step_name,
            _op_counter=dict(new_counter),
        )

    def _merge_pipeline(self, right: Frame) -> tuple[dict[str, Any], str | None]:
        """Merge *right*'s datasets, operations, steps, and retrievers into
        this frame's namespace so the two can be joined.

        Identically-defined entries (shared ancestry) are shared; a name that
        collides with a different definition gets a unique suffix on the
        right side, with all references inside right's steps and ops
        rewritten to match. Returns ``_copy`` kwargs for the merged frame
        plus right's join input (its last step, or its source dataset) under
        its post-merge name.
        """
        ds_ren: dict[str, str] = {}
        op_ren: dict[str, str] = {}
        step_ren: dict[str, str] = {}
        ret_ren: dict[str, str] = {}

        datasets = dict(self._datasets)
        for name, cfg in right._datasets.items():
            if name in datasets and datasets[name] != cfg:
                ds_ren[name] = _unique(name, datasets)
                datasets[ds_ren[name]] = cfg
            else:
                datasets[name] = cfg

        retrievers = dict(self._retrievers)
        for name, cfg in right._retrievers.items():
            if name in retrievers and retrievers[name] != cfg:
                ret_ren[name] = _unique(name, retrievers)
                retrievers[ret_ren[name]] = cfg
            else:
                retrievers[name] = cfg

        operations = list(self._operations)
        ops_by_name = {op["name"]: op for op in operations}
        for op in right._operations:
            op = dict(op)
            if op.get("retriever") in ret_ren:
                op["retriever"] = ret_ren[op["retriever"]]
            existing = ops_by_name.get(op["name"])
            if existing == op:
                continue
            if existing is not None:
                op_ren[op["name"]] = _unique(op["name"], ops_by_name)
                op["name"] = op_ren[op["name"]]
            operations.append(op)
            ops_by_name[op["name"]] = op

        def ref(name: str) -> str:
            return step_ren.get(name, ds_ren.get(name, name))

        steps = list(self._steps)
        steps_by_name = {s["name"]: s for s in steps}
        for step in right._steps:
            step = dict(step)
            if step.get("input"):
                step["input"] = ref(step["input"])
            step["operations"] = [
                (
                    op_ren.get(entry, entry)
                    if isinstance(entry, str)
                    else {
                        op_ren.get(jn, jn): {side: ref(v) for side, v in jc.items()}
                        for jn, jc in entry.items()
                    }
                )
                for entry in step.get("operations", [])
            ]
            existing = steps_by_name.get(step["name"])
            if existing == step:
                continue
            if existing is not None:
                step_ren[step["name"]] = _unique(step["name"], steps_by_name)
                step["name"] = step_ren[step["name"]]
            steps.append(step)
            steps_by_name[step["name"]] = step

        counter = dict(self._op_counter)
        for k, v in right._op_counter.items():
            counter[k] = max(counter.get(k, 0), v)

        right_input = (
            ref(right._last_step)
            if right._last_step
            else ds_ren.get(right._first_dataset, right._first_dataset)
        )

        kw = dict(
            datasets=datasets,
            operations=operations,
            steps=steps,
            _op_counter=counter,
            _retrievers=retrievers,
            _settings={**right._settings, **self._settings},
        )
        return kw, right_input

    # ── LLM operations ─────────────────────────────────────────────

    def map(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        optimize: bool | None = None,
        recursively_optimize: bool | None = None,
        sample: int | None = None,
        agent: Any | None = None,
        validate: list[str] | None = None,
        num_retries_on_validate_failure: int | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        enable_observability: bool | None = None,
        max_batch_size: int | None = None,
        clustering_method: str | None = None,
        batch_prompt: str | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        pdf_url_key: str | None = None,
        flush_partial_results: bool | None = None,
        limit: int | None = None,
        calibrate: bool | None = None,
        num_calibration_docs: int | None = None,
        retriever: Retriever | str | None = None,
        execution: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "map",
            name,
            {
                "prompt": prompt,
                "output": output,
                "model": model,
                "optimize": optimize,
                "recursively_optimize": recursively_optimize,
                "sample": sample,
                "agent": agent,
                "validate": validate,
                "num_retries_on_validate_failure": num_retries_on_validate_failure,
                "drop_keys": drop_keys,
                "timeout": timeout,
                "enable_observability": enable_observability,
                "max_batch_size": max_batch_size,
                "clustering_method": clustering_method,
                "batch_prompt": batch_prompt,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                "pdf_url_key": pdf_url_key,
                "flush_partial_results": flush_partial_results,
                "limit": limit,
                "calibrate": calibrate,
                "num_calibration_docs": num_calibration_docs,
                "retriever": retriever,
                "execution": execution,
                **kwargs,
            },
        )

    def parallel_map(
        self,
        name: str | None = None,
        *,
        prompts: list[dict[str, Any]] | None = None,
        output: dict[str, Any] | None = None,
        drop_keys: list[str] | None = None,
        enable_observability: bool | None = None,
        pdf_url_key: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "parallel_map",
            name,
            {
                "prompts": prompts,
                "output": output,
                "drop_keys": drop_keys,
                "enable_observability": enable_observability,
                "pdf_url_key": pdf_url_key,
                **kwargs,
            },
        )

    def filter(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        optimize: bool | None = None,
        agent: Any | None = None,
        validate: list[str] | None = None,
        drop_keys: list[str] | None = None,
        timeout: int | None = None,
        max_batch_size: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        retriever: Retriever | str | None = None,
        cascade: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "filter",
            name,
            {
                "prompt": prompt,
                "output": output,
                "model": model,
                "optimize": optimize,
                "agent": agent,
                "validate": validate,
                "drop_keys": drop_keys,
                "timeout": timeout,
                "max_batch_size": max_batch_size,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                "limit": limit,
                "retriever": retriever,
                "cascade": cascade,
                **kwargs,
            },
        )

    def reduce(
        self,
        name: str | None = None,
        *,
        reduce_key: str | list[str] | None = None,
        prompt: str | None = None,
        output: dict[str, Any] | None = None,
        model: str | None = None,
        input: dict[str, Any] | None = None,
        optimize: bool | None = None,
        synthesize_resolve: bool | None = None,
        pass_through: bool | None = None,
        associative: bool | None = None,
        fold_prompt: str | None = None,
        fold_batch_size: int | None = None,
        merge_prompt: str | None = None,
        merge_batch_size: int | None = None,
        value_sampling: dict[str, Any] | None = None,
        verbose: bool | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        agent: Any | None = None,
        enable_observability: bool | None = None,
        limit: int | None = None,
        retriever: Retriever | str | None = None,
        execution: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "reduce",
            name,
            {
                "reduce_key": reduce_key,
                "prompt": prompt,
                "output": output,
                "model": model,
                "input": input,
                "optimize": optimize,
                "synthesize_resolve": synthesize_resolve,
                "pass_through": pass_through,
                "associative": associative,
                "fold_prompt": fold_prompt,
                "fold_batch_size": fold_batch_size,
                "merge_prompt": merge_prompt,
                "merge_batch_size": merge_batch_size,
                "value_sampling": value_sampling,
                "verbose": verbose,
                "timeout": timeout,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                "agent": agent,
                "enable_observability": enable_observability,
                "limit": limit,
                "retriever": retriever,
                "execution": execution,
                **kwargs,
            },
        )

    def resolve(
        self,
        name: str | None = None,
        *,
        comparison_prompt: str | None = None,
        resolution_prompt: str | None = None,
        output: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        resolution_model: str | None = None,
        comparison_model: str | None = None,
        blocking_keys: list[str] | None = None,
        blocking_threshold: float | None = None,
        blocking_target_recall: float | None = None,
        blocking_conditions: list[str] | None = None,
        input: dict[str, Any] | None = None,
        embedding_batch_size: int | None = None,
        compare_batch_size: int | None = None,
        limit_comparisons: int | None = None,
        optimize: bool | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        enable_observability: bool | None = None,
        cascade: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "resolve",
            name,
            {
                "comparison_prompt": comparison_prompt,
                "resolution_prompt": resolution_prompt,
                "output": output,
                "embedding_model": embedding_model,
                "resolution_model": resolution_model,
                "comparison_model": comparison_model,
                "blocking_keys": blocking_keys,
                "blocking_threshold": blocking_threshold,
                "blocking_target_recall": blocking_target_recall,
                "blocking_conditions": blocking_conditions,
                "input": input,
                "embedding_batch_size": embedding_batch_size,
                "compare_batch_size": compare_batch_size,
                "limit_comparisons": limit_comparisons,
                "optimize": optimize,
                "timeout": timeout,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                "enable_observability": enable_observability,
                "cascade": cascade,
                **kwargs,
            },
        )

    def equijoin(
        self,
        right: Frame,
        name: str | None = None,
        *,
        comparison_prompt: str | None = None,
        output: dict[str, Any] | None = None,
        blocking_threshold: float | None = None,
        blocking_target_recall: float | None = None,
        blocking_conditions: list[str] | None = None,
        limits: dict[str, int] | None = None,
        comparison_model: str | None = None,
        optimize: bool | None = None,
        embedding_model: str | None = None,
        embedding_batch_size: int | None = None,
        compare_batch_size: int | None = None,
        limit_comparisons: int | None = None,
        blocking_keys: dict[str, list[str]] | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        cascade: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Frame:
        config = {
            "comparison_prompt": comparison_prompt,
            "output": output,
            "blocking_threshold": blocking_threshold,
            "blocking_target_recall": blocking_target_recall,
            "blocking_conditions": blocking_conditions,
            "limits": limits,
            "comparison_model": comparison_model,
            "optimize": optimize,
            "embedding_model": embedding_model,
            "embedding_batch_size": embedding_batch_size,
            "compare_batch_size": compare_batch_size,
            "limit_comparisons": limit_comparisons,
            "blocking_keys": blocking_keys,
            "timeout": timeout,
            "litellm_completion_kwargs": litellm_completion_kwargs,
            "cascade": cascade,
            **kwargs,
        }

        # Join each side's *current* output: the last step if operations have
        # been chained, otherwise the raw source dataset.
        merged_kw, right_input = self._merge_pipeline(right)
        left_input = self._last_step or self._first_dataset
        return self._copy(**merged_kw)._append_equijoin(
            name, left_input, right_input, config
        )

    def extract(
        self,
        name: str | None = None,
        *,
        prompt: str | None = None,
        document_keys: list[str] | None = None,
        model: str | None = None,
        format_extraction: bool | None = None,
        extraction_key_suffix: str | None = None,
        extraction_method: str | None = None,
        timeout: int | None = None,
        litellm_completion_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        retriever: Retriever | str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "extract",
            name,
            {
                "prompt": prompt,
                "document_keys": document_keys,
                "model": model,
                "format_extraction": format_extraction,
                "extraction_key_suffix": extraction_key_suffix,
                "extraction_method": extraction_method,
                "timeout": timeout,
                "litellm_completion_kwargs": litellm_completion_kwargs,
                "limit": limit,
                "retriever": retriever,
                **kwargs,
            },
        )

    # ── structural operations ──────────────────────────────────────

    def split(
        self,
        name: str | None = None,
        *,
        split_key: str | None = None,
        method: str | None = None,
        method_kwargs: dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "split",
            name,
            {
                "split_key": split_key,
                "method": method,
                "method_kwargs": method_kwargs,
                "model": model,
                **kwargs,
            },
        )

    def gather(
        self,
        name: str | None = None,
        *,
        content_key: str | None = None,
        doc_id_key: str | None = None,
        order_key: str | None = None,
        peripheral_chunks: dict[str, Any] | None = None,
        doc_header_key: str | None = None,
        main_chunk_start: str | None = None,
        main_chunk_end: str | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "gather",
            name,
            {
                "content_key": content_key,
                "doc_id_key": doc_id_key,
                "order_key": order_key,
                "peripheral_chunks": peripheral_chunks,
                "doc_header_key": doc_header_key,
                "main_chunk_start": main_chunk_start,
                "main_chunk_end": main_chunk_end,
                **kwargs,
            },
        )

    def unnest(
        self,
        name: str | None = None,
        *,
        unnest_key: str | None = None,
        keep_empty: bool | None = None,
        expand_fields: list[str] | None = None,
        recursive: bool | None = None,
        depth: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "unnest",
            name,
            {
                "unnest_key": unnest_key,
                "keep_empty": keep_empty,
                "expand_fields": expand_fields,
                "recursive": recursive,
                "depth": depth,
                **kwargs,
            },
        )

    def cluster(
        self,
        name: str | None = None,
        *,
        embedding_keys: list[str] | None = None,
        summary_schema: dict[str, Any] | None = None,
        summary_prompt: str | None = None,
        output_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        max_batch_size: int | None = None,
        validate: list[str] | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "cluster",
            name,
            {
                "embedding_keys": embedding_keys,
                "summary_schema": summary_schema,
                "summary_prompt": summary_prompt,
                "output_key": output_key,
                "model": model,
                "embedding_model": embedding_model,
                "max_batch_size": max_batch_size,
                "validate": validate,
                **kwargs,
            },
        )

    def sample(
        self,
        name: str | None = None,
        *,
        method: str | None = None,
        samples: int | float | list | None = None,
        stratify_key: str | list[str] | None = None,
        samples_per_group: bool | None = None,
        method_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "sample",
            name,
            {
                "method": method,
                "samples": samples,
                "stratify_key": stratify_key,
                "samples_per_group": samples_per_group,
                "method_kwargs": method_kwargs,
                "random_state": random_state,
                **kwargs,
            },
        )

    def code_map(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        drop_keys: list[str] | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "code_map",
            name,
            {
                "code": code,
                "drop_keys": drop_keys,
                "concurrent_thread_count": concurrent_thread_count,
                "limit": limit,
                **kwargs,
            },
        )

    def code_reduce(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        reduce_key: str | list[str] | None = None,
        pass_through: bool | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "code_reduce",
            name,
            {
                "code": code,
                "reduce_key": reduce_key,
                "pass_through": pass_through,
                "concurrent_thread_count": concurrent_thread_count,
                "limit": limit,
                **kwargs,
            },
        )

    def code_filter(
        self,
        name: str | None = None,
        *,
        code: str | None = None,
        concurrent_thread_count: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Frame:
        return self._append_op(
            "code_filter",
            name,
            {
                "code": code,
                "concurrent_thread_count": concurrent_thread_count,
                "limit": limit,
                **kwargs,
            },
        )

    # ── inspection ─────────────────────────────────────────────────

    def schema(self) -> dict[str, str]:
        """Return the output schema of this pipeline.

        Delegates to the logical plan's schema propagation (the single
        implementation of the ``transform_schema`` fold), so structural
        ops like split, unnest, gather, and extract are reflected, and
        equijoin frames get a real left+right schema merge instead of a
        linear fold. Static and best-effort — nothing is executed, and
        keys produced by code operations (arbitrary Python) can't be
        known ahead of time.
        """
        from docetl.plan import lift, output_schema

        return output_schema(lift(self._build_config(checkpoint=False)))

    def plan(self) -> "LogicalPlan":
        """Lift this pipeline into its typed logical plan (docetl.plan).

        Built without checkpoint settings so the plan doesn't vary with
        ``docetl.intermediate_dir``. Purely static — nothing executes.
        """
        from docetl.plan import lift

        return lift(self._build_config(checkpoint=False))

    def explain(self, *, schemas: bool = True, optimized: bool = False) -> str:
        """Render the logical plan as an indented tree and return it
        (also printed for interactive use).

        With ``optimized=True``, applies the plan rewrite rules first
        (the same set the runner would use, honoring this frame's
        ``plan_rewrites`` setting) and lists what fired as
        ``-- applied <rule>: ...`` header lines.
        """
        from docetl.display import format_query_plan
        from docetl.plan import apply_rules, configured_rules

        plan = self.plan()
        lines = []
        if optimized:
            rules = configured_rules(self._build_config(checkpoint=False))
            for rewrite in apply_rules(plan, rules=rules):
                lines.append(f"-- applied {rewrite}")
        _, plan_text = format_query_plan(plan)
        lines.append(plan_text)
        text = "\n".join(lines)
        from rich import print as rprint

        rprint(text)
        return text

    # ── terminal actions ───────────────────────────────────────────

    def _build_config(
        self, output_path: str = "", checkpoint: bool = True
    ) -> dict[str, Any]:
        output_cfg: dict[str, Any] = {"type": "file", "path": output_path}
        if checkpoint and _config.intermediate_dir:
            output_cfg["intermediate_dir"] = _config.intermediate_dir

        config: dict[str, Any] = {
            "datasets": self._datasets,
            "operations": self._operations,
            "pipeline": {
                "steps": self._steps,
                "output": output_cfg,
            },
        }
        if self._retrievers:
            config["retrievers"] = self._retrievers
        # Frame-level settings (e.g. from from_yaml) win over the globals.
        config.update({**_config.runner_settings(), **self._settings})
        return config

    def _build_runner(
        self,
        output_path: str = "",
        max_threads: int | None = None,
        checkpoint: bool = True,
    ):
        from docetl.runner import DSLRunner

        threads = max_threads or _config.max_threads
        return DSLRunner(
            self._build_config(output_path, checkpoint=checkpoint),
            max_threads=threads,
        )

    def show(self, max: int = 5, max_threads: int | None = None) -> "pd.DataFrame":
        """Run the pipeline on the first *max* input documents and print results.

        If no operations have been added yet, just shows the raw input data.
        Sampled runs never read or write checkpoints, so a ``show()`` can't
        interfere with a later ``collect()``. Returns the resulting DataFrame.
        """
        import pandas as pd

        if not self._operations:
            data = self._load_input_data(limit=max)
            df = pd.DataFrame(data)
        else:
            # Cache the sampled frame so repeated show() calls reuse its memo
            # instead of re-reading inputs and re-running the sampled pipeline.
            sampled = self._sample_cache.get(max)
            if sampled is None:
                sampled = self._copy(datasets=self._sample_datasets(max))
                self._sample_cache[max] = sampled
            data, cost = sampled._execute(max_threads=max_threads, checkpoint=False)
            df = pd.DataFrame(data)
            df.attrs["_total_cost"] = cost
            df.attrs["_token_usage"] = sampled._token_usage
        print(df.to_string())
        return df

    def count(self, max_threads: int | None = None) -> int:
        """Return the number of output rows.

        On a bare dataset (no operations), returns the input count without
        executing anything.  With operations, executes the pipeline and
        counts the results.
        """
        if not self._operations:
            ds = self._datasets.get(self._first_dataset)
            if not ds:
                return 0
            # Cheap when possible: parquet metadata / CSV streaming.
            return _data_loader(ds, self._settings.get("parsing_tools")).count()
        data, _ = self._execute(max_threads=max_threads)
        return len(data)

    def _load_input_data(self, limit: int | None = None) -> list[dict]:
        """Load the primary input dataset (with parsing applied) without
        executing operations. *limit* caps the raw rows read (parsing then
        applies to just those rows)."""
        ds = self._datasets.get(self._first_dataset)
        if not ds:
            return []
        return _data_loader(ds, self._settings.get("parsing_tools")).load(limit=limit)

    def _sample_datasets(self, n: int) -> dict[str, dict[str, Any]]:
        """Return a copy of the datasets dict with each dataset truncated to
        its first *n* raw rows.

        File reads stop after *n* rows where the format allows. Parsing
        configs are kept on the truncated datasets, so parsing tools still
        run at execution time — but only over the sampled rows.
        """
        return {
            name: {
                **ds,
                "type": "memory",
                "path": _data_loader(ds, apply_parsing=False).load(limit=n),
            }
            for name, ds in self._datasets.items()
        }

    def collect(self, max_threads: int | None = None) -> list[dict]:
        """Execute the pipeline and return the result rows as a list of dicts.

        Results are memoized on the Frame: repeated terminal actions
        (``count()``, ``collect()``, ``to_pandas()``, ``write_*()``) with an
        unchanged configuration reuse the previous result instead of
        re-running operations. Edits to input *files* between calls are
        not detected; rebuild the Frame to force a re-run.
        """
        data, _ = self._execute(max_threads=max_threads)
        return data

    def to_pandas(self, max_threads: int | None = None) -> "pd.DataFrame":
        """Execute the pipeline and return results as a pandas DataFrame.

        Cost and token usage are attached as ``df.attrs["_total_cost"]`` and
        ``df.attrs["_token_usage"]`` (also available on the Frame itself).
        """
        import pandas as pd

        data, cost = self._execute(max_threads=max_threads)
        df = pd.DataFrame(data)
        df.attrs["_total_cost"] = cost
        df.attrs["_token_usage"] = self._token_usage
        return df

    def _execute(
        self, max_threads: int | None = None, checkpoint: bool = True
    ) -> tuple[list[dict], float]:
        """Run the pipeline, memoizing on the built config.

        Repeated terminal actions (``count()`` then ``collect()``, ...)
        reuse the previous result instead of re-running operations. The
        memo key is the full pipeline config — changing ops, data,
        settings, or globals naturally invalidates it. Edits to input
        *files* between calls are not detected.
        """
        config = self._build_config(checkpoint=checkpoint)
        memo_key = self._memo_key(config)
        if self._memo is not None and self._memo[0] == memo_key:
            _, output, cost, usage = self._memo
            self._total_cost = cost
            self._token_usage = dict(usage)
            return [dict(row) for row in output], cost

        from docetl.runner import DSLRunner

        runner = DSLRunner(config, max_threads=max_threads or _config.max_threads)
        output, elapsed = runner.run()
        runner._log_summary(elapsed, "")

        self._total_cost = runner.total_cost
        self._token_usage = dict(runner.total_token_usage)
        self._memo = (memo_key, output, runner.total_cost, self._token_usage)
        # Hand out shallow row copies so caller mutations can't corrupt
        # the memoized result.
        return [dict(row) for row in output], runner.total_cost

    def _memo_key(self, config: dict[str, Any]) -> str:
        """Serialize *config* for memoization, substituting cached content
        digests for memory datasets so a memo check is O(config), not
        O(dataset bytes), on every terminal action."""
        import hashlib
        import json

        datasets = {}
        for name, ds in config.get("datasets", {}).items():
            if ds.get("type") == "memory":
                digest = self._ds_digests.get(name)
                if digest is None:
                    digest = hashlib.sha256(
                        json.dumps(ds, sort_keys=True, default=str).encode()
                    ).hexdigest()
                    self._ds_digests[name] = digest
                datasets[name] = digest
            else:
                datasets[name] = ds
        return json.dumps({**config, "datasets": datasets}, sort_keys=True, default=str)

    @property
    def total_cost(self) -> float:
        """Total cost of the last execution or optimization."""
        return self._total_cost

    @property
    def token_usage(self) -> dict[str, dict[str, int]]:
        """Token usage per model from the last execution."""
        return self._token_usage

    def _write(self, path: str, max_threads: int | None = None) -> None:
        from docetl.runner import save_output

        data, _ = self._execute(max_threads=max_threads)
        save_output(data, path, get_console())

    def write_json(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a JSON file.

        Cost and token usage are available afterwards via
        ``frame.total_cost`` and ``frame.token_usage``.
        """
        self._write(path, max_threads)

    def write_csv(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a CSV file. See write_json."""
        self._write(path, max_threads)

    def write_parquet(self, path: str, max_threads: int | None = None) -> None:
        """Execute the pipeline and write results to a Parquet file. See write_json."""
        self._write(path, max_threads)

    # ── optimization ───────────────────────────────────────────────

    def optimize(
        self,
        *,
        eval_fn: Any = None,
        metric_key: str | None = None,
        models: list[str] | None = None,
        agent_model: str | None = None,
        max_iterations: int = 20,
        save_dir: str | None = None,
        exploration_weight: float = 1.414,
        dataset_path: str | None = None,
        max_threads: int | None = None,
        max_concurrent_agents: int = 3,
    ) -> Frame:
        """Optimize this pipeline and return an optimized Frame.

        Uses MOAR (Multi-Objective Agentic Rewrites) to search over
        model choices and prompt strategies. The best result from the
        Pareto frontier is returned as a new Frame you can ``.collect()``
        or ``.write_json()`` as usual.

        Access the full search results (Pareto frontier, costs, all plans)
        via ``frame.search_results`` after optimization.
        """
        from docetl.moar.optimizer import run_moar

        result = run_moar(
            self._build_config(),
            eval_fn=eval_fn,
            metric_key=metric_key,
            models=models,
            agent_model=agent_model or _config.agent_model,
            max_iterations=max_iterations,
            save_dir=save_dir,
            exploration_weight=exploration_weight,
            dataset_path=dataset_path,
            max_threads=max_threads or _config.max_threads,
            max_concurrent_agents=max_concurrent_agents,
        )

        self._total_cost = result.total_search_cost

        best = result.best()
        if best is None:
            raise RuntimeError("Optimization found no valid pipelines.")

        optimized = Frame.from_yaml(best.yaml_path)
        optimized._search_results = result
        optimized._total_cost = result.total_search_cost
        return optimized

    @property
    def search_results(self):
        """Access MOAR search results after calling ``.optimize()``.

        Returns ``None`` if this Frame was not created by optimization.
        """
        return getattr(self, "_search_results", None)

    # ── YAML loading ───────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> Frame:
        """Load a YAML pipeline config and return a Frame.

        Pipeline-level settings in the YAML (default_model, rate_limits,
        bypass_cache, fallbacks, parsing_tools, system_prompt) are carried
        on the returned Frame and take precedence over the ``docetl.<attr>``
        globals — loading a pipeline never changes process-wide settings.
        """
        from docetl.utils import load_config

        config = load_config(path)

        settings = {
            key: config[key] for key in _SETTING_KEYS if config.get(key) is not None
        }

        datasets = dict(config.get("datasets", {}))

        ops_by_name = {op["name"]: op for op in config.get("operations", [])}

        operations: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        last_step: str | None = None

        for step_cfg in config.get("pipeline", {}).get("steps", []):
            step: dict[str, Any] = {"name": step_cfg["name"], "operations": []}
            if "input" in step_cfg and step_cfg["input"] is not None:
                step["input"] = step_cfg["input"]

            for op_ref in step_cfg.get("operations", []):
                ref_name = op_ref_name(op_ref)
                if ref_name in ops_by_name:
                    operations.append(ops_by_name[ref_name])
                step["operations"].append(op_ref)

            steps.append(step)
            last_step = step["name"]

        return cls(
            datasets,
            operations,
            steps,
            _last_step=last_step,
            _retrievers=config.get("retrievers") or {},
            _settings=settings,
        )

    def to_yaml(self, path: str | None = None) -> str:
        """Export this pipeline as a YAML config string.

        If *path* is given, also writes the YAML to that file.
        """
        import yaml

        if _has_python_only_agent(self._operations):
            raise ValueError(
                "Agent tools are Python-only and cannot be exported to YAML."
            )
        config = self._build_config(output_path="output.json")
        out = yaml.dump(config, default_flow_style=False, sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(out)
        return out

    # ── code generation ────────────────────────────────────────────

    def to_python(self) -> str:
        """Generate Python source that recreates this pipeline using the Frame API."""
        if _has_python_only_agent(self._operations):
            raise ValueError(
                "Agent tools cannot be exported to Python source automatically; "
                "recreate the docetl.Agent and tools in Python code."
            )
        ops_by_name = {op["name"]: op for op in self._operations}
        steps_by_name = {s["name"]: s for s in self._steps}
        lines: list[str] = ["import docetl", ""]

        for attr in (
            "default_model",
            "agent_model",
            "max_threads",
            "bypass_cache",
            "rate_limits",
            "fallback_models",
            "fallback_embedding_models",
            "intermediate_dir",
        ):
            val = self._settings.get(attr, getattr(_config, attr, None))
            if val is not None and val is not False:
                lines.append(f"docetl.{attr} = {repr(val)}")
        dropped = [
            k for k in ("parsing_tools", "system_prompt") if self._settings.get(k)
        ]
        if dropped:
            lines.append(
                f"# NOTE: settings with no Frame API equivalent were dropped "
                f"(configure via YAML): {', '.join(dropped)}"
            )
        if lines[-1] != "":
            lines.append("")

        # Retriever objects, referenced by ops via retriever=<var>. A
        # retriever over an auxiliary dataset carries its data inline
        # (data=...); one over the input or a step output references it by
        # name (dataset=...).
        retriever_vars: dict[str, str] = {}
        retriever_datasets = set()
        for rname, rcfg in self._retrievers.items():
            var = re.sub(r"\W", "_", rname)
            retriever_vars[rname] = var
            ds_name = rcfg.get("dataset")
            parts = []
            if ds_name in self._datasets and ds_name != self._first_dataset:
                retriever_datasets.add(ds_name)
                parts.append(f"data={_fmt(self._datasets[ds_name].get('path'))}")
            else:
                parts.append(f"dataset={_fmt(ds_name)}")
            parts += [
                f"{k}={_fmt(rcfg[k])}"
                for k in (
                    "index_dir",
                    "index_types",
                    "fts",
                    "embedding",
                    "query",
                    "build_index",
                )
                if k in rcfg and not (k == "build_index" and rcfg[k] == "if_missing")
            ]
            lines.append(_format_call(f"{var} = docetl.Retriever(", parts))
        if self._retrievers:
            lines.append("")

        # Steps that feed the right side of an equijoin are rendered inline
        # as a nested expression, not as part of the main chain.
        right_branch: set[str] = set()
        branch_sources: set[str] = set()
        for step in self._steps:
            entries = step.get("operations", [])
            if entries and isinstance(entries[0], dict):
                cur = next(iter(entries[0].values())).get("right")
                while cur in steps_by_name and cur not in right_branch:
                    right_branch.add(cur)
                    cur = steps_by_name[cur].get("input")
                if cur in self._datasets:
                    branch_sources.add(cur)

        def entry_call(entry) -> tuple[str, list[str]] | None:
            """(method name, args) for one step-operations entry — the single
            place that renders an entry, used by both the main chain and
            inline branch expressions."""
            op = ops_by_name.get(op_ref_name(entry))
            if op is None:
                return None
            args = _format_op_args(op, retriever_vars)
            if isinstance(entry, str):
                return op.get("type", "map"), args
            right_expr = branch_expr(entry[op_ref_name(entry)].get("right"))
            return "equijoin", [right_expr] + args

        def branch_expr(tip: str | None) -> str:
            """One-line expression for the sub-pipeline that produces *tip*."""
            chain: list[dict] = []
            while tip in steps_by_name:
                chain.append(steps_by_name[tip])
                tip = steps_by_name[tip].get("input")
            expr = (
                _format_reader(tip, self._datasets[tip])
                if tip in self._datasets
                else "docetl.from_list([])"
            )
            for st in reversed(chain):
                for entry in st.get("operations", []):
                    call = entry_call(entry)
                    if call is not None:
                        expr += f".{call[0]}({', '.join(call[1])})"
            return expr

        source = (
            _format_reader(self._first_dataset, self._datasets[self._first_dataset])
            if self._first_dataset in self._datasets
            else "docetl.from_list([])"
        )
        lines.append("frame = (")
        lines.append(f"    {source}")

        for step in self._steps:
            if step["name"] in right_branch:
                continue
            for entry in step.get("operations", []):
                call = entry_call(entry)
                if call is not None:
                    lines.append(_format_call(f"    .{call[0]}(", call[1]))

        lines.append("    .collect()")
        lines.append(")")
        lines.append("")
        return "\n".join(lines)


# ── YAML to Python (standalone) ────────────────────────────────────


def yaml_to_python(yaml_path: str) -> str:
    """Convert a YAML pipeline config file to equivalent Python Frame code."""
    return Frame.from_yaml(yaml_path).to_python()


# ── reader entry points ────────────────────────────────────────────


def _read_file_frame(path: str, parsing: list[dict[str, str]] | None) -> Frame:
    """Create a Frame over a file dataset (format dispatched by extension)."""
    name = os.path.splitext(os.path.basename(path))[0]
    ds: dict[str, Any] = {"type": "file", "path": path}
    if parsing:
        ds["parsing"] = parsing
    return Frame({name: ds}, _first_dataset=name)


def read_json(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a JSON file and return a Frame."""
    return _read_file_frame(path, parsing)


def read_csv(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a CSV file and return a Frame."""
    return _read_file_frame(path, parsing)


def read_parquet(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read a Parquet file and return a Frame."""
    return _read_file_frame(path, parsing)


def read_dir(path: str, *, parsing: list[dict[str, str]] | None = None) -> Frame:
    """Read every file in a directory (recursively) as text.

    Each file becomes one row with ``filename``, ``path``, and ``text``
    keys, where ``text`` is the file's extracted content. PDF, Word, PowerPoint, and Excel files are converted to text;
    other files are read as UTF-8, and binary files with no extractor are
    skipped with a warning.
    """
    return _read_file_frame(path, parsing)


def from_list(data: list[dict], name: str = "data") -> Frame:
    """Create a Frame from an in-memory list of dicts."""
    return Frame({name: {"type": "memory", "path": data}}, _first_dataset=name)


# ── codegen formatting helpers ─────────────────────────────────────

_SKIP_KEYS = {"name", "type"}


def _has_python_only_agent(operations: list[dict[str, Any]]) -> bool:
    return any(op.get("agent") is not None for op in operations)


def _unique(name: str, taken) -> str:
    """Return *name* suffixed with the first free ``_<n>``."""
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _fmt(value: Any) -> str:
    if isinstance(value, str) and "\n" in value and "\r" not in value:
        # Triple-quote for readability, escaping anything that would change
        # the value or break the literal (backslashes, embedded/trailing
        # quote runs).
        body = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        if body.endswith('"'):
            body = body[:-1] + '\\"'
        return f'"""{body}"""'
    return repr(value)


def _format_reader(ds_name: str, ds_cfg: dict[str, Any]) -> str:
    is_memory = ds_cfg.get("type") == "memory"
    path = ds_cfg.get("path", "")

    if is_memory:
        return f"docetl.from_list({_fmt(path)}, name={repr(ds_name)})"

    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".csv":
        reader = "docetl.read_csv"
    elif ext == ".parquet":
        reader = "docetl.read_parquet"
    elif ext == "":
        reader = "docetl.read_dir"
    else:
        reader = "docetl.read_json"

    parts = [repr(path)]
    parsing = ds_cfg.get("parsing")
    if parsing:
        parts.append(f"parsing={_fmt(parsing)}")

    return f"{reader}({', '.join(parts)})"


def _format_op_args(
    op: dict[str, Any], retriever_vars: dict[str, str] | None = None
) -> list[str]:
    """The argument list for a Frame method call recreating *op*.

    *retriever_vars* maps retriever names to the variable names of the
    ``docetl.Retriever`` objects emitted earlier in the generated source.
    """
    parts = [repr(op["name"])]
    for k, v in op.items():
        if k in _SKIP_KEYS or v is None:
            continue
        if k == "retriever" and retriever_vars and v in retriever_vars:
            parts.append(f"retriever={retriever_vars[v]}")
        else:
            parts.append(f"{k}={_fmt(v)}")
    return parts


def _format_call(prefix: str, parts: list[str]) -> str:
    """Render a call, wrapping its arguments when the line would run long."""
    joined = ", ".join(parts)
    if len(prefix) + len(joined) + 1 <= 100:
        return f"{prefix}{joined})"
    indent = " " * len(prefix)
    return prefix + f",\n{indent}".join(parts) + ")"
