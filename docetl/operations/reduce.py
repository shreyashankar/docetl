"""
Implements a reduce operation on input data using language models.

Extends BaseOperation to reduce grouped data using batch, incremental, and parallel strategies.

Manages performance metrics and dynamically adjusts processing (i.e., number of parallel folds) based on these metrics.
"""

import math
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

import jinja2
import numpy as np
from jinja2 import Template
from litellm.utils import ModelResponse
from pydantic import Field, field_validator, model_validator

from docetl.execution import (
    SUPPORTED_LITELLM_BATCH_PROVIDERS,
    ExecutionError,
    ExecutionPolicy,
    ExecutionTarget,
)
from docetl.operations.base import BaseOperation, Cardinality
from docetl.operations.clustering_utils import (
    cluster_documents,
    get_embeddings_for_clustering,
)
from docetl.operations.utils import (
    lookup_field,
    rich_as_completed,
    strict_render,
    validate_output_types,
)

# Import OutputMode enum for structured output checks
from docetl.operations.utils.api import OutputMode
from docetl.utils import (
    completion_cost,
    ensure_non_jinja_prompt_confirmed,
    has_jinja_syntax,
)


class ReduceOperation(BaseOperation):
    """
    A class that implements a reduce operation on input data using language models.

    This class extends BaseOperation to provide functionality for reducing grouped data
    using various strategies including batch reduce, incremental reduce, and parallel fold and merge.
    """

    class schema(BaseOperation.schema):
        type: str = "reduce"
        reduce_key: str | list[str]
        output: dict[str, Any]
        prompt: str
        optimize: bool | None = None
        synthesize_resolve: bool | None = None
        model: str | None = None
        input: dict[str, Any] | None = None
        pass_through: bool | None = None
        associative: bool | None = None
        fold_prompt: str | None = None
        fold_batch_size: int | None = Field(None, gt=0)
        merge_prompt: str | None = None
        merge_batch_size: int | None = Field(None, gt=0)
        value_sampling: dict[str, Any] | None = None
        verbose: bool | None = None
        timeout: int | None = None
        litellm_completion_kwargs: dict[str, Any] = Field(default_factory=dict)
        agent: Any | None = None
        enable_observability: bool = False
        limit: int | None = Field(None, gt=0)
        execution: dict[str, Any] | None = None

        @field_validator("prompt")
        def validate_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    return v
                try:
                    template = Template(v)
                    template_vars = template.environment.parse(v).find_all(
                        jinja2.nodes.Name
                    )
                    template_var_names = {var.name for var in template_vars}
                    if "inputs" not in template_var_names:
                        raise ValueError(
                            "Prompt template must include the 'inputs' variable"
                        )
                except Exception as e:
                    raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}")
            return v

        @field_validator("fold_prompt")
        def validate_fold_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    return v
                try:
                    fold_template = Template(v)
                    fold_template_vars = fold_template.environment.parse(v).find_all(
                        jinja2.nodes.Name
                    )
                    fold_template_var_names = {var.name for var in fold_template_vars}
                    required_vars = {"inputs", "output"}
                    if not required_vars.issubset(fold_template_var_names):
                        raise ValueError(
                            f"Fold template must include variables: {required_vars}. Current template includes: {fold_template_var_names}"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'fold_prompt': {str(e)}"
                    )
            return v

        @field_validator("merge_prompt")
        def validate_merge_prompt(cls, v):
            if v is not None:
                # Check if it has Jinja syntax
                if not has_jinja_syntax(v):
                    # This will be handled during initialization with user confirmation
                    return v
                try:
                    merge_template = Template(v)
                    merge_template_vars = merge_template.environment.parse(v).find_all(
                        jinja2.nodes.Name
                    )
                    merge_template_var_names = {var.name for var in merge_template_vars}
                    if "outputs" not in merge_template_var_names:
                        raise ValueError(
                            "Merge template must include the 'outputs' variable"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Invalid Jinja2 template in 'merge_prompt': {str(e)}"
                    )
            return v

        @field_validator("value_sampling")
        def validate_value_sampling(cls, v):
            if v is not None:
                if v["enabled"]:
                    if v["method"] not in ["random", "first_n", "cluster", "sem_sim"]:
                        raise ValueError(
                            "Invalid 'method'. Must be 'random', 'first_n', 'cluster', or 'sem_sim'"
                        )

                    if v["method"] == "embedding":
                        if "embedding_model" not in v:
                            raise ValueError(
                                "'embedding_model' is required when using embedding-based sampling"
                            )
                        if "embedding_keys" not in v:
                            raise ValueError(
                                "'embedding_keys' is required when using embedding-based sampling"
                            )
            return v

        @model_validator(mode="after")
        def validate_complex_requirements(self):
            if self.agent is not None and self.gleaning is not None:
                raise ValueError("Agentic operations cannot be combined with gleaning")
            # Check dependencies between merge_prompt and fold_prompt
            if self.merge_prompt and not self.fold_prompt:
                raise ValueError(
                    "'fold_prompt' is required when 'merge_prompt' is specified"
                )

            # Check batch size requirements
            if self.fold_prompt and not self.fold_batch_size:
                raise ValueError(
                    "'fold_batch_size' is required when 'fold_prompt' is specified"
                )
            if self.merge_prompt and not self.merge_batch_size:
                raise ValueError(
                    "'merge_batch_size' is required when 'merge_prompt' is specified"
                )

            return self

        @model_validator(mode="after")
        def validate_execution(self):
            if self.execution is None:
                return self
            policy = ExecutionPolicy.from_config(self.execution)
            if (
                policy.target is ExecutionTarget.PROVIDER_BATCH
                and self.execution.get("provider", "openai")
                not in SUPPORTED_LITELLM_BATCH_PROVIDERS
            ):
                raise ValueError(
                    "LiteLLM batch execution currently supports provider values "
                    + ", ".join(sorted(SUPPORTED_LITELLM_BATCH_PROVIDERS))
                )
            incompatible = [
                name
                for name, value in (
                    ("agent", self.agent),
                    ("gleaning", self.gleaning),
                    (
                        "validate",
                        self.model_extra.get("validate") if self.model_extra else None,
                    ),
                    ("fold_prompt", self.fold_prompt),
                    ("merge_prompt", self.merge_prompt),
                )
                if value
            ]
            if incompatible:
                raise ValueError(
                    "compiled reduce execution issues one request per group "
                    "and cannot be combined with " + ", ".join(incompatible)
                )
            return self

    # ── plan traits ────────────────────────────────────────────────
    # fields_read/fields_written stay at the conservative None default:
    # reduce prompts render whole grouped rows (``{{ inputs }}``) and
    # output rows are reshaped wholesale.

    @classmethod
    def cardinality(cls, config: dict[str, Any]) -> Cardinality:
        return Cardinality.MANY_TO_ONE

    @classmethod
    def is_llm(cls, config: dict[str, Any]) -> bool:
        return True

    def __init__(self, *args, **kwargs):
        """Initialize the ReduceOperation."""
        super().__init__(*args, **kwargs)
        self.min_samples = 5
        self.max_samples = 1000
        self.fold_times = deque(maxlen=self.max_samples)
        self.merge_times = deque(maxlen=self.max_samples)
        self.lock = Lock()
        self.config["reduce_key"] = (
            [self.config["reduce_key"]]
            if isinstance(self.config["reduce_key"], str)
            else self.config["reduce_key"]
        )
        self.intermediates = {}
        self.lineage_keys = self.config.get("output", {}).get("lineage", [])
        ensure_non_jinja_prompt_confirmed(
            self.config,
            "prompt",
            "_append_document_to_prompt",
            console=self.console,
            status=self.status,
            extra_flags={"_is_reduce_operation": True},
        )
        ensure_non_jinja_prompt_confirmed(
            self.config,
            "fold_prompt",
            "_append_document_to_fold_prompt",
            console=self.console,
            status=self.status,
            extra_flags={"_is_reduce_operation": True},
        )
        ensure_non_jinja_prompt_confirmed(
            self.config,
            "merge_prompt",
            "_append_document_to_merge_prompt",
            console=self.console,
            status=self.status,
            extra_flags={"_is_reduce_operation": True},
        )

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        """
        Execute the reduce operation on the provided input data.

        This method sorts and groups the input data by the reduce key(s), then processes each group
        using either parallel fold and merge, incremental reduce, or batch reduce strategies.

        Args:
            input_data (list[dict]): The input data to process.

        Returns:
            tuple[list[dict], float]: A tuple containing the processed results and the total cost of the operation.
        """
        if self.config.get("gleaning", {}).get("validation_prompt", None):
            self.console.log(
                f"Using gleaning with validation prompt: {self.config.get('gleaning', {}).get('validation_prompt', '')}"
            )

        reduce_keys = self.config["reduce_key"]
        if isinstance(reduce_keys, str):
            reduce_keys = [reduce_keys]
        input_schema = self.config.get("input", {}).get("schema", {})

        if self.status:
            self.status.stop()

        # Check if we need to group everything into one group
        if reduce_keys == ["_all"] or reduce_keys == "_all":
            grouped_data = [("_all", input_data)]
        else:
            # Group the input data by the reduce key(s) while maintaining original order
            def get_group_key(item):
                key_values = []
                for key in reduce_keys:
                    value = lookup_field(item, key)
                    # Special handling for list-type values
                    if isinstance(value, list):
                        key_values.append(
                            tuple(sorted(value))
                        )  # Convert list to sorted tuple
                    else:
                        key_values.append(value)
                return tuple(key_values)

            grouped_data = {}
            for item in input_data:
                key = get_group_key(item)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(item)

            # Convert the grouped data to a list of tuples
            grouped_data = list(grouped_data.items())

        limit_value = self.config.get("limit")
        if limit_value is not None:
            # Sort by group size (smallest first) and take the limit
            grouped_data = sorted(grouped_data, key=lambda x: len(x[1]))
            grouped_data = grouped_data[:limit_value]

        if self.config.get("execution") is not None:
            try:
                return self._execute_compiled_reduce(grouped_data, input_schema)
            finally:
                if self.status:
                    self.status.start()

        def process_group(
            key: tuple, group_elems: list[dict]
        ) -> tuple[dict | None, float]:
            group_list, retrieval_context, total_cost = self._prepare_reduce_group(
                key, group_elems, input_schema
            )

            # Only execute merge-based plans if associative = True
            if "merge_prompt" in self.config and self.config.get("associative", True):
                result, prompts, cost = self._parallel_fold_and_merge(
                    key, group_list, retrieval_context
                )
            elif self.config.get("fold_batch_size", None) and self.config.get(
                "fold_batch_size"
            ) >= len(group_list):
                # If the fold batch size is greater than or equal to the number of items in the group,
                # we can just run a single fold operation
                result, prompt, cost = self._batch_reduce(
                    key, group_list, None, retrieval_context
                )
                prompts = [prompt]
            elif "fold_prompt" in self.config:
                result, prompts, cost = self._incremental_reduce(
                    key, group_list, retrieval_context
                )
            else:
                result, prompt, cost = self._batch_reduce(
                    key, group_list, None, retrieval_context
                )
                prompts = [prompt]

            total_cost += cost
            if result is None:
                return None, total_cost
            return (
                self._decorate_reduce_result(
                    result, group_elems, retrieval_context, prompts
                ),
                total_cost,
            )

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]
            results = []
            total_cost = 0
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc=f"Processing {self.config['name']} (reduce) on all documents",
                leave=True,
                console=self.console,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if output is not None:
                    results.append(output)

        if limit_value is not None and len(results) > limit_value:
            results = results[:limit_value]

        if self.config.get("persist_intermediates", False):
            for result in results:
                key = tuple(result[k] for k in self.config["reduce_key"])
                if key in self.intermediates:
                    result[f"_{self.config['name']}_intermediates"] = (
                        self.intermediates[key]
                    )

        if self.status:
            self.status.start()

        return results, total_cost

    def _prepare_reduce_group(
        self,
        key: tuple,
        group_elems: list[dict],
        input_schema: dict[str, Any],
    ) -> tuple[list[dict], str, float]:
        """Apply input projection, retrieval, and value sampling for one group."""
        if input_schema:
            group_list = [
                {k: item[k] for k in input_schema.keys() if k in item}
                for item in group_elems
            ]
        else:
            group_list = group_elems

        total_cost = 0.0
        # Build retrieval context once per group
        try:
            retrieval_context = self._maybe_build_retrieval_context(
                {
                    "reduce_key": dict(zip(self.config["reduce_key"], key)),
                    "inputs": group_list,
                }
            )
        except Exception:
            retrieval_context = "No extra context available."

        # Apply value sampling if enabled
        value_sampling = self.config.get("value_sampling", {})
        if value_sampling.get("enabled", False):
            sample_size = min(value_sampling["sample_size"], len(group_list))
            method = value_sampling["method"]

            if method == "random":
                group_sample = random.sample(group_list, sample_size)
                group_sample.sort(key=lambda x: group_list.index(x))
            elif method == "first_n":
                group_sample = group_list[:sample_size]
            elif method == "cluster":
                group_sample, embedding_cost = self._cluster_based_sampling(
                    group_list, value_sampling, sample_size
                )
                group_sample.sort(key=lambda x: group_list.index(x))
                total_cost += embedding_cost
            elif method == "sem_sim":
                group_sample, embedding_cost = self._semantic_similarity_sampling(
                    key, group_list, value_sampling, sample_size
                )
                group_sample.sort(key=lambda x: group_list.index(x))
                total_cost += embedding_cost

            group_list = group_sample
        return group_list, retrieval_context, total_cost

    def _decorate_reduce_result(
        self,
        result: dict,
        group_elems: list[dict],
        retrieval_context: str,
        prompts: list[str],
    ) -> dict:
        """Attach metadata shared by direct and compiled reduce execution."""
        # Add the counts of items in the group to the result
        result[f"_counts_prereduce_{self.config['name']}"] = len(group_elems)

        if self.config.get("enable_observability", False):
            # Add the _observability_{self.config['name']} key to the result
            result[f"_observability_{self.config['name']}"] = {"prompts": prompts}

        # Add retrieved context if save_retriever_output is enabled
        if self.config.get("save_retriever_output", False):
            ctx = (
                retrieval_context
                if retrieval_context
                and retrieval_context != "No extra context available."
                else ""
            )
            result[f"_{self.config['name']}_retrieved_context"] = ctx

        # Apply pass-through at the group level
        if self.config.get("pass_through", False) and group_elems:
            for k, v in group_elems[0].items():
                if k not in self.config["output"]["schema"] and k not in result:
                    result[k] = v

        # Add lineage information
        if self.lineage_keys:
            lineage = []
            for item in group_elems:
                lineage_item = {k: item.get(k) for k in self.lineage_keys if k in item}
                if lineage_item:
                    lineage.append(lineage_item)
            result[f"{self.config['name']}_lineage"] = lineage

        return result

    def _execute_compiled_reduce(
        self,
        grouped_data: list[tuple[Any, list[dict]]],
        input_schema: dict[str, Any],
    ) -> tuple[list[dict], float]:
        """Compile one request per reduce group and execute them as one plan."""
        prepared: list[tuple[Any, list[dict], str, str, float]] = []
        calls: list[dict[str, Any]] = []
        model = self.config.get("model", self.default_model)

        for key, group_elems in grouped_data:
            group_list, retrieval_context, local_cost = self._prepare_reduce_group(
                key, group_elems, input_schema
            )

            prompt = strict_render(
                self.config["prompt"],
                {
                    "reduce_key": dict(zip(self.config["reduce_key"], key)),
                    "inputs": group_list,
                    "retrieval_context": retrieval_context or "",
                },
            )
            if retrieval_context and "retrieval_context" not in self.config["prompt"]:
                prompt = f"Here is some extra context:\n{retrieval_context}\n\n{prompt}"
            prepared.append((key, group_elems, retrieval_context, prompt, local_cost))
            calls.append(
                {
                    "model": model,
                    "op_type": "reduce",
                    "messages": [{"role": "user", "content": prompt}],
                    "output_schema": self.config["output"]["schema"],
                    "bypass_cache": self.config.get("bypass_cache", self.bypass_cache),
                    "litellm_completion_kwargs": self.config.get(
                        "litellm_completion_kwargs", {}
                    ),
                    "op_config": self.config,
                }
            )

        llm_results = self.runner.api.call_llm_compiled(calls, self.config["execution"])
        structured_mode = (
            self.config.get("output", {}).get("mode")
            == OutputMode.STRUCTURED_OUTPUT.value
        )
        results: list[dict] = []
        total_cost = sum(item[-1] for item in prepared)
        for prepared_group, llm_result in zip(prepared, llm_results):
            key, group_elems, retrieval_context, prompt, _ = prepared_group
            total_cost += llm_result.total_cost
            if not llm_result.validated or llm_result.response is None:
                if self.config.get("skip_on_error", False):
                    continue
                raise ExecutionError(
                    f"Compiled request failed in reduce operation "
                    f"{self.config['name']!r} for key {key!r}"
                )
            parsed_outputs = self.runner.api.parse_llm_response(
                llm_result.response,
                schema=self.config["output"]["schema"],
                manually_fix_errors=self.manually_fix_errors,
                use_structured_output=structured_mode,
            )
            if len(parsed_outputs) != 1:
                raise ExecutionError(
                    "A compiled reduce request must produce exactly one output"
                )
            result = parsed_outputs[0]
            result.update(dict(zip(self.config["reduce_key"], key)))
            results.append(
                self._decorate_reduce_result(
                    result, group_elems, retrieval_context, [prompt]
                )
            )

        return results, total_cost

    def _cluster_based_sampling(
        self, group_list: list[dict], value_sampling: dict, sample_size: int
    ) -> tuple[list[dict], float]:
        if sample_size >= len(group_list):
            return group_list, 0

        clusters, cost = cluster_documents(
            group_list, value_sampling, sample_size, self.runner.api
        )

        sampled_items = []
        idx_added_already = set()
        num_clusters = len(clusters)
        for i in range(sample_size):
            # Add a random item from the cluster
            idx = i % num_clusters

            # Skip if there are no items in the cluster
            if len(clusters[idx]) == 0:
                continue

            if len(clusters[idx]) == 1:
                # If there's only one item in the cluster, add it directly if we haven't already
                if idx not in idx_added_already:
                    sampled_items.append(clusters[idx][0])
                continue

            random_choice_idx = random.randint(0, len(clusters[idx]) - 1)
            max_attempts = 10
            while random_choice_idx in idx_added_already and max_attempts > 0:
                random_choice_idx = random.randint(0, len(clusters[idx]) - 1)
                max_attempts -= 1
            idx_added_already.add(random_choice_idx)
            sampled_items.append(clusters[idx][random_choice_idx])

        return sampled_items, cost

    def _semantic_similarity_sampling(
        self, key: tuple, group_list: list[dict], value_sampling: dict, sample_size: int
    ) -> tuple[list[dict], float]:
        embedding_model = value_sampling["embedding_model"]
        query_text = strict_render(
            value_sampling["query_text"],
            {"reduce_key": dict(zip(self.config["reduce_key"], key))},
        )

        embeddings, cost = get_embeddings_for_clustering(
            group_list, value_sampling, self.runner.api
        )

        query_response = self.runner.api.gen_embedding(embedding_model, [query_text])
        query_embedding = query_response["data"][0]["embedding"]
        cost += completion_cost(query_response)

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity([query_embedding], embeddings)[0]

        top_k_indices = np.argsort(similarities)[-sample_size:]

        return [group_list[i] for i in top_k_indices], cost

    def _parallel_fold_and_merge(
        self, key: tuple, group_list: list[dict], retrieval_context: str
    ) -> tuple[dict | None, float]:
        """
        Perform parallel folding and merging on a group of items.

        This method implements a strategy that combines parallel folding of input items
        and merging of intermediate results to efficiently process large groups. It works as follows:
        1. The input group is initially divided into smaller batches for efficient processing.
        2. The method performs an initial round of folding operations on these batches.
        3. After the first round of folds, a few merges are performed to estimate the merge runtime.
        4. Based on the estimated merge runtime and observed fold runtime, it calculates the optimal number of parallel folds. Subsequent rounds of folding are then performed concurrently, with the number of parallel folds determined by the runtime estimates.
        5. The folding process repeats in rounds, progressively reducing the number of items to be processed.
        6. Once all folding operations are complete, the method recursively performs final merges on the fold results to combine them into a final result.
        7. Throughout this process, the method may adjust the number of parallel folds based on updated performance metrics (i.e., fold and merge runtimes) to maintain efficiency.

        Args:
            key (tuple): The reduce key tuple for the group.
            group_list (list[dict]): The list of items in the group to be processed.

        Returns:
            tuple[dict | None, float]: A tuple containing the final merged result (or None if processing failed)
            and the total cost of the operation.
        """
        fold_batch_size = self.config["fold_batch_size"]
        merge_batch_size = self.config["merge_batch_size"]
        total_cost = 0
        prompts = []

        def calculate_num_parallel_folds():
            fold_time, fold_default = self.get_fold_time()
            merge_time, merge_default = self.get_merge_time()
            num_group_items = len(group_list)
            return (
                max(
                    1,
                    int(
                        (fold_time * num_group_items * math.log(merge_batch_size))
                        / (fold_batch_size * merge_time)
                    ),
                ),
                fold_default or merge_default,
            )

        num_parallel_folds, used_default_times = calculate_num_parallel_folds()
        fold_results = []
        remaining_items = group_list

        if self.config.get("persist_intermediates", False):
            self.intermediates[key] = []
            iter_count = 0

        # Parallel folding and merging
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            while remaining_items:
                # Folding phase
                fold_futures = []
                for i in range(min(num_parallel_folds, len(remaining_items))):
                    batch = remaining_items[:fold_batch_size]
                    remaining_items = remaining_items[fold_batch_size:]
                    current_output = fold_results[i] if i < len(fold_results) else None
                    fold_futures.append(
                        executor.submit(
                            self._increment_fold, key, batch, current_output
                        )
                    )

                new_fold_results = []
                for future in as_completed(fold_futures):
                    result, prompt, cost = future.result()
                    total_cost += cost
                    prompts.append(prompt)
                    if result is not None:
                        new_fold_results.append(result)
                        if self.config.get("persist_intermediates", False):
                            self.intermediates[key].append(
                                {
                                    "iter": iter_count,
                                    "intermediate": result,
                                    "scratchpad": result["updated_scratchpad"],
                                }
                            )
                            iter_count += 1

                # Update fold_results with new results
                fold_results = new_fold_results + fold_results[len(new_fold_results) :]

                # Single pass merging phase
                if (
                    len(self.merge_times) < self.min_samples
                    and len(fold_results) >= merge_batch_size
                ):
                    merge_futures = []
                    for i in range(0, len(fold_results), merge_batch_size):
                        batch = fold_results[i : i + merge_batch_size]
                        merge_futures.append(
                            executor.submit(self._merge_results, key, batch)
                        )

                    new_results = []
                    for future in as_completed(merge_futures):
                        result, prompt, cost = future.result()
                        total_cost += cost
                        prompts.append(prompt)
                        if result is not None:
                            new_results.append(result)
                            if self.config.get("persist_intermediates", False):
                                self.intermediates[key].append(
                                    {
                                        "iter": iter_count,
                                        "intermediate": result,
                                        "scratchpad": None,
                                    }
                                )
                                iter_count += 1

                    fold_results = new_results

                # Recalculate num_parallel_folds if we used default times
                if used_default_times:
                    new_num_parallel_folds, used_default_times = (
                        calculate_num_parallel_folds()
                    )
                    if not used_default_times:
                        self.console.log(
                            f"Recalculated num_parallel_folds from {num_parallel_folds} to {new_num_parallel_folds}"
                        )
                        num_parallel_folds = new_num_parallel_folds

        # Final merging if needed
        while len(fold_results) > 1:
            self.console.log(f"Finished folding! Merging {len(fold_results)} items.")
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                merge_futures = []
                for i in range(0, len(fold_results), merge_batch_size):
                    batch = fold_results[i : i + merge_batch_size]
                    merge_futures.append(
                        executor.submit(self._merge_results, key, batch)
                    )

                new_results = []
                for future in as_completed(merge_futures):
                    result, prompt, cost = future.result()
                    total_cost += cost
                    prompts.append(prompt)
                    if result is not None:
                        new_results.append(result)
                        if self.config.get("persist_intermediates", False):
                            self.intermediates[key].append(
                                {
                                    "iter": iter_count,
                                    "intermediate": result,
                                    "scratchpad": None,
                                }
                            )
                            iter_count += 1

                fold_results = new_results

        return (
            (fold_results[0], prompts, total_cost)
            if fold_results
            else (None, prompts, total_cost)
        )

    def _incremental_reduce(
        self, key: tuple, group_list: list[dict], retrieval_context: str
    ) -> tuple[dict | None, list[str], float]:
        """
        Perform an incremental reduce operation on a group of items.

        This method processes the group in batches, incrementally folding the results.

        Args:
            key (tuple): The reduce key tuple for the group.
            group_list (list[dict]): The list of items in the group to be processed.

        Returns:
            tuple[dict | None, list[str], float]: A tuple containing the final reduced result (or None if processing failed),
            the list of prompts used, and the total cost of the operation.
        """
        fold_batch_size = self.config["fold_batch_size"]
        total_cost = 0
        current_output = None
        prompts = []

        # Calculate and log the number of folds to be performed
        num_folds = (len(group_list) + fold_batch_size - 1) // fold_batch_size

        scratchpad = ""
        if self.config.get("persist_intermediates", False):
            self.intermediates[key] = []
            iter_count = 0

        for i in range(0, len(group_list), fold_batch_size):
            # Log the current iteration and total number of folds
            current_fold = i // fold_batch_size + 1
            if self.config.get("verbose", False):
                self.console.log(
                    f"Processing fold {current_fold} of {num_folds} for group with key {key}"
                )
            batch = group_list[i : i + fold_batch_size]

            folded_output, prompt, fold_cost = self._increment_fold(
                key, batch, current_output, scratchpad
            )
            total_cost += fold_cost
            prompts.append(prompt)

            if folded_output is None:
                continue

            if self.config.get("persist_intermediates", False):
                self.intermediates[key].append(
                    {
                        "iter": iter_count,
                        "intermediate": folded_output,
                        "scratchpad": folded_output.get("updated_scratchpad", ""),
                    }
                )
                iter_count += 1

            # Pop off updated_scratchpad
            if "updated_scratchpad" in folded_output:
                scratchpad = folded_output["updated_scratchpad"]
                if self.config.get("verbose", False):
                    self.console.log(
                        f"Updated scratchpad for fold {current_fold}: {scratchpad}"
                    )
                del folded_output["updated_scratchpad"]

            current_output = folded_output

        return current_output, prompts, total_cost

    def validation_fn(self, response: dict[str, Any]):
        structured_mode = (
            self.config.get("output", {}).get("mode")
            == OutputMode.STRUCTURED_OUTPUT.value
        )
        output = (
            self.runner.api.parse_llm_response(
                response,
                schema=self.config["output"]["schema"],
                use_structured_output=structured_mode,
            )[0]
            if isinstance(response, ModelResponse)
            else response
        )
        # Enforce type validation against output schema
        is_types_valid, _errors = validate_output_types(
            output,
            self.config["output"]["schema"],
        )
        if not is_types_valid:
            return output, False
        if self.runner.api.validate_output(self.config, output, self.console):
            return output, True
        return output, False

    def _parse_reduce_response(self, response: Any) -> dict[str, Any]:
        if not isinstance(response, ModelResponse):
            return response
        structured_mode = (
            self.config.get("output", {}).get("mode")
            == OutputMode.STRUCTURED_OUTPUT.value
        )
        return self.runner.api.parse_llm_response(
            response,
            schema=self.config["output"]["schema"],
            manually_fix_errors=self.manually_fix_errors,
            use_structured_output=structured_mode,
        )[0]

    def _increment_fold(
        self,
        key: tuple,
        batch: list[dict],
        current_output: dict | None,
        scratchpad: str | None = None,
        retrieval_context: str | None = None,
    ) -> tuple[dict | None, str, float]:
        """
        Perform an incremental fold operation on a batch of items.

        This method folds a batch of items into the current output using the fold prompt.

        Args:
            key (tuple): The reduce key tuple for the group.
            batch (list[dict]): The batch of items to be folded.
            current_output (dict | None): The current accumulated output, if any.
            scratchpad (str | None): The scratchpad to use for the fold operation.
        Returns:
            tuple[dict | None, str, float]: A tuple containing the folded output (or None if processing failed),
            the prompt used, and the cost of the fold operation.
        """
        if current_output is None:
            return self._batch_reduce(key, batch, scratchpad, retrieval_context)

        start_time = time.time()
        fold_prompt = strict_render(
            self.config["fold_prompt"],
            {
                "inputs": batch,
                "output": current_output,
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
                "retrieval_context": retrieval_context or "",
            },
        )
        if retrieval_context and "retrieval_context" not in self.config.get(
            "fold_prompt", ""
        ):
            fold_prompt = (
                f"Here is some extra context:\n{retrieval_context}\n\n{fold_prompt}"
            )

        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            [{"role": "user", "content": fold_prompt}],
            self.config["output"]["schema"],
            scratchpad=scratchpad,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
            ),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
            agent_config=self.config.get("agent"),
        )

        end_time = time.time()
        self._update_fold_time(end_time - start_time)
        fold_cost = response.total_cost

        if response.validated:
            folded_output = self._parse_reduce_response(response.response)

            folded_output.update(dict(zip(self.config["reduce_key"], key)))
            fold_cost = response.total_cost

            return folded_output, fold_prompt, fold_cost

        return None, fold_prompt, fold_cost

    def _merge_results(
        self, key: tuple, outputs: list[dict], retrieval_context: str | None = None
    ) -> tuple[dict | None, str, float]:
        """
        Merge multiple outputs into a single result.

        This method merges a list of outputs using the merge prompt.

        Args:
            key (tuple): The reduce key tuple for the group.
            outputs (list[dict]): The list of outputs to be merged.

        Returns:
            tuple[dict | None, str, float]: A tuple containing the merged output (or None if processing failed),
            the prompt used, and the cost of the merge operation.
        """
        start_time = time.time()
        merge_prompt = strict_render(
            self.config["merge_prompt"],
            {
                "outputs": outputs,
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
                "retrieval_context": retrieval_context or "",
            },
        )
        if retrieval_context and "retrieval_context" not in self.config.get(
            "merge_prompt", ""
        ):
            merge_prompt = (
                f"Here is some extra context:\n{retrieval_context}\n\n{merge_prompt}"
            )
        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "merge",
            [{"role": "user", "content": merge_prompt}],
            self.config["output"]["schema"],
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
            agent_config=self.config.get("agent"),
        )

        end_time = time.time()
        self._update_merge_time(end_time - start_time)
        merge_cost = response.total_cost

        if response.validated:
            merged_output = self._parse_reduce_response(response.response)
            merged_output.update(dict(zip(self.config["reduce_key"], key)))
            merge_cost = response.total_cost
            return merged_output, merge_prompt, merge_cost

        return None, merge_prompt, merge_cost

    def get_fold_time(self) -> tuple[float, bool]:
        """
        Get the average fold time or a default value.

        Returns:
            tuple[float, bool]: A tuple containing the average fold time (or default) and a boolean
            indicating whether the default value was used.
        """
        if "fold_time" in self.config:
            return self.config["fold_time"], False
        with self.lock:
            if len(self.fold_times) >= self.min_samples:
                return sum(self.fold_times) / len(self.fold_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def get_merge_time(self) -> tuple[float, bool]:
        """
        Get the average merge time or a default value.

        Returns:
            tuple[float, bool]: A tuple containing the average merge time (or default) and a boolean
            indicating whether the default value was used.
        """
        if "merge_time" in self.config:
            return self.config["merge_time"], False
        with self.lock:
            if len(self.merge_times) >= self.min_samples:
                return sum(self.merge_times) / len(self.merge_times), False
        return 1.0, True  # Default to 1 second if no data is available

    def _update_fold_time(self, time: float) -> None:
        """
        Update the fold time statistics.

        Args:
            time (float): The time taken for a fold operation.
        """
        with self.lock:
            self.fold_times.append(time)

    def _update_merge_time(self, time: float) -> None:
        """
        Update the merge time statistics.

        Args:
            time (float): The time taken for a merge operation.
        """
        with self.lock:
            self.merge_times.append(time)

    def _batch_reduce(
        self,
        key: tuple,
        group_list: list[dict],
        scratchpad: str | None = None,
        retrieval_context: str | None = None,
    ) -> tuple[dict | None, str, float]:
        """
        Perform a batch reduce operation on a group of items.

        This method reduces a group of items into a single output using the reduce prompt.

        Args:
            key (tuple): The reduce key tuple for the group.
            group_list (list[dict]): The list of items to be reduced.
            scratchpad (str | None): The scratchpad to use for the reduce operation.
        Returns:
            tuple[dict | None, str, float]: A tuple containing the reduced output (or None if processing failed),
            the prompt used, and the cost of the reduce operation.
        """
        prompt = strict_render(
            self.config["prompt"],
            {
                "reduce_key": dict(zip(self.config["reduce_key"], key)),
                "inputs": group_list,
                "retrieval_context": retrieval_context or "",
            },
        )
        if retrieval_context and "retrieval_context" not in self.config.get(
            "prompt", ""
        ):
            prompt = f"Here is some extra context:\n{retrieval_context}\n\n{prompt}"
        item_cost = 0

        response = self.runner.api.call_llm(
            self.config.get("model", self.default_model),
            "reduce",
            [{"role": "user", "content": prompt}],
            self.config["output"]["schema"],
            scratchpad=scratchpad,
            timeout_seconds=self.config.get("timeout", 120),
            max_retries_per_timeout=self.config.get("max_retries_per_timeout", 2),
            bypass_cache=self.config.get("bypass_cache", self.bypass_cache),
            validation_config=(
                {
                    "num_retries": self.num_retries_on_validate_failure,
                    "val_rule": self.config.get("validate", []),
                    "validation_fn": self.validation_fn,
                }
                if self.config.get("validate", None)
                else None
            ),
            gleaning_config=self.config.get("gleaning", None),
            verbose=self.config.get("verbose", False),
            litellm_completion_kwargs=self.config.get("litellm_completion_kwargs", {}),
            op_config=self.config,
            agent_config=self.config.get("agent"),
        )

        item_cost += response.total_cost

        if response.validated:
            output = self._parse_reduce_response(response.response)
            output.update(dict(zip(self.config["reduce_key"], key)))

            return output, prompt, item_cost
        return None, prompt, item_cost
