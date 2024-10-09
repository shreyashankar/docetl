"""
The `EquijoinOperation` class is a subclass of `BaseOperation` that performs an equijoin operation on two datasets. It uses a combination of blocking techniques and LLM-based comparisons to efficiently join the datasets.
"""

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import numpy as np
from jinja2 import Template
from litellm import model_cost
from rich.prompt import Confirm

from docetl.operations.base import BaseOperation
from docetl.operations.utils import (
    rich_as_completed,
)
from docetl.utils import completion_cost

# Global variables to store shared data
_right_data = None
_blocking_conditions = None


def init_worker(right_data, blocking_conditions):
    global _right_data, _blocking_conditions
    _right_data = right_data
    _blocking_conditions = blocking_conditions


def is_match(left_item: Dict[str, Any], right_item: Dict[str, Any]) -> bool:
    return any(
        eval(condition, {"left": left_item, "right": right_item})
        for condition in _blocking_conditions
    )


def process_left_item(
    left_item: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return [
        (left_item, right_item)
        for right_item in _right_data
        if is_match(left_item, right_item)
    ]


class EquijoinOperation(BaseOperation):
    def compare_pair(
        self,
        comparison_prompt: str,
        model: str,
        item1: Dict,
        item2: Dict,
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
    ) -> Tuple[bool, float]:
        """
        Compares two items using an LLM model to determine if they match.

        Args:
            comparison_prompt (str): The prompt template for comparison.
            model (str): The LLM model to use for comparison.
            item1 (Dict): The first item to compare.
            item2 (Dict): The second item to compare.
            timeout_seconds (int): The timeout for the LLM call in seconds.
            max_retries_per_timeout (int): The maximum number of retries per timeout.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating whether the items match and the cost of the comparison.
        """

        prompt_template = Template(comparison_prompt)
        prompt = prompt_template.render(left=item1, right=item2)
        response = self.runner.api.call_llm(
            model,
            "compare",
            [{"role": "user", "content": prompt}],
            {"is_match": "bool"},
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=max_retries_per_timeout,
        )
        output = self.runner.api.parse_llm_response(response, {"is_match": "bool"})[0]
        return output["is_match"], completion_cost(response)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the EquijoinOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or if the blocking_keys structure is invalid.
            Specifically:
            - Raises if 'comparison_prompt' is missing from the config.
            - Raises if 'left' or 'right' are missing from the 'blocking_keys' structure (if present).
            - Raises if 'left' or 'right' are missing from the 'limits' structure (if present).
        """
        if "comparison_prompt" not in self.config:
            raise ValueError(
                "Missing required key 'comparison_prompt' in EquijoinOperation configuration"
            )

        if "blocking_keys" in self.config:
            if (
                "left" not in self.config["blocking_keys"]
                or "right" not in self.config["blocking_keys"]
            ):
                raise ValueError(
                    "Both 'left' and 'right' must be specified in 'blocking_keys'"
                )

        if "limits" in self.config:
            if (
                "left" not in self.config["limits"]
                or "right" not in self.config["limits"]
            ):
                raise ValueError(
                    "Both 'left' and 'right' must be specified in 'limits'"
                )

        if "limit_comparisons" in self.config:
            if not isinstance(self.config["limit_comparisons"], int):
                raise ValueError("limit_comparisons must be an integer")

    def execute(
        self, left_data: List[Dict], right_data: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        Executes the equijoin operation on the provided datasets.

        Args:
            left_data (List[Dict]): The left dataset to join.
            right_data (List[Dict]): The right dataset to join.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the joined results and the total cost of the operation.

        Usage:
        ```python
        from docetl.operations import EquijoinOperation

        config = {
            "blocking_keys": {
                "left": ["id"],
                "right": ["user_id"]
            },
            "limits": {
                "left": 1,
                "right": 1
            },
            "comparison_prompt": "Compare {{left}} and {{right}} and determine if they match.",
            "blocking_threshold": 0.8,
            "blocking_conditions": ["left['id'] == right['user_id']"],
            "limit_comparisons": 1000
        }
        equijoin_op = EquijoinOperation(config)
        left_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        right_data = [{"user_id": 1, "age": 30}, {"user_id": 2, "age": 25}]
        results, cost = equijoin_op.execute(left_data, right_data)
        print(f"Joined results: {results}")
        print(f"Total cost: {cost}")
        ```

        This method performs the following steps:
        1. Initial blocking based on specified conditions (if any)
        2. Embedding-based blocking (if threshold is provided)
        3. LLM-based comparison for blocked pairs
        4. Result aggregation and validation

        The method also calculates and logs statistics such as comparisons saved by blocking and join selectivity.
        """

        blocking_keys = self.config.get("blocking_keys", {})
        left_keys = blocking_keys.get(
            "left", list(left_data[0].keys()) if left_data else []
        )
        right_keys = blocking_keys.get(
            "right", list(right_data[0].keys()) if right_data else []
        )
        limits = self.config.get(
            "limits", {"left": float("inf"), "right": float("inf")}
        )
        left_limit = limits["left"]
        right_limit = limits["right"]
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])
        limit_comparisons = self.config.get("limit_comparisons")
        total_cost = 0

        # LLM-based comparison for blocked pairs
        def get_hashable_key(item: Dict) -> str:
            return json.dumps(item, sort_keys=True)

        if len(left_data) == 0 or len(right_data) == 0:
            return [], 0

        if self.status:
            self.status.stop()

        # Initial blocking using multiprocessing
        num_processes = min(cpu_count(), len(left_data))

        self.console.log(
            f"Starting to run code-based blocking rules for {len(left_data)} left and {len(right_data)} right rows ({len(left_data) * len(right_data)} total pairs) with {num_processes} processes..."
        )

        with Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(right_data, blocking_conditions),
        ) as pool:
            blocked_pairs_nested = pool.map(process_left_item, left_data)

        # Flatten the nested list of blocked pairs
        blocked_pairs = [pair for sublist in blocked_pairs_nested for pair in sublist]

        # Check if we have exceeded the pairwise comparison limit
        if limit_comparisons is not None and len(blocked_pairs) > limit_comparisons:
            # Sample pairs randomly
            sampled_pairs = random.sample(blocked_pairs, limit_comparisons)

            # Calculate number of dropped pairs
            dropped_pairs = len(blocked_pairs) - limit_comparisons

            # Prompt the user for confirmation
            if self.status:
                self.status.stop()
            if not Confirm.ask(
                f"[yellow]Warning: {dropped_pairs} pairs will be dropped due to the comparison limit. "
                f"Proceeding with {limit_comparisons} randomly sampled pairs. "
                f"Do you want to continue?[/yellow]",
                self.console,
            ):
                raise ValueError("Operation cancelled by user due to pair limit.")

            if self.status:
                self.status.start()

            blocked_pairs = sampled_pairs

        self.console.log(
            f"Number of blocked pairs after initial blocking: {len(blocked_pairs)}"
        )

        if blocking_threshold is not None:
            embedding_model = self.config.get("embedding_model", self.default_model)
            model_input_context_length = model_cost.get(embedding_model, {}).get(
                "max_input_tokens", 8192
            )

            def get_embeddings(
                input_data: List[Dict[str, Any]], keys: List[str], name: str
            ) -> Tuple[List[List[float]], float]:
                texts = [
                    " ".join(str(item[key]) for key in keys if key in item)[
                        : model_input_context_length * 4
                    ]
                    for item in input_data
                ]

                embeddings = []
                total_cost = 0
                batch_size = 2000
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    self.console.log(
                        f"On iteration {i} for creating embeddings for {name} data"
                    )
                    response = self.runner.api.gen_embedding(
                        model=embedding_model,
                        input=batch,
                    )
                    embeddings.extend([data["embedding"] for data in response["data"]])
                    total_cost += completion_cost(response)
                return embeddings, total_cost

            left_embeddings, left_cost = get_embeddings(left_data, left_keys, "left")
            right_embeddings, right_cost = get_embeddings(
                right_data, right_keys, "right"
            )
            total_cost += left_cost + right_cost
            self.console.log(
                f"Created embeddings for datasets. Total embedding creation cost: {total_cost}"
            )

            # Compute all cosine similarities in one call
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(left_embeddings, right_embeddings)

            # Additional blocking based on embeddings
            # Find indices where similarity is above threshold
            above_threshold = np.argwhere(similarities >= blocking_threshold)
            self.console.log(
                f"There are {above_threshold.shape[0]} pairs above the threshold."
            )
            block_pair_set = set(
                (get_hashable_key(left_item), get_hashable_key(right_item))
                for left_item, right_item in blocked_pairs
            )

            # If limit_comparisons is set, take only the top pairs
            if limit_comparisons is not None:
                # First, get all pairs above threshold
                above_threshold_pairs = [(int(i), int(j)) for i, j in above_threshold]

                # Sort these pairs by their similarity scores
                sorted_pairs = sorted(
                    above_threshold_pairs,
                    key=lambda pair: similarities[pair[0], pair[1]],
                    reverse=True,
                )

                # Take the top 'limit_comparisons' pairs
                top_pairs = sorted_pairs[:limit_comparisons]

                # Create new blocked_pairs based on top similarities and existing blocked pairs
                new_blocked_pairs = []
                remaining_limit = limit_comparisons - len(blocked_pairs)

                # First, include all existing blocked pairs
                final_blocked_pairs = blocked_pairs.copy()

                # Then, add new pairs from top similarities until we reach the limit
                for i, j in top_pairs:
                    if remaining_limit <= 0:
                        break
                    left_item, right_item = left_data[i], right_data[j]
                    left_key = get_hashable_key(left_item)
                    right_key = get_hashable_key(right_item)
                    if (left_key, right_key) not in block_pair_set:
                        new_blocked_pairs.append((left_item, right_item))
                        block_pair_set.add((left_key, right_key))
                        remaining_limit -= 1

                final_blocked_pairs.extend(new_blocked_pairs)
                blocked_pairs = final_blocked_pairs

                self.console.log(
                    f"Limited comparisons to top {limit_comparisons} pairs, including {len(blocked_pairs) - len(new_blocked_pairs)} from code-based blocking and {len(new_blocked_pairs)} based on cosine similarity. Lowest cosine similarity included: {similarities[top_pairs[-1]]:.4f}"
                )
            else:
                # Add new pairs to blocked_pairs
                for i, j in above_threshold:
                    left_item, right_item = left_data[i], right_data[j]
                    left_key = get_hashable_key(left_item)
                    right_key = get_hashable_key(right_item)
                    if (left_key, right_key) not in block_pair_set:
                        blocked_pairs.append((left_item, right_item))
                        block_pair_set.add((left_key, right_key))

        # If there are no blocking conditions or embedding threshold, use all pairs
        if not blocking_conditions and blocking_threshold is None:
            blocked_pairs = [
                (left_item, right_item)
                for left_item in left_data
                for right_item in right_data
            ]

        # If there's a limit on the number of comparisons, randomly sample pairs
        if limit_comparisons is not None and len(blocked_pairs) > limit_comparisons:
            self.console.log(
                f"Randomly sampling {limit_comparisons} pairs out of {len(blocked_pairs)} blocked pairs."
            )
            blocked_pairs = random.sample(blocked_pairs, limit_comparisons)

        self.console.log(
            f"Total pairs to compare after blocking and sampling: {len(blocked_pairs)}"
        )

        # Calculate and print statistics
        total_possible_comparisons = len(left_data) * len(right_data)
        comparisons_made = len(blocked_pairs)
        comparisons_saved = total_possible_comparisons - comparisons_made
        self.console.log(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )

        left_match_counts = defaultdict(int)
        right_match_counts = defaultdict(int)
        results = []
        comparison_costs = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_pair = {
                executor.submit(
                    self.compare_pair,
                    self.config["comparison_prompt"],
                    self.config.get("comparison_model", self.default_model),
                    left,
                    right,
                    self.config.get("timeout", 120),
                    self.config.get("max_retries_per_timeout", 2),
                ): (left, right)
                for left, right in blocked_pairs
            }

            for future in rich_as_completed(
                future_to_pair,
                total=len(future_to_pair),
                desc="Comparing pairs",
                console=self.console,
            ):
                pair = future_to_pair[future]
                is_match, cost = future.result()
                comparison_costs += cost

                if is_match:
                    joined_item = {}
                    left_item, right_item = pair
                    left_key_hash = get_hashable_key(left_item)
                    right_key_hash = get_hashable_key(right_item)
                    if (
                        left_match_counts[left_key_hash] >= left_limit
                        or right_match_counts[right_key_hash] >= right_limit
                    ):
                        continue

                    for key, value in left_item.items():
                        joined_item[f"{key}_left" if key in right_item else key] = value
                    for key, value in right_item.items():
                        joined_item[f"{key}_right" if key in left_item else key] = value
                    if self.runner.api.validate_output(
                        self.config, joined_item, self.console
                    ):
                        results.append(joined_item)
                        left_match_counts[left_key_hash] += 1
                        right_match_counts[right_key_hash] += 1

                    # TODO: support retry in validation failure

        total_cost += comparison_costs

        # Calculate and print the join selectivity
        join_selectivity = (
            len(results) / (len(left_data) * len(right_data))
            if len(left_data) * len(right_data) > 0
            else 0
        )
        self.console.log(f"Equijoin selectivity: {join_selectivity:.4f}")

        if self.status:
            self.status.start()

        return results, total_cost
