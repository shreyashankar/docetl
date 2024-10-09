"""
The `ResolveOperation` class is a subclass of `BaseOperation` that performs a resolution operation on a dataset. It uses a combination of blocking techniques and LLM-based comparisons to efficiently identify and resolve duplicate or related entries within the dataset.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import jinja2
from jinja2 import Template
from rich.prompt import Confirm

from docetl.operations.base import BaseOperation
from docetl.operations.utils import RichLoopBar, rich_as_completed
from docetl.utils import completion_cost, extract_jinja_variables


class ResolveOperation(BaseOperation):
    def compare_pair(
        self,
        comparison_prompt: str,
        model: str,
        item1: Dict,
        item2: Dict,
        blocking_keys: List[str] = [],
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

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating whether the items match and the cost of the comparison.
        """
        if blocking_keys:
            if all(
                key in item1
                and key in item2
                and item1[key].lower() == item2[key].lower()
                for key in blocking_keys
            ):
                return True, 0

        prompt_template = Template(comparison_prompt)
        prompt = prompt_template.render(input1=item1, input2=item2)
        response = self.runner.api.call_llm(
            model,
            "compare",
            [{"role": "user", "content": prompt}],
            {"is_match": "bool"},
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=max_retries_per_timeout,
        )
        output = self.runner.api.parse_llm_response(
            response,
            {"is_match": "bool"},
        )[0]
        return output["is_match"], completion_cost(response)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the ResolveOperation for required keys and valid structure.

        This method performs the following checks:
        1. Verifies the presence of required keys: 'comparison_prompt' and 'output'.
        2. Ensures 'output' contains a 'schema' key.
        3. Validates that 'schema' in 'output' is a non-empty dictionary.
        4. Checks if 'comparison_prompt' is a valid Jinja2 template with 'input1' and 'input2' variables.
        5. If 'resolution_prompt' is present, verifies it as a valid Jinja2 template with 'inputs' variable.
        6. Optionally checks if 'model' is a string (if present).
        7. Optionally checks 'blocking_keys' (if present, further checks are performed).

        Raises:
            ValueError: If required keys are missing, if templates are invalid or missing required variables,
                        or if any other configuration aspect is incorrect or inconsistent.
            TypeError: If the types of configuration values are incorrect, such as 'schema' not being a dict
                       or 'model' not being a string.
        """
        required_keys = ["comparison_prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ResolveOperation configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the comparison_prompt is a valid Jinja2 template
        try:
            comparison_template = Template(self.config["comparison_prompt"])
            comparison_vars = comparison_template.environment.parse(
                self.config["comparison_prompt"]
            ).find_all(jinja2.nodes.Name)
            comparison_var_names = {var.name for var in comparison_vars}
            if (
                "input1" not in comparison_var_names
                or "input2" not in comparison_var_names
            ):
                raise ValueError(
                    "'comparison_prompt' must contain both 'input1' and 'input2' variables"
                )

            if "resolution_prompt" in self.config:
                reduction_template = Template(self.config["resolution_prompt"])
                reduction_vars = reduction_template.environment.parse(
                    self.config["resolution_prompt"]
                ).find_all(jinja2.nodes.Name)
                reduction_var_names = {var.name for var in reduction_vars}
                if "inputs" not in reduction_var_names:
                    raise ValueError(
                        "'resolution_prompt' must contain 'inputs' variable"
                    )
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {str(e)}")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check blocking_keys (optional)
        if "blocking_keys" in self.config:
            if not isinstance(self.config["blocking_keys"], list):
                raise TypeError("'blocking_keys' must be a list")
            if not all(isinstance(key, str) for key in self.config["blocking_keys"]):
                raise TypeError("All items in 'blocking_keys' must be strings")

        # Check blocking_threshold (optional)
        if "blocking_threshold" in self.config:
            if not isinstance(self.config["blocking_threshold"], (int, float)):
                raise TypeError("'blocking_threshold' must be a number")
            if not 0 <= self.config["blocking_threshold"] <= 1:
                raise ValueError("'blocking_threshold' must be between 0 and 1")

        # Check blocking_conditions (optional)
        if "blocking_conditions" in self.config:
            if not isinstance(self.config["blocking_conditions"], list):
                raise TypeError("'blocking_conditions' must be a list")
            if not all(
                isinstance(cond, str) for cond in self.config["blocking_conditions"]
            ):
                raise TypeError("All items in 'blocking_conditions' must be strings")

        # Check if input schema is provided and valid (optional)
        if "input" in self.config:
            if "schema" not in self.config["input"]:
                raise ValueError("Missing 'schema' in 'input' configuration")
            if not isinstance(self.config["input"]["schema"], dict):
                raise TypeError(
                    "'schema' in 'input' configuration must be a dictionary"
                )

        # Check limit_comparisons (optional)
        if "limit_comparisons" in self.config:
            if not isinstance(self.config["limit_comparisons"], int):
                raise TypeError("'limit_comparisons' must be an integer")
            if self.config["limit_comparisons"] <= 0:
                raise ValueError("'limit_comparisons' must be a positive integer")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Executes the resolve operation on the provided dataset.

        Args:
            input_data (List[Dict]): The dataset to resolve.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the resolved results and the total cost of the operation.

        This method performs the following steps:
        1. Initial blocking based on specified conditions and/or embedding similarity
        2. Pairwise comparison of potentially matching entries using LLM
        3. Clustering of matched entries
        4. Resolution of each cluster into a single entry (if applicable)
        5. Result aggregation and validation

        The method also calculates and logs statistics such as comparisons saved by blocking and self-join selectivity.
        """
        if len(input_data) == 0:
            return [], 0

        blocking_keys = self.config.get("blocking_keys", [])
        blocking_threshold = self.config.get("blocking_threshold")
        blocking_conditions = self.config.get("blocking_conditions", [])
        if self.status:
            self.status.stop()

        if not blocking_threshold and not blocking_conditions:
            # Prompt the user for confirmation
            if not Confirm.ask(
                f"[yellow]Warning: No blocking keys or conditions specified. "
                f"This may result in a large number of comparisons. "
                f"We recommend specifying at least one blocking key or condition, or using the optimizer to automatically come up with these. "
                f"Do you want to continue without blocking?[/yellow]",
                self.console,
            ):
                raise ValueError("Operation cancelled by user.")

        input_schema = self.config.get("input", {}).get("schema", {})
        if not blocking_keys:
            # Set them to all keys in the input data
            blocking_keys = list(input_data[0].keys())
        limit_comparisons = self.config.get("limit_comparisons")
        total_cost = 0

        def is_match(item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
            return any(
                eval(condition, {"input1": item1, "input2": item2})
                for condition in blocking_conditions
            )

        # Calculate embeddings if blocking_threshold is set
        embeddings = None
        if blocking_threshold is not None:
            embedding_model = self.config.get("embedding_model", self.default_model)

            def get_embeddings_batch(
                items: List[Dict[str, Any]]
            ) -> List[Tuple[List[float], float]]:
                texts = [
                    " ".join(str(item[key]) for key in blocking_keys if key in item)
                    for item in items
                ]
                response = self.runner.api.gen_embedding(
                    model=embedding_model, input=texts
                )
                return [
                    (data["embedding"], completion_cost(response))
                    for data in response["data"]
                ]

            embeddings = []
            costs = []
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                for i in range(
                    0, len(input_data), self.config.get("embedding_batch_size", 1000)
                ):
                    batch = input_data[
                        i : i + self.config.get("embedding_batch_size", 1000)
                    ]
                    batch_results = list(executor.map(get_embeddings_batch, [batch]))

                    for result in batch_results:
                        embeddings.extend([r[0] for r in result])
                        costs.extend([r[1] for r in result])

                total_cost += sum(costs)

        # Initialize clusters
        clusters = [{i} for i in range(len(input_data))]
        cluster_map = {i: i for i in range(len(input_data))}

        def find_cluster(item):
            while item != cluster_map[item]:
                cluster_map[item] = cluster_map[cluster_map[item]]
                item = cluster_map[item]
            return item

        def merge_clusters(item1, item2):
            root1, root2 = find_cluster(item1), find_cluster(item2)
            if root1 != root2:
                if len(clusters[root1]) < len(clusters[root2]):
                    root1, root2 = root2, root1
                clusters[root1] |= clusters[root2]
                cluster_map[root2] = root1
                clusters[root2] = set()

        # Generate all pairs to compare
        # TODO: virtualize this if possible
        all_pairs = [
            (i, j)
            for i in range(len(input_data))
            for j in range(i + 1, len(input_data))
        ]

        # Filter pairs based on blocking conditions
        def meets_blocking_conditions(pair):
            i, j = pair
            return (
                is_match(input_data[i], input_data[j]) if blocking_conditions else False
            )

        blocked_pairs = list(filter(meets_blocking_conditions, all_pairs))

        # Apply limit_comparisons to blocked pairs
        if limit_comparisons is not None and len(blocked_pairs) > limit_comparisons:
            self.console.log(
                f"Randomly sampling {limit_comparisons} pairs out of {len(blocked_pairs)} blocked pairs."
            )
            blocked_pairs = random.sample(blocked_pairs, limit_comparisons)

        # If there are remaining comparisons, fill with highest cosine similarities
        remaining_comparisons = (
            limit_comparisons - len(blocked_pairs)
            if limit_comparisons is not None
            else float("inf")
        )
        if remaining_comparisons > 0 and blocking_threshold is not None:
            # Compute cosine similarity for all pairs efficiently
            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(embeddings)

            cosine_pairs = []
            for i, j in all_pairs:
                if (i, j) not in blocked_pairs and find_cluster(i) != find_cluster(j):
                    similarity = similarity_matrix[i, j]
                    if similarity >= blocking_threshold:
                        cosine_pairs.append((i, j, similarity))

            if remaining_comparisons != float("inf"):
                cosine_pairs.sort(key=lambda x: x[2], reverse=True)
                additional_pairs = [
                    (i, j) for i, j, _ in cosine_pairs[: int(remaining_comparisons)]
                ]
                blocked_pairs.extend(additional_pairs)
            else:
                blocked_pairs.extend((i, j) for i, j, _ in cosine_pairs)

        if not blocking_threshold and not blocking_conditions:
            filtered_pairs = all_pairs
        else:
            filtered_pairs = blocked_pairs

        # Calculate and print statistics
        total_possible_comparisons = len(input_data) * (len(input_data) - 1) // 2
        comparisons_made = len(filtered_pairs)
        comparisons_saved = total_possible_comparisons - comparisons_made
        self.console.log(
            f"[green]Comparisons saved by blocking: {comparisons_saved} "
            f"({(comparisons_saved / total_possible_comparisons) * 100:.2f}%)[/green]"
        )

        # Compare pairs and update clusters in real-time
        batch_size = self.config.get("compare_batch_size", 100)
        pair_costs = 0

        pbar = RichLoopBar(
            range(0, len(filtered_pairs), batch_size),
            desc=f"Processing batches of {batch_size} LLM comparisons",
            console=self.console,
        )
        for i in pbar:
            batch = filtered_pairs[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_pair = {
                    executor.submit(
                        self.compare_pair,
                        self.config["comparison_prompt"],
                        self.config.get("comparison_model", self.default_model),
                        input_data[pair[0]],
                        input_data[pair[1]],
                        blocking_keys,
                        timeout_seconds=self.config.get("timeout", 120),
                        max_retries_per_timeout=self.config.get(
                            "max_retries_per_timeout", 2
                        ),
                    ): pair
                    for pair in batch
                }

                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    is_match_result, cost = future.result()
                    pair_costs += cost
                    if is_match_result:
                        merge_clusters(pair[0], pair[1])

                    pbar.update(i)

        total_cost += pair_costs

        # Collect final clusters
        final_clusters = [cluster for cluster in clusters if cluster]

        # Process each cluster
        results = []

        def process_cluster(cluster):
            if len(cluster) > 1:
                cluster_items = [input_data[i] for i in cluster]
                reduction_template = Template(self.config["resolution_prompt"])
                if input_schema:
                    cluster_items = [
                        {k: item[k] for k in input_schema.keys() if k in item}
                        for item in cluster_items
                    ]

                resolution_prompt = reduction_template.render(inputs=cluster_items)
                reduction_response = self.runner.api.call_llm(
                    self.config.get("resolution_model", self.default_model),
                    "reduce",
                    [{"role": "user", "content": resolution_prompt}],
                    self.config["output"]["schema"],
                    console=self.console,
                    timeout_seconds=self.config.get("timeout", 120),
                    max_retries_per_timeout=self.config.get(
                        "max_retries_per_timeout", 2
                    ),
                )
                reduction_output = self.runner.api.parse_llm_response(
                    reduction_response,
                    self.config["output"]["schema"],
                    manually_fix_errors=self.manually_fix_errors,
                )[0]
                reduction_cost = completion_cost(reduction_response)

                if self.runner.api.validate_output(
                    self.config, reduction_output, self.console
                ):
                    # If the output is overwriting an existing key, we want to save the kv pairs
                    keys_in_output = [
                        k
                        for k in set(reduction_output.keys())
                        if k in cluster_items[0].keys()
                    ]

                    return (
                        [
                            {
                                **item,
                                f"_kv_pairs_preresolve_{self.config['name']}": {
                                    k: item[k] for k in keys_in_output
                                },
                                **{
                                    k: reduction_output[k]
                                    for k in self.config["output"]["schema"]
                                },
                            }
                            for item in [input_data[i] for i in cluster]
                        ],
                        reduction_cost,
                    )
                return [], reduction_cost
            else:
                # Set the output schema to be the keys found in the compare_prompt
                compare_prompt_keys = extract_jinja_variables(
                    self.config["comparison_prompt"]
                )
                # Get the set of keys in the compare_prompt
                compare_prompt_keys = set(
                    [
                        k.replace("input1.", "")
                        for k in compare_prompt_keys
                        if "input1" in k
                    ]
                )

                # For each key in the output schema, find the most similar key in the compare_prompt
                output_keys = set(self.config["output"]["schema"].keys())
                key_mapping = {}
                for output_key in output_keys:
                    best_match = None
                    best_score = 0
                    for compare_key in compare_prompt_keys:
                        score = sum(
                            c1 == c2 for c1, c2 in zip(output_key, compare_key)
                        ) / max(len(output_key), len(compare_key))
                        if score > best_score:
                            best_score = score
                            best_match = compare_key
                    key_mapping[output_key] = best_match

                # Create the result dictionary using the key mapping
                result = input_data[list(cluster)[0]].copy()
                result[f"_kv_pairs_preresolve_{self.config['name']}"] = {
                    ok: result[ck] for ok, ck in key_mapping.items() if ck in result
                }
                for output_key, compare_key in key_mapping.items():
                    if compare_key in input_data[list(cluster)[0]]:
                        result[output_key] = input_data[list(cluster)[0]][compare_key]
                    else:
                        result[output_key] = None  # or some default value

                return [result], 0

        # Calculate the number of records before and clusters after
        num_records_before = len(input_data)
        num_clusters_after = len(final_clusters)
        self.console.log(f"Number of keys before resolution: {num_records_before}")
        self.console.log(
            f"Number of distinct keys after resolution: {num_clusters_after}"
        )

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_cluster, cluster) for cluster in final_clusters
            ]
            for future in rich_as_completed(
                futures,
                total=len(futures),
                desc="Determining resolved key for each group of equivalent keys",
                console=self.console,
            ):
                cluster_results, cluster_cost = future.result()
                results.extend(cluster_results)
                total_cost += cluster_cost

        total_pairs = len(input_data) * (len(input_data) - 1) // 2
        true_match_count = sum(
            len(cluster) * (len(cluster) - 1) // 2
            for cluster in final_clusters
            if len(cluster) > 1
        )
        true_match_selectivity = (
            true_match_count / total_pairs if total_pairs > 0 else 0
        )
        self.console.log(f"Self-join selectivity: {true_match_selectivity:.4f}")

        if self.status:
            self.status.start()

        return results, total_cost
