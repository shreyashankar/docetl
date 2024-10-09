import json
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from litellm import model_cost
from rich.console import Console
from rich.prompt import Confirm
from rich.status import Status

from docetl.operations.equijoin import EquijoinOperation
from docetl.operations.resolve import ResolveOperation
from docetl.utils import completion_cost, extract_jinja_variables


class JoinOptimizer:
    def __init__(
        self,
        runner,
        config: Dict[str, Any],
        op_config: Dict[str, Any],
        console: Console,
        llm_client: Any,
        max_threads: int,
        target_recall: float = 0.95,
        sample_size: int = 500,
        sampling_weight: float = 20,
        agent_max_retries: int = 5,
        estimated_selectivity: float = None,
        status: Status = None,
    ):
        self.runner = runner
        self.config = config
        self.op_config = op_config
        self.llm_client = llm_client
        self.max_threads = max_threads
        self.console = console
        self.target_recall = target_recall
        self.sample_size = sample_size
        self.sampling_weight = sampling_weight
        self.agent_max_retries = agent_max_retries
        self.estimated_selectivity = estimated_selectivity
        self.console.log(f"Target Recall: {self.target_recall}")
        self.status = status
        # if self.estimated_selectivity is not None:
        #     self.console.log(
        #         f"[yellow]Using estimated selectivity of {self.estimated_selectivity}[/yellow]"
        #     )

    def _analyze_map_prompt_categorization(self, map_prompt: str) -> bool:
        """
        Analyze the map prompt to determine if it's explicitly categorical.

        Args:
            map_prompt (str): The map prompt to analyze.

        Returns:
            bool: True if the prompt is explicitly categorical, False otherwise.
        """
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant tasked with analyzing prompts for data processing operations.",
            },
            {
                "role": "user",
                "content": f"""Analyze the following map operation prompt and determine if it is explicitly categorical,
                meaning it details a specific set of possible outputs:

                {map_prompt}

                Respond with 'Yes' if the prompt is explicitly categorical, detailing a finite set of possible outputs.
                Respond with 'No' if the prompt allows for open-ended or non-categorical responses.
                Provide a brief explanation for your decision.""",
            },
        ]

        response = self.llm_client.generate(
            messages,
            "You are an expert in analyzing natural language prompts for data processing tasks.",
            {
                "type": "object",
                "properties": {
                    "is_categorical": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Whether the prompt is explicitly categorical",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the decision",
                    },
                },
                "required": ["is_categorical", "explanation"],
            },
        )

        analysis = json.loads(response.choices[0].message.content)

        self.console.log("[bold]Map Prompt Analysis:[/bold]")
        self.console.log(f"Is Categorical: {analysis['is_categorical']}")
        self.console.log(f"Explanation: {analysis['explanation']}")

        return analysis["is_categorical"].lower() == "yes"

    def _determine_duplicate_keys(
        self,
        input_data: List[Dict[str, Any]],
        reduce_key: List[str],
        map_prompt: Optional[str] = None,
    ) -> bool:
        # Prepare a sample of the input data for analysis
        sample_size = min(10, len(input_data))
        data_sample = random.sample(
            [{rk: item[rk] for rk in reduce_key} for item in input_data], sample_size
        )

        context_prefix = ""
        if map_prompt:
            context_prefix = f"For context, these values came out of a pipeline with the following prompt:\n\n{map_prompt}\n\n"

        messages = [
            {
                "role": "user",
                "content": f"{context_prefix}I want to do a reduce operation on these values, and I need to determine if there are semantic duplicates in the data, where the strings are different but they technically belong in the same group. Note that exact string duplicates should not be considered here.\n\nHere's a sample of the data (showing the '{reduce_key}' field(s)): {data_sample}\n\nBased on this {'context and ' if map_prompt else ''}sample, are there likely to be such semantic duplicates (not exact string matches) in the dataset? Respond with 'yes' only if you think there are semantic duplicates, or 'no' if you don't see evidence of semantic duplicates or if you only see exact string duplicates.",
            },
        ]
        response = self.llm_client.generate(
            messages,
            "You are an expert data analyst. Analyze the given data sample and determine if there are likely to be semantic duplicate values that belong in the same group, even if the strings are different.",
            {
                "type": "object",
                "properties": {
                    "likely_duplicates": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                        "description": "Whether duplicates are likely to exist in the full dataset",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation for the decision",
                    },
                },
                "required": ["likely_duplicates", "explanation"],
            },
        )

        analysis = json.loads(response.choices[0].message.content)

        self.console.log(f"[bold]Duplicate Analysis for '{reduce_key}':[/bold]")
        self.console.log(f"Likely Duplicates: {analysis['likely_duplicates']}")
        self.console.log(f"Explanation: {analysis['explanation']}")

        if analysis["likely_duplicates"].lower() == "yes":
            self.console.log(
                "[yellow]Duplicates are likely. Consider using a deduplication strategy in the resolution step.[/yellow]"
            )
            return True
        return False

    def _sample_random_pairs(
        self, input_data: List[Dict[str, Any]], n: int
    ) -> List[Tuple[int, int]]:
        """Sample random pairs of indices, excluding exact matches."""
        pairs = set()
        max_attempts = n * 10  # Avoid infinite loop
        attempts = 0

        while len(pairs) < n and attempts < max_attempts:
            i, j = random.sample(range(len(input_data)), 2)
            if i != j and input_data[i] != input_data[j]:
                pairs.add((min(i, j), max(i, j)))  # Ensure ordered pairs
            attempts += 1

        return list(pairs)

    def _check_duplicates_with_llm(
        self,
        input_data: List[Dict[str, Any]],
        pairs: List[Tuple[int, int]],
        reduce_key: List[str],
        map_prompt: Optional[str],
    ) -> bool:
        """Use LLM to check if any pairs are duplicates."""

        content = "Analyze the following pairs of entries and determine if any of them are likely duplicates. Respond with 'Yes' if you find any likely duplicates, or 'No' if none of the pairs seem to be duplicates. Provide a brief explanation for your decision.\n\n"

        if map_prompt:
            content = (
                f"For reference, here is the map prompt used earlier in the pipeline: {map_prompt}\n\n"
                + content
            )

        for i, (idx1, idx2) in enumerate(pairs, 1):
            content += f"Pair {i}:\n"
            content += "Entry 1:\n"
            for key in reduce_key:
                content += f"{key}: {json.dumps(input_data[idx1][key], indent=2)}\n"
            content += "\nEntry 2:\n"
            for key in reduce_key:
                content += f"{key}: {json.dumps(input_data[idx2][key], indent=2)}\n"
            content += "\n"

        messages = [{"role": "user", "content": content}]

        system_prompt = "You are an AI assistant tasked with identifying potential duplicate entries in a dataset."
        response_schema = {
            "type": "object",
            "properties": {
                "duplicates_found": {"type": "string", "enum": ["Yes", "No"]},
                "explanation": {"type": "string"},
            },
            "required": ["duplicates_found", "explanation"],
        }

        response = self.llm_client.generate(messages, system_prompt, response_schema)

        # Print the duplicates_found and explanation
        self.console.log(
            f"[bold]Duplicates in keys found:[/bold] {response['duplicates_found']}\n"
            f"[bold]Explanation:[/bold] {response['explanation']}"
        )

        return response["duplicates_found"].lower() == "yes"

    def synthesize_compare_prompt(
        self, map_prompt: Optional[str], reduce_key: List[str]
    ) -> str:

        system_prompt = f"You are an AI assistant tasked with creating a comparison prompt for LLM-assisted entity resolution. Your task is to create a comparison prompt that will be used to compare two entities, referred to as input1 and input2, to see if they are likely the same entity based on the following reduce key(s): {', '.join(reduce_key)}."
        if map_prompt:
            system_prompt += f"\n\nFor context, here is the prompt used earlier in the pipeline to create the inputs to resolve: {map_prompt}"

        messages = [
            {
                "role": "user",
                "content": f"""
    Create a comparison prompt for entity resolution: The prompt should:
    1. Be tailored to the specific domain and type of data being compared ({reduce_key}), based on the context provided.
    2. Instruct to compare two entities, referred to as input1 and input2.
    3. Specifically mention comparing each reduce key in input1 and input2 (e.g., input1.{{key}} and input2.{{key}} for each key in {reduce_key}). You can reference other fields in the input as well, as long as they are short.
    4. Include instructions to consider relevant attributes or characteristics for comparison.
    5. Ask to respond with "True" if the entities are likely the same, or "False" if they are likely different.

    Example structure:
    ```
    Compare the following two {reduce_key} from [entity or document type]:

    [Entity 1]:
    {{{{ input1.key1 }}}}
    {{{{ input1.optional_key2 }}}}

    [Entity 2]:
    {{{{ input2.key1 }}}}
    {{{{ input2.optional_key2 }}}}

    Are these [entities] likely referring to the same [entity type]? Consider [list relevant attributes or characteristics to compare]. Respond with "True" if they are likely the same [entity type], or "False" if they are likely different [entity types].
    ```

    Please generate the comparison prompt:
    """,
            }
        ]

        response = self.llm_client.generate(
            messages,
            system_prompt,
            {
                "type": "object",
                "properties": {
                    "comparison_prompt": {
                        "type": "string",
                        "description": "Detailed comparison prompt for entity resolution",
                    }
                },
                "required": ["comparison_prompt"],
            },
        )

        comparison_prompt = json.loads(response.choices[0].message.content)[
            "comparison_prompt"
        ]

        # Log the synthesized comparison prompt
        self.console.log("[green]Synthesized comparison prompt:[/green]")
        self.console.log(comparison_prompt)

        if not comparison_prompt:
            raise ValueError(
                "Could not synthesize a comparison prompt. Please provide a comparison prompt in the config."
            )

        return comparison_prompt

    def synthesize_resolution_prompt(
        self,
        map_prompt: Optional[str],
        reduce_key: List[str],
        output_schema: Dict[str, str],
    ) -> str:
        system_prompt = f"""You are an AI assistant tasked with creating a resolution prompt for LLM-assisted entity resolution.
        Your task is to create a prompt that will be used to merge multiple duplicate keys into a single, consolidated key.
        The key(s) being resolved (known as the reduce_key) are {', '.join(reduce_key)}.
        The duplicate keys will be provided in a list called 'inputs' in a Jinja2 template.
        """

        if map_prompt:
            system_prompt += f"\n\nFor context, here is the prompt used earlier in the pipeline to create the inputs to resolve: {map_prompt}"

        messages = [
            {
                "role": "user",
                "content": f"""
    Create a resolution prompt for merging duplicate keys into a single key. The prompt should:
    1. Be tailored to the specific domain and type of data being merged, based on the context provided.
    2. Use a Jinja2 template to iterate over the duplicate keys (accessed as 'inputs', where each item is a dictionary containing the reduce_key fields, which you can access as entry.reduce_key for each reduce_key in {reduce_key}).
    3. Instruct to create a single, consolidated key from the duplicate keys.
    4. Include guidelines for resolving conflicts (e.g., choosing the most recent, most complete, or most reliable information).
    5. Specify that the output of the resolution prompt should conform to the given output schema: {json.dumps(output_schema, indent=2)}

    Example structure:
    ```
    Analyze the following duplicate entries for the {reduce_key} key:

    {{% for key in inputs %}}
    Entry {{{{ loop.index }}}}:
    {{ % for key in reduce_key %}}
    {{{{ key }}}}: {{{{ key[reduce_key] }}}}
    {{% endfor %}}

    {{% endfor %}}

    Merge these into a single key.
    When merging, follow these guidelines:
    1. [Provide specific merging instructions relevant to the data type]
    2. [Do not make the prompt too long]

    Ensure that the merged key conforms to the following schema:
    {json.dumps(output_schema, indent=2)}

    Return the consolidated key as a single [appropriate data type] value.
    ```

    Please generate the resolution prompt:
    """,
            }
        ]

        response = self.llm_client.generate(
            messages,
            system_prompt,
            {
                "type": "object",
                "properties": {
                    "resolution_prompt": {
                        "type": "string",
                        "description": "Detailed resolution prompt for merging duplicate keys",
                    }
                },
                "required": ["resolution_prompt"],
            },
        )

        resolution_prompt = json.loads(response.choices[0].message.content)[
            "resolution_prompt"
        ]

        # Log the synthesized resolution prompt
        self.console.log("[green]Synthesized resolution prompt:[/green]")
        self.console.log(resolution_prompt)

        if not resolution_prompt:
            raise ValueError(
                "Could not synthesize a resolution prompt. Please provide a resolution prompt in the config."
            )

        return resolution_prompt

    def optimize_resolve(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:

        # Check if the operation is marked as empty
        if self.op_config.get("empty", False):
            # Extract the map prompt from the intermediates
            map_prompt = self.op_config["_intermediates"]["map_prompt"]
            reduce_key = self.op_config["_intermediates"]["reduce_key"]

            if reduce_key is None:
                raise ValueError(
                    "[yellow]Warning: No reduce key found in intermediates for synthesized resolve operation.[/yellow]"
                )

            dedup = True

            if map_prompt:
                # Analyze the map prompt
                analysis = self._analyze_map_prompt_categorization(map_prompt)

                if analysis:
                    dedup = False
            else:
                self.console.log(
                    "[yellow]No map prompt found in intermediates for analysis.[/yellow]"
                )

            # TODO: figure out why this would ever be the case
            if not map_prompt:
                map_prompt = "N/A"

            if dedup is False:
                dedup = self._determine_duplicate_keys(
                    input_data, reduce_key, map_prompt
                )

            # Now do the last attempt of pairwise comparisons
            if dedup is False:
                # Sample up to 20 random pairs of keys for duplicate analysis
                sampled_pairs = self._sample_random_pairs(input_data, 20)

                # Use LLM to check for duplicates
                duplicates_found = self._check_duplicates_with_llm(
                    input_data, sampled_pairs, reduce_key, map_prompt
                )

                if duplicates_found:
                    dedup = True

            if dedup is False:
                # If no deduplication is needed, return the same config with 0 cost
                return self.op_config, 0.0

            # Add the reduce key to the output schema in the config
            self.op_config["output"] = {"schema": {rk: "string" for rk in reduce_key}}
            for attempt in range(2):  # Try up to 2 times
                self.op_config["comparison_prompt"] = self.synthesize_compare_prompt(
                    map_prompt, reduce_key
                )
                if (
                    "input1" in self.op_config["comparison_prompt"]
                    and "input2" in self.op_config["comparison_prompt"]
                ):
                    break
                elif attempt == 0:
                    self.console.log(
                        "[yellow]Warning: 'input1' or 'input2' not found in comparison prompt. Retrying...[/yellow]"
                    )
            if (
                "input1" not in self.op_config["comparison_prompt"]
                or "input2" not in self.op_config["comparison_prompt"]
            ):
                self.console.log(
                    "[red]Error: Failed to generate comparison prompt with 'input1' and 'input2'. Using last generated prompt.[/red]"
                )
            for attempt in range(2):  # Try up to 2 times
                self.op_config["resolution_prompt"] = self.synthesize_resolution_prompt(
                    map_prompt, reduce_key, self.op_config["output"]["schema"]
                )
                if "inputs" in self.op_config["resolution_prompt"]:
                    break
                elif attempt == 0:
                    self.console.log(
                        "[yellow]Warning: 'inputs' not found in resolution prompt. Retrying...[/yellow]"
                    )
            if "inputs" not in self.op_config["resolution_prompt"]:
                self.console.log(
                    "[red]Error: Failed to generate resolution prompt with 'inputs'. Using last generated prompt.[/red]"
                )

            # Pop off the empty flag
            self.op_config.pop("empty")

        embeddings, blocking_keys, embedding_cost = self._compute_embeddings(input_data)
        self.console.log(
            f"[bold]Cost of creating embeddings on the sample: ${embedding_cost:.4f}[/bold]"
        )

        similarities = self._calculate_cosine_similarities(embeddings)

        sampled_pairs = self._sample_pairs(similarities)
        comparison_results, comparison_cost = self._perform_comparisons_resolve(
            input_data, sampled_pairs
        )

        self._print_similarity_histogram(similarities, comparison_results)

        threshold, estimated_selectivity = self._find_optimal_threshold(
            comparison_results, similarities
        )

        blocking_rules = self._generate_blocking_rules(
            blocking_keys, input_data, comparison_results
        )

        if blocking_rules:
            false_negatives, rule_selectivity = self._verify_blocking_rule(
                input_data,
                blocking_rules[0],
                blocking_keys,
                comparison_results,
            )
            if not false_negatives and rule_selectivity <= estimated_selectivity:
                self.console.log(
                    "[green]Blocking rule verified. No false negatives detected in the sample and selectivity is within estimated selectivity.[/green]"
                )
            else:
                if false_negatives:
                    self.console.log(
                        f"[red]Blocking rule rejected. {len(false_negatives)} false negatives detected in the sample.[/red]"
                    )
                    for i, j in false_negatives[:5]:  # Show up to 5 examples
                        self.console.log(
                            f"  Filtered pair: {{ {blocking_keys[0]}: {input_data[i][blocking_keys[0]]} }} and {{ {blocking_keys[0]}: {input_data[j][blocking_keys[0]]} }}"
                        )
                    if len(false_negatives) > 5:
                        self.console.log(f"  ... and {len(false_negatives) - 5} more.")
                if rule_selectivity > estimated_selectivity:
                    self.console.log(
                        f"[red]Blocking rule rejected. Rule selectivity ({rule_selectivity:.4f}) is higher than the estimated selectivity ({estimated_selectivity:.4f}).[/red]"
                    )
                blocking_rules = (
                    []
                )  # Clear the blocking rule if it introduces false negatives or is too selective

        optimized_config = self._update_config(threshold, blocking_keys, blocking_rules)
        return optimized_config, embedding_cost + comparison_cost

    def optimize_equijoin(
        self, left_data: List[Dict[str, Any]], right_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        left_keys = self.op_config.get("blocking_keys", {}).get("left", [])
        right_keys = self.op_config.get("blocking_keys", {}).get("right", [])

        if not left_keys and not right_keys:
            # Ask the LLM agent if it would be beneficial to do a map operation on
            # one of the datasets before doing an equijoin
            apply_transformation, dataset_to_transform, reason = (
                self._should_apply_map_transformation(
                    left_keys, right_keys, left_data, right_data
                )
            )

            if apply_transformation:
                self.console.log(
                    f"LLM agent suggested applying a map transformation to {dataset_to_transform} dataset because: {reason}"
                )
                extraction_prompt, output_key, new_comparison_prompt = (
                    self._generate_map_and_new_join_transformation(
                        dataset_to_transform, reason, left_data, right_data
                    )
                )
                self.console.log(
                    f"Generated map transformation prompt: {extraction_prompt}"
                )
                self.console.log(f"\nNew output key: {output_key}")
                self.console.log(
                    f"\nNew equijoin comparison prompt: {new_comparison_prompt}"
                )

                # Update the comparison prompt
                self.op_config["comparison_prompt"] = new_comparison_prompt

                # Add the output key to the left_keys or right_keys
                if dataset_to_transform == "left":
                    left_keys.append(output_key)
                else:
                    right_keys.append(output_key)

                # Reset the blocking keys in the config
                self.op_config["blocking_keys"] = {
                    "left": left_keys,
                    "right": right_keys,
                }

                # Bubble up this config and return the transformation prompt, so we can optimize the map operation
                return (
                    self.op_config,
                    0.0,
                    {
                        "optimize_map": True,
                        "map_prompt": extraction_prompt,
                        "output_key": output_key,
                        "dataset_to_transform": dataset_to_transform,
                    },
                )

            # Print the reason for not applying a map transformation
            self.console.log(
                f"Reason for not synthesizing a map transformation for either left or right dataset: {reason}"
            )

        # If there are no blocking keys, generate them
        if not left_keys or not right_keys:
            generated_left_keys, generated_right_keys = (
                self._generate_blocking_keys_equijoin(left_data, right_data)
            )
            left_keys.extend(generated_left_keys)
            right_keys.extend(generated_right_keys)
            left_keys = list(set(left_keys))
            right_keys = list(set(right_keys))

            # Log the generated blocking keys
            self.console.log(
                f"[bold]Generated blocking keys (for embeddings-based blocking):[/bold]"
            )
            self.console.log(f"Left keys: {left_keys}")
            self.console.log(f"Right keys: {right_keys}")

        left_embeddings, _, left_embedding_cost = self._compute_embeddings(
            left_data, keys=left_keys
        )
        right_embeddings, _, right_embedding_cost = self._compute_embeddings(
            right_data, keys=right_keys
        )
        self.console.log(
            f"[bold]Cost of creating embeddings on the sample: ${left_embedding_cost + right_embedding_cost:.4f}[/bold]"
        )

        similarities = self._calculate_cross_similarities(
            left_embeddings, right_embeddings
        )

        sampled_pairs = self._sample_pairs(similarities)
        comparison_results, comparison_cost = self._perform_comparisons_equijoin(
            left_data, right_data, sampled_pairs
        )
        self._print_similarity_histogram(similarities, comparison_results)
        while not any(result[2] for result in comparison_results):
            self.console.log(
                "[yellow]No matches found in the current sample. Resampling pairs to compare...[/yellow]"
            )
            sampled_pairs = self._sample_pairs(similarities)
            comparison_results, current_cost = self._perform_comparisons_equijoin(
                left_data, right_data, sampled_pairs
            )
            comparison_cost += current_cost
            self._print_similarity_histogram(similarities, comparison_results)

        threshold, estimated_selectivity = self._find_optimal_threshold(
            comparison_results, similarities
        )
        self.estimated_selectivity = estimated_selectivity

        blocking_rules = self._generate_blocking_rules_equijoin(
            left_keys, right_keys, left_data, right_data, comparison_results
        )

        if blocking_rules:
            false_negatives, rule_selectivity = self._verify_blocking_rule_equijoin(
                left_data,
                right_data,
                blocking_rules[0],
                left_keys,
                right_keys,
                comparison_results,
            )
            if not false_negatives and rule_selectivity <= estimated_selectivity:
                self.console.log(
                    "[green]Blocking rule verified. No false negatives detected in the sample and selectivity is within bounds.[/green]"
                )
            else:
                if false_negatives:
                    self.console.log(
                        f"[red]Blocking rule rejected. {len(false_negatives)} false negatives detected in the sample.[/red]"
                    )
                    for i, j in false_negatives[:5]:  # Show up to 5 examples
                        self.console.log(
                            f"  Filtered pair: Left: {{{', '.join(f'{key}: {left_data[i][key]}' for key in left_keys)}}} and Right: {{{', '.join(f'{key}: {right_data[j][key]}' for key in right_keys)}}}"
                        )
                    if len(false_negatives) > 5:
                        self.console.log(f"  ... and {len(false_negatives) - 5} more.")
                if rule_selectivity > estimated_selectivity:
                    self.console.log(
                        f"[red]Blocking rule rejected. Rule selectivity ({rule_selectivity:.4f}) is higher than the estimated selectivity ({estimated_selectivity:.4f}).[/red]"
                    )
                blocking_rules = (
                    []
                )  # Clear the blocking rule if it introduces false negatives or is too selective

        containment_rules = self._generate_containment_rules_equijoin(
            left_data, right_data
        )
        self.console.log(
            f"[bold]Generated {len(containment_rules)} containment rules. Please select which ones to use as blocking conditions:[/bold]"
        )
        selected_containment_rules = []
        for rule in containment_rules:
            self.console.log(f"[green]{rule}[/green]")
            # Temporarily stop the status
            if self.status:
                self.status.stop()
            # Use Rich's Confirm for input
            if Confirm.ask("Use this rule?", self.console):
                selected_containment_rules.append(rule)
            # Restart the status
            if self.status:
                self.status.start()

        if len(containment_rules) > 0:
            self.console.log(
                f"[bold]Selected {len(selected_containment_rules)} containment rules for blocking.[/bold]"
            )
        blocking_rules.extend(selected_containment_rules)

        optimized_config = self._update_config_equijoin(
            threshold, left_keys, right_keys, blocking_rules
        )
        return (
            optimized_config,
            left_embedding_cost + right_embedding_cost + comparison_cost,
            {},
        )

    def _should_apply_map_transformation(
        self,
        left_keys: List[str],
        right_keys: List[str],
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        sample_size: int = 5,
    ) -> Tuple[bool, str, str]:
        # Sample data
        left_sample = random.sample(left_data, min(sample_size, len(left_data)))
        right_sample = random.sample(right_data, min(sample_size, len(right_data)))

        # Get keys and their average lengths
        all_left_keys = {
            k: sum(len(str(d[k])) for d in left_sample) / len(left_sample)
            for k in left_sample[0].keys()
        }
        all_right_keys = {
            k: sum(len(str(d[k])) for d in right_sample) / len(right_sample)
            for k in right_sample[0].keys()
        }

        messages = [
            {
                "role": "user",
                "content": f"""Analyze the following datasets and determine if an additional LLM transformation should be applied to generate a new key-value pair for easier joining:

                Comparison prompt for the join operation: {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

                Left dataset keys and average lengths: {json.dumps(all_left_keys, indent=2)}
                Right dataset keys and average lengths: {json.dumps(all_right_keys, indent=2)}

                Left dataset sample:
                {json.dumps(left_sample, indent=2)}

                Right dataset sample:
                {json.dumps(right_sample, indent=2)}

                Current keys used for embedding-based ranking of likely matches:
                Left keys: {left_keys}
                Right keys: {right_keys}

                Consider the following:
                1. Are the current keys sufficient for accurate embedding-based ranking of likely matches? We don't want to use too many keys, or keys with too much information, as this will dilute the signal in the embeddings.
                2. Are there any keys particularly long (e.g., full text fields), containing information that is not relevant for the join operation?
                3. Is there information spread across multiple fields that could be combined?
                4. Would a summary or extraction of key information be beneficial?
                5. Is there a mismatch in information representation between the datasets?
                6. Could an additional LLM-generated field improve the accuracy of embeddings or join comparisons?

                If you believe an additional LLM transformation would be beneficial, specify which dataset (left or right) should be transformed and explain why. In most cases, you should pick the dataset with the longer keys unless there is a specific reason to pick the other dataset. Otherwise, indicate that no additional transformation is needed and explain why the current blocking keys are sufficient.""",
            }
        ]

        response = self.llm_client.generate(
            messages,
            "You are an AI expert in data analysis and entity matching.",
            {
                "type": "object",
                "properties": {
                    "apply_transformation": {"type": "boolean"},
                    "dataset_to_transform": {
                        "type": "string",
                        "enum": ["left", "right", "none"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["apply_transformation", "dataset_to_transform", "reason"],
            },
        )

        result = json.loads(response.choices[0].message.content)

        return (
            result["apply_transformation"],
            result["dataset_to_transform"],
            result["reason"],
        )

    def _generate_map_and_new_join_transformation(
        self,
        dataset_to_transform: str,
        reason: str,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        sample_size: int = 5,
    ) -> Tuple[str, str, str]:
        # Sample data
        left_sample = random.sample(left_data, min(sample_size, len(left_data)))
        right_sample = random.sample(right_data, min(sample_size, len(right_data)))

        target_data = left_sample if dataset_to_transform == "left" else right_sample

        messages = [
            {
                "role": "user",
                "content": f"""Generate an LLM prompt to transform the {dataset_to_transform} dataset for easier joining. The transformation should create a new key-value pair.

                Current comparison prompt for the join operation: {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

                Target ({dataset_to_transform}) dataset sample:
                {json.dumps(target_data, indent=2)}

                Other ({'left' if dataset_to_transform == "right" else "right"}) dataset sample:
                {json.dumps(right_sample if dataset_to_transform == "left" else left_sample, indent=2)}

                Reason for transforming {dataset_to_transform} dataset: {reason}

                Please provide:
                1. An LLM prompt to extract a smaller representation of what is relevant to the join task. The prompt should be a Jinja2 template, referring to any fields in the input data as {{ input.field_name }}. The prompt should instruct the LLM to return some **non-empty** string-valued output. The transformation should be tailored to the join task if possible, not just a generic summary of the data.
                2. A name for the new output key that will store the transformed data.
                3. An edited comparison prompt that leverages the new attribute created by the transformation. This prompt should be a Jinja2 template, referring to any fields in the input data as {{ left.field_name }} and {{ right.field_name }}. The prompt should be the same as the current comparison prompt, but with a new instruction that leverages the new attribute created by the transformation. The prompt should instruct the LLM to return a boolean-valued output, like the current comparison prompt.""",
            }
        ]

        response = self.llm_client.generate(
            messages,
            "You are an AI expert in data analysis and decomposing complex data processing pipelines.",
            {
                "type": "object",
                "properties": {
                    "extraction_prompt": {"type": "string"},
                    "output_key": {"type": "string"},
                    "new_comparison_prompt": {"type": "string"},
                },
                "required": [
                    "extraction_prompt",
                    "output_key",
                    "new_comparison_prompt",
                ],
            },
        )

        result = json.loads(response.choices[0].message.content)

        return (
            result["extraction_prompt"],
            result["output_key"],
            result["new_comparison_prompt"],
        )

    def _generate_blocking_keys_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        sample_size: int = 5,
    ) -> Tuple[List[str], List[str]]:
        # Sample data
        left_sample = random.sample(left_data, min(sample_size, len(left_data)))
        right_sample = random.sample(right_data, min(sample_size, len(right_data)))

        # Prepare sample data for LLM
        left_keys = list(left_sample[0].keys())
        right_keys = list(right_sample[0].keys())

        messages = [
            {
                "role": "user",
                "content": f"""Given the following sample data from two datasets, select appropriate blocking keys for an equijoin operation.
                The blocking process works as follows:
                1. We create embeddings for the selected keys from both datasets.
                2. We use cosine similarity between these embeddings to filter pairs for more detailed LLM comparison.
                3. Pairs with high similarity will be passed to the LLM for final comparison.

                The blocking keys should have relatively short values and be useful for generating embeddings that capture the essence of potential matches.

                Left dataset keys: {left_keys}
                Right dataset keys: {right_keys}

                Sample from left dataset:
                {json.dumps(left_sample, indent=2)}

                Sample from right dataset:
                {json.dumps(right_sample, indent=2)}

                For context, here is the comparison prompt that will be used for the more detailed LLM comparison:
                {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

                Please select one or more keys from each dataset that would be suitable for blocking. The keys should contain information that's likely to be similar in matching records and align with the comparison prompt's focus.""",
            }
        ]

        response = self.llm_client.generate(
            messages,
            "You are an expert in entity matching and database operations.",
            {
                "type": "object",
                "properties": {
                    "left_blocking_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of selected blocking keys from the left dataset",
                    },
                    "right_blocking_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of selected blocking keys from the right dataset",
                    },
                },
                "required": ["left_blocking_keys", "right_blocking_keys"],
            },
        )

        result = json.loads(response.choices[0].message.content)
        left_blocking_keys = result["left_blocking_keys"]
        right_blocking_keys = result["right_blocking_keys"]

        return left_blocking_keys, right_blocking_keys

    def _compute_embeddings(
        self,
        input_data: List[Dict[str, Any]],
        keys: List[str] = None,
        is_join: bool = True,
    ) -> Tuple[List[List[float]], List[str], float]:
        if keys is None:
            keys = self.op_config.get("blocking_keys", [])
            if not keys:
                prompt_template = self.op_config.get("comparison_prompt", "")
                prompt_vars = extract_jinja_variables(prompt_template)
                # Get rid of input, input1, input2
                prompt_vars = [
                    var
                    for var in prompt_vars
                    if var not in ["input", "input1", "input2"]
                ]

                # strip all things before . in the prompt_vars
                keys += list(set([var.split(".")[-1] for var in prompt_vars]))
            if not keys:
                self.console.log(
                    "[yellow]Warning: No blocking keys found. Using all keys for blocking.[/yellow]"
                )
                keys = list(input_data[0].keys())

        model_input_context_length = model_cost.get(
            self.op_config.get("embedding_model", "text-embedding-3-small"), {}
        ).get("max_input_tokens", 8192)
        texts = [
            " ".join(str(item[key]) for key in keys if key in item)[
                :model_input_context_length
            ]
            for item in input_data
        ]

        embeddings = []
        total_cost = 0
        batch_size = 2000
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            self.console.log(
                f"[cyan]Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}[/cyan]"
            )
            response = self.runner.api.gen_embedding(
                model=self.op_config.get("embedding_model", "text-embedding-3-small"),
                input=batch,
            )
            embeddings.extend([data["embedding"] for data in response["data"]])
            total_cost += completion_cost(response)
        embeddings = [data["embedding"] for data in response["data"]]
        cost = completion_cost(response)
        return embeddings, keys, cost

    def _calculate_cosine_similarities(
        self, embeddings: List[List[float]]
    ) -> List[Tuple[int, int, float]]:
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1)
        dot_products = np.dot(embeddings_array, embeddings_array.T)
        similarities_matrix = dot_products / np.outer(norms, norms)
        i, j = np.triu_indices(len(embeddings), k=1)
        similarities = list(
            zip(i.tolist(), j.tolist(), similarities_matrix[i, j].tolist())
        )
        return similarities

    def _print_similarity_histogram(
        self,
        similarities: List[Tuple[int, int, float]],
        comparison_results: List[Tuple[int, int, bool]],
    ):
        flat_similarities = [sim[-1] for sim in similarities if sim[-1] != 1]
        hist, bin_edges = np.histogram(flat_similarities, bins=20)
        max_bar_width, max_count = 50, max(hist)
        normalized_hist = [int(count / max_count * max_bar_width) for count in hist]

        # Create a dictionary to store true labels
        true_labels = {(i, j): is_match for i, j, is_match in comparison_results}

        self.console.log("\n[bold]Embedding Cosine Similarity Distribution:[/bold]")
        for i, count in enumerate(normalized_hist):
            bar = "█" * count
            label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"

            # Count true matches and not matches in this bin
            true_matches = 0
            not_matches = 0
            labeled_count = 0
            for sim in similarities:
                if bin_edges[i] <= sim[2] < bin_edges[i + 1]:
                    if (sim[0], sim[1]) in true_labels:
                        labeled_count += 1
                        if true_labels[(sim[0], sim[1])]:
                            true_matches += 1
                        else:
                            not_matches += 1

            # Calculate percentages of labeled pairs
            if labeled_count > 0:
                true_match_percent = (true_matches / labeled_count) * 100
                not_match_percent = (not_matches / labeled_count) * 100
            else:
                true_match_percent = 0
                not_match_percent = 0

            self.console.log(
                f"{label}: {bar} "
                f"(Labeled: {labeled_count}/{hist[i]}, [green]{true_match_percent:.1f}% match[/green], [red]{not_match_percent:.1f}% not match[/red])"
            )
        self.console.log("\n")

    def _sample_pairs(
        self, similarities: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int]]:
        # Sort similarities in descending order
        sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        # Calculate weights using exponential weighting with self.sampling_weight
        similarities_array = np.array([sim[2] for sim in sorted_similarities])
        weights = np.exp(self.sampling_weight * similarities_array)
        weights /= weights.sum()  # Normalize weights to sum to 1

        # Sample pairs based on the calculated weights
        sampled_indices = np.random.choice(
            len(sorted_similarities),
            size=min(self.sample_size, len(sorted_similarities)),
            replace=False,
            p=weights,
        )

        sampled_pairs = [
            (sorted_similarities[i][0], sorted_similarities[i][1])
            for i in sampled_indices
        ]
        return sampled_pairs

    def _calculate_cross_similarities(
        self, left_embeddings: List[List[float]], right_embeddings: List[List[float]]
    ) -> List[Tuple[int, int, float]]:
        left_array = np.array(left_embeddings)
        right_array = np.array(right_embeddings)
        dot_product = np.dot(left_array, right_array.T)
        norm_left = np.linalg.norm(left_array, axis=1)
        norm_right = np.linalg.norm(right_array, axis=1)
        similarities = dot_product / np.outer(norm_left, norm_right)
        return [
            (i, j, sim)
            for i, row in enumerate(similarities)
            for j, sim in enumerate(row)
        ]

    def _perform_comparisons_resolve(
        self, input_data: List[Dict[str, Any]], pairs: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int, bool]], float]:
        comparisons, total_cost = [], 0
        op = ResolveOperation(
            self,
            self.op_config,
            self.runner.default_model,
            self.max_threads,
            self.console,
            self.status)
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    op.compare_pair,
                    self.op_config["comparison_prompt"],
                    self.op_config.get(
                        "comparison_model", self.config.get("model", "gpt-4o-mini")
                    ),
                    input_data[i],
                    input_data[j],
                )
                for i, j in pairs
            ]
            for future, (i, j) in zip(futures, pairs):
                is_match, cost = future.result()
                comparisons.append((i, j, is_match))
                total_cost += cost

        self.console.log(
            f"[bold]Cost of pairwise comparisons on the sample: ${total_cost:.4f}[/bold]"
        )
        return comparisons, total_cost

    def _perform_comparisons_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int, bool]], float]:
        comparisons, total_cost = [], 0
        op = EquijoinOperation(
            self,
            self.op_config,
            self.runner.default_model,
            self.max_threads,
            self.console,
            self.status)
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    op.compare_pair,
                    self.op_config["comparison_prompt"],
                    self.op_config.get(
                        "comparison_model", self.config.get("model", "gpt-4o-mini")
                    ),
                    left_data[i],
                    right_data[j] if right_data else left_data[j],
                )
                for i, j in pairs
            ]
            for future, (i, j) in zip(futures, pairs):
                is_match, cost = future.result()
                comparisons.append((i, j, is_match))
                total_cost += cost

        self.console.log(
            f"[bold]Cost of pairwise comparisons on the sample: ${total_cost:.4f}[/bold]"
        )
        return comparisons, total_cost

    def _find_optimal_threshold(
        self,
        comparisons: List[Tuple[int, int, bool]],
        similarities: List[Tuple[int, int, float]],
    ) -> Tuple[float, float, float]:
        true_labels = np.array([comp[2] for comp in comparisons])
        sim_dict = {(i, j): sim for i, j, sim in similarities}
        sim_scores = np.array([sim_dict[(i, j)] for i, j, _ in comparisons])

        thresholds = np.linspace(0, 1, 100)
        precisions, recalls = [], []

        for threshold in thresholds:
            predictions = sim_scores >= threshold
            tp = np.sum(predictions & true_labels)
            fp = np.sum(predictions & ~true_labels)
            fn = np.sum(~predictions & true_labels)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        valid_indices = [i for i, r in enumerate(recalls) if r >= self.target_recall]
        if not valid_indices:
            optimal_threshold = float(thresholds[np.argmax(recalls)])
        else:
            optimal_threshold = float(thresholds[max(valid_indices)])

        # Improved selectivity estimation
        all_similarities = np.array([s[2] for s in similarities])
        sampled_similarities = sim_scores

        # Calculate sampling probabilities
        sampling_probs = np.exp(self.sampling_weight * sampled_similarities)
        sampling_probs /= sampling_probs.sum()

        # Estimate selectivity using importance sampling
        weights = 1 / (len(all_similarities) * sampling_probs)
        numerator = np.sum(weights * true_labels)
        denominator = np.sum(weights)
        selectivity_estimate = numerator / denominator

        self.console.log(
            "[bold cyan]┌─ Estimated Self-Join Selectivity ─────────────────────────┐[/bold cyan]"
        )
        self.console.log(
            f"[bold cyan]│[/bold cyan] [yellow]Target Recall:[/yellow] {self.target_recall:.0%}"
        )
        self.console.log(
            f"[bold cyan]│[/bold cyan] [yellow]Estimate:[/yellow] {selectivity_estimate:.4f}"
        )
        self.console.log(
            "[bold cyan]└───────────────────────────────────────────────────────────┘[/bold cyan]"
        )
        self.console.log(
            f"[bold]Chosen similarity threshold for blocking: {optimal_threshold:.4f}[/bold]"
        )

        return round(optimal_threshold, 4), selectivity_estimate

    def _generate_blocking_rules(
        self,
        blocking_keys: List[str],
        input_data: List[Dict[str, Any]],
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[str]:
        # Sample 2 true and 2 false comparisons
        true_comparisons = [comp for comp in comparisons if comp[2]][:2]
        false_comparisons = [comp for comp in comparisons if not comp[2]][:2]
        sample_datas = [
            (
                {key: input_data[i][key] for key in blocking_keys},
                {key: input_data[j][key] for key in blocking_keys},
                is_match,
            )
            for i, j, is_match in true_comparisons + false_comparisons
        ]

        messages = [
            {
                "role": "user",
                "content": f"""Given the following sample comparisons between entities, generate a single-line Python statement that acts as a blocking rule for entity resolution. This rule will be used in the form: `eval(blocking_rule, {{"input1": item1, "input2": item2}})`.

    Sample comparisons (note: these are just a few examples and may not represent all possible cases):
    {json.dumps(sample_datas, indent=2)}

    For context, here is the comparison prompt that will be used for the more expensive, detailed comparison:
    {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

    Please generate ONE one-line blocking rule that adheres to the following criteria:
    1. The rule should evaluate to True if the entities are possibly a match and require further comparison.
    2. The rule should evaluate to False ONLY if the entities are definitely not a match.
    3. The rule must be a single Python expression that can be evaluated using the eval() function.
    4. The rule should be much faster to evaluate than the full comparison prompt.
    5. The rule should capture the essence of the comparison prompt but in a simplified manner.
    6. The rule should be general enough to work well on the entire dataset, not just these specific examples.
    7. The rule should handle inconsistent casing by using string methods like .lower() when comparing string values.
    8. The rule should err on the side of inclusivity - it's better to have false positives than false negatives.

    Example structure of a one-line blocking rule:
    "(condition1) or (condition2) or (condition3)"

    Where conditions could be comparisons like:
    "input1['field'].lower() == input2['field'].lower()"
    "abs(len(input1['text']) - len(input2['text'])) <= 5"
    "any(word in input1['description'].lower() for word in input2['description'].lower().split())"

    If there's no clear rule that can be generated based on the given information, return the string "True" to ensure all pairs are compared.

    Remember, the primary goal of the blocking rule is to safely reduce the number of comparisons by quickly identifying pairs that are definitely not matches, while keeping all potential matches for further evaluation.""",
            }
        ]

        for attempt in range(self.agent_max_retries):  # Up to 3 attempts
            # Generate blocking rule using the LLM
            response = self.llm_client.generate(
                messages,
                "You are an expert in entity resolution and Python programming. Your task is to generate one efficient blocking rule based on the given sample comparisons and data structure.",
                {
                    "type": "object",
                    "properties": {
                        "blocking_rule": {
                            "type": "string",
                            "description": "One-line Python statement acting as a blocking rule",
                        }
                    },
                    "required": ["blocking_rule"],
                },
            )

            # Extract the blocking rule from the LLM response
            blocking_rule = response.choices[0].message.content
            blocking_rule = json.loads(blocking_rule).get("blocking_rule")

            if blocking_rule:
                self.console.log("")  # Print a newline

                if blocking_rule.strip() == "True":
                    self.console.log(
                        "[yellow]No suitable blocking rule could be found. Proceeding without a blocking rule.[/yellow]"
                    )
                    return []

                self.console.log(
                    f"[bold]Generated blocking rule (Attempt {attempt + 1}):[/bold] {blocking_rule}"
                )

                # Test the blocking rule
                filtered_pairs = self._test_blocking_rule(
                    input_data, blocking_keys, blocking_rule, comparisons
                )

                if not filtered_pairs:
                    self.console.log(
                        "[green]Blocking rule looks good! No known matches were filtered out.[/green]"
                    )
                    return [blocking_rule]
                else:
                    feedback = f"The previous rule incorrectly filtered out {len(filtered_pairs)} known matches. "
                    feedback += (
                        "Here are up to 3 examples of incorrectly filtered pairs:\n"
                    )
                    for i, j in filtered_pairs[:3]:
                        feedback += f"Item 1: {json.dumps({key: input_data[i][key] for key in blocking_keys})}\nItem 2: {json.dumps({key: input_data[j][key] for key in blocking_keys})}\n"
                        feedback += "These pairs are known matches but were filtered out by the rule.\n"
                    feedback += "Please generate a new rule that doesn't filter out these matches."

                    messages.append({"role": "assistant", "content": blocking_rule})
                    messages.append({"role": "user", "content": feedback})
            else:
                self.console.log("[yellow]No blocking rule generated.[/yellow]")
                return []

        self.console.log(
            f"[yellow]Failed to generate a suitable blocking rule after {self.agent_max_retries} attempts. Proceeding without a blocking rule.[/yellow]"
        )
        return []

    def _test_blocking_rule(
        self,
        input_data: List[Dict[str, Any]],
        blocking_keys: List[str],
        blocking_rule: str,
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[Tuple[int, int]]:
        def apply_blocking_rule(item1, item2):
            try:
                return eval(blocking_rule, {"input1": item1, "input2": item2})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        filtered_pairs = []

        for i, j, is_match in comparisons:
            if is_match:
                item1 = {
                    k: input_data[i][k] for k in blocking_keys if k in input_data[i]
                }
                item2 = {
                    k: input_data[j][k] for k in blocking_keys if k in input_data[j]
                }

                if not apply_blocking_rule(item1, item2):
                    filtered_pairs.append((i, j))

        if filtered_pairs:
            self.console.log(
                f"[yellow italic]LLM Correction: The blocking rule incorrectly filtered out {len(filtered_pairs)} known positive matches.[/yellow italic]"
            )
            for i, j in filtered_pairs[:5]:  # Show up to 5 examples
                self.console.log(
                    f"  Incorrectly filtered pair 1: {json.dumps({key: input_data[i][key] for key in blocking_keys})}  and pair 2: {json.dumps({key: input_data[j][key] for key in blocking_keys})}"
                )
            if len(filtered_pairs) > 5:
                self.console.log(
                    f"  ... and {len(filtered_pairs) - 5} more incorrect pairs."
                )

        return filtered_pairs

    def _generate_containment_rules_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
    ) -> List[str]:
        # Get all available keys from the sample data
        left_keys = set(left_data[0].keys())
        right_keys = set(right_data[0].keys())

        # Sample a few records from each dataset
        sample_left = random.sample(left_data, min(3, len(left_data)))
        sample_right = random.sample(right_data, min(3, len(right_data)))

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant tasked with generating containment-based blocking rules for an equijoin operation.",
            },
            {
                "role": "user",
                "content": f"""Generate multiple one-line Python statements that act as containment-based blocking rules for equijoin. These rules will be used in the form: `eval(blocking_rule, {{"left": item1, "right": item2}})`.

Available keys in left dataset: {', '.join(left_keys)}
Available keys in right dataset: {', '.join(right_keys)}

Sample data from left dataset:
{json.dumps(sample_left, indent=2)}

Sample data from right dataset:
{json.dumps(sample_right, indent=2)}

Comparison prompt used for detailed comparison:
{self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

Please generate multiple one-line blocking rules that adhere to the following criteria:
1. The rules should focus on containment relationships between fields in the left and right datasets. Containment can mean that the left field contains all the words in the right field, or the right field contains all the words in the left field.
2. Each rule should evaluate to True if there's a potential match based on containment, False otherwise.
3. Rules must be single Python expressions that can be evaluated using the eval() function.
4. Rules should handle inconsistent casing by using string methods like .lower() when comparing string values.
5. Consider the length of the fields when generating rules: for example, if the left field is much longer than the right field, it's more likely to contain all the words in the right field.

Example structures of containment-based blocking rules:
"all(word in left['{{left_key}}'].lower() for word in right['{{right_key}}'].lower().split())"
"any(word in right['{{right_key}}'].lower().split() for word in left['{{left_key}}'].lower().split())"

Please provide 3-5 different containment-based blocking rules, based on the keys and sample data provided.""",
            },
        ]

        response = self.llm_client.generate(
            messages,
            "You are an expert in data matching and Python programming.",
            {
                "type": "object",
                "properties": {
                    "containment_rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of containment-based blocking rules as Python expressions",
                    }
                },
                "required": ["containment_rules"],
            },
        )

        containment_rules = response.choices[0].message.content
        containment_rules = json.loads(containment_rules).get("containment_rules")
        return containment_rules

    def _generate_blocking_rules_equijoin(
        self,
        left_keys: List[str],
        right_keys: List[str],
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[str]:
        if not left_keys or not right_keys:
            left_keys = list(left_data[0].keys())
            right_keys = list(right_data[0].keys())

        # Sample 2 true and 2 false comparisons
        true_comparisons = [comp for comp in comparisons if comp[2]][:2]
        false_comparisons = [comp for comp in comparisons if not comp[2]][:2]
        sample_datas = [
            (
                {key: left_data[i][key] for key in left_keys if key in left_data[i]},
                {key: right_data[j][key] for key in right_keys if key in right_data[j]},
                is_match,
            )
            for i, j, is_match in true_comparisons + false_comparisons
        ]

        messages = [
            {
                "role": "user",
                "content": f"""Given the following sample comparisons between entities, generate a single-line Python statement that acts as a blocking rule for equijoin. This rule will be used in the form: `eval(blocking_rule, {{"left": item1, "right": item2}})`.

    Sample comparisons (note: these are just a few examples and may not represent all possible cases):
    {json.dumps(sample_datas, indent=2)}

    For context, here is the comparison prompt that will be used for the more expensive, detailed comparison:
    {self.op_config.get('comparison_prompt', 'No comparison prompt provided.')}

    Please generate ONE one-line blocking rule that adheres to the following criteria:
    1. The rule should evaluate to True if the entities are possibly a match and require further comparison.
    2. The rule should evaluate to False ONLY if the entities are definitely not a match.
    3. The rule must be a single Python expression that can be evaluated using the eval() function.
    4. The rule should be much faster to evaluate than the full comparison prompt.
    5. The rule should capture the essence of the comparison prompt but in a simplified manner.
    6. The rule should be general enough to work well on the entire dataset, not just these specific examples.
    7. The rule should handle inconsistent casing by using string methods like .lower() when comparing string values.
    8. The rule should err on the side of inclusivity - it's better to have false positives than false negatives.

    Example structure of a one-line blocking rule:
    "(condition1) or (condition2) or (condition3)"

    Where conditions could be comparisons like:
    "left['{left_keys[0]}'].lower() == right['{right_keys[0]}'].lower()"
    "abs(len(left['{left_keys[0]}']) - len(right['{right_keys[0]}'])) <= 5"
    "any(word in left['{left_keys[0]}'].lower() for word in right['{right_keys[0]}'].lower().split())"

    If there's no clear rule that can be generated based on the given information, return the string "True" to ensure all pairs are compared.

    Remember, the primary goal of the blocking rule is to safely reduce the number of comparisons by quickly identifying pairs that are definitely not matches, while keeping all potential matches for further evaluation.""",
            }
        ]

        for attempt in range(self.agent_max_retries):
            response = self.llm_client.generate(
                messages,
                "You are an expert in entity resolution and Python programming. Your task is to generate one efficient blocking rule based on the given sample comparisons and data structure.",
                {
                    "type": "object",
                    "properties": {
                        "blocking_rule": {
                            "type": "string",
                            "description": "One-line Python statement acting as a blocking rule",
                        }
                    },
                    "required": ["blocking_rule"],
                },
            )

            blocking_rule = response.choices[0].message.content
            blocking_rule = json.loads(blocking_rule).get("blocking_rule")

            if blocking_rule:
                self.console.log("")

                if blocking_rule.strip() == "True":
                    self.console.log(
                        "[yellow]No suitable blocking rule could be found. Proceeding without a blocking rule.[/yellow]"
                    )
                    return []

                self.console.log(
                    f"[bold]Generated blocking rule (Attempt {attempt + 1}):[/bold] {blocking_rule}"
                )

                # Test the blocking rule
                filtered_pairs = self._test_blocking_rule_equijoin(
                    left_data,
                    right_data,
                    left_keys,
                    right_keys,
                    blocking_rule,
                    comparisons,
                )

                if not filtered_pairs:
                    self.console.log(
                        "[green]Blocking rule looks good! No known matches were filtered out.[/green]"
                    )
                    return [blocking_rule]
                else:
                    feedback = f"The previous rule incorrectly filtered out {len(filtered_pairs)} known matches. "
                    feedback += (
                        "Here are up to 3 examples of incorrectly filtered pairs:\n"
                    )
                    for i, j in filtered_pairs[:3]:
                        feedback += f"Left: {json.dumps({key: left_data[i][key] for key in left_keys})}\n"
                        feedback += f"Right: {json.dumps({key: right_data[j][key] for key in right_keys})}\n"
                        feedback += "These pairs are known matches but were filtered out by the rule.\n"
                    feedback += "Please generate a new rule that doesn't filter out these matches."

                    messages.append({"role": "assistant", "content": blocking_rule})
                    messages.append({"role": "user", "content": feedback})
            else:
                self.console.log("[yellow]No blocking rule generated.[/yellow]")
                return []

        self.console.log(
            f"[yellow]Failed to generate a suitable blocking rule after {self.agent_max_retries} attempts. Proceeding without a blocking rule.[/yellow]"
        )
        return []

    def _test_blocking_rule_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        left_keys: List[str],
        right_keys: List[str],
        blocking_rule: str,
        comparisons: List[Tuple[int, int, bool]],
    ) -> List[Tuple[int, int]]:
        def apply_blocking_rule(left, right):
            try:
                return eval(blocking_rule, {"left": left, "right": right})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        filtered_pairs = []

        for i, j, is_match in comparisons:
            if is_match:
                left = left_data[i]
                right = right_data[j]
                if not apply_blocking_rule(left, right):
                    filtered_pairs.append((i, j))

        if filtered_pairs:
            self.console.log(
                f"[yellow italic]LLM Correction: The blocking rule incorrectly filtered out {len(filtered_pairs)} known positive matches.[/yellow italic]"
            )
            for i, j in filtered_pairs[:5]:  # Show up to 5 examples
                left_dict = {key: left_data[i][key] for key in left_keys}
                right_dict = {key: right_data[j][key] for key in right_keys}
                self.console.log(
                    f"  Incorrectly filtered pair - Left: {json.dumps(left_dict)}  Right: {json.dumps(right_dict)}"
                )
            if len(filtered_pairs) > 5:
                self.console.log(
                    f"  ... and {len(filtered_pairs) - 5} more incorrect pairs."
                )

        return filtered_pairs

    def _verify_blocking_rule_equijoin(
        self,
        left_data: List[Dict[str, Any]],
        right_data: List[Dict[str, Any]],
        blocking_rule: str,
        left_keys: List[str],
        right_keys: List[str],
        comparison_results: List[Tuple[int, int, bool]],
    ) -> Tuple[List[Tuple[int, int]], float]:
        def apply_blocking_rule(left, right):
            try:
                return eval(blocking_rule, {"left": left, "right": right})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        false_negatives = []
        total_pairs = 0
        blocked_pairs = 0

        for i, j, is_match in comparison_results:
            total_pairs += 1
            left = left_data[i]
            right = right_data[j]
            if apply_blocking_rule(left, right):
                blocked_pairs += 1
                if is_match:
                    false_negatives.append((i, j))

        rule_selectivity = blocked_pairs / total_pairs if total_pairs > 0 else 0

        return false_negatives, rule_selectivity

    def _update_config_equijoin(
        self,
        threshold: float,
        left_keys: List[str],
        right_keys: List[str],
        blocking_rules: List[str],
    ) -> Dict[str, Any]:
        optimized_config = self.op_config.copy()
        optimized_config["blocking_keys"] = {
            "left": left_keys,
            "right": right_keys,
        }
        optimized_config["blocking_threshold"] = threshold
        if blocking_rules:
            optimized_config["blocking_conditions"] = blocking_rules
        if "embedding_model" not in optimized_config:
            optimized_config["embedding_model"] = "text-embedding-3-small"
        return optimized_config

    def _verify_blocking_rule(
        self,
        input_data: List[Dict[str, Any]],
        blocking_rule: str,
        blocking_keys: List[str],
        comparison_results: List[Tuple[int, int, bool]],
    ) -> Tuple[List[Tuple[int, int]], float]:
        def apply_blocking_rule(item1, item2):
            try:
                return eval(blocking_rule, {"input1": item1, "input2": item2})
            except Exception as e:
                self.console.log(f"[red]Error applying blocking rule: {e}[/red]")
                return True  # If there's an error, we default to comparing the pair

        false_negatives = []
        total_pairs = 0
        blocked_pairs = 0

        for i, j, is_match in comparison_results:
            total_pairs += 1
            item1 = {k: input_data[i][k] for k in blocking_keys if k in input_data[i]}
            item2 = {k: input_data[j][k] for k in blocking_keys if k in input_data[j]}

            if apply_blocking_rule(item1, item2):
                blocked_pairs += 1
                if is_match:
                    false_negatives.append((i, j))

        rule_selectivity = blocked_pairs / total_pairs if total_pairs > 0 else 0

        return false_negatives, rule_selectivity

    def _update_config(
        self, threshold: float, blocking_keys: List[str], blocking_rules: List[str]
    ) -> Dict[str, Any]:
        optimized_config = self.op_config.copy()
        optimized_config["blocking_keys"] = blocking_keys
        optimized_config["blocking_threshold"] = threshold
        if blocking_rules:
            optimized_config["blocking_conditions"] = blocking_rules
        if "embedding_model" not in optimized_config:
            optimized_config["embedding_model"] = "text-embedding-3-small"
        return optimized_config
