import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from jinja2 import Template
from litellm import model_cost
from rich.console import Console
from rich.prompt import Confirm
from rich.status import Status

from docetl.operations.base import BaseOperation
from docetl.operations.utils import truncate_messages
from docetl.optimizers.join_optimizer import JoinOptimizer
from docetl.optimizers.utils import LLMClient
from docetl.utils import count_tokens, extract_jinja_variables


class ReduceOptimizer:
    """
    A class that optimizes reduce operations in data processing pipelines.

    This optimizer analyzes the input and output of a reduce operation, creates and evaluates
    multiple reduce plans, and selects the best plan for optimizing the operation's performance.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the optimizer.
        console (Console): Rich console object for pretty printing.
        llm_client (LLMClient): Client for interacting with a language model.
        _run_operation (Callable): Function to run an operation.
        max_threads (int): Maximum number of threads to use for parallel processing.
        num_fold_prompts (int): Number of fold prompts to generate.
        num_samples_in_validation (int): Number of samples to use in validation.
    """

    def __init__(
        self,
        runner,
        config: Dict[str, Any],
        console: Console,
        llm_client: LLMClient,
        max_threads: int,
        run_operation: Callable,
        num_fold_prompts: int = 1,
        num_samples_in_validation: int = 10,
        status: Optional[Status] = None,
    ):
        """
        Initialize the ReduceOptimizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the optimizer.
            console (Console): Rich console object for pretty printing.
            llm_client (LLMClient): Client for interacting with a language model.
            max_threads (int): Maximum number of threads to use for parallel processing.
            run_operation (Callable): Function to run an operation.
            num_fold_prompts (int, optional): Number of fold prompts to generate. Defaults to 1.
            num_samples_in_validation (int, optional): Number of samples to use in validation. Defaults to 10.
        """
        self.runner = runner
        self.config = config
        self.console = console
        self.llm_client = llm_client
        self._run_operation = run_operation
        self.max_threads = max_threads
        self.num_fold_prompts = num_fold_prompts
        self.num_samples_in_validation = num_samples_in_validation
        self.status = status

    def optimize(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        level: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Optimize the reduce operation based on the given configuration and input data.

        This method performs the following steps:
        1. Run the original operation
        2. Generate a validator prompt
        3. Validate the output
        4. If improvement is needed:
           a. Evaluate if decomposition is beneficial
           b. If decomposition is beneficial, recursively optimize each sub-operation
           c. If not, proceed with single operation optimization
        5. Run the optimized operation(s)

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]: A tuple containing the list of optimized configurations
            and the list of outputs from the optimized operation(s), and the cost of the operation due to synthesizing any resolve operations.
        """
        # Check if we're running out of token limits for the reduce prompt
        model = op_config.get("model", self.config.get("default_model", "gpt-4o-mini"))
        model_input_context_length = model_cost.get(model, {}).get(
            "max_input_tokens", 4096
        )

        # Find the key with the longest value
        longest_key = max(
            op_config["reduce_key"], key=lambda k: len(str(input_data[0][k]))
        )
        sample_key = tuple(
            input_data[0][k] if k == longest_key else input_data[0][k]
            for k in op_config["reduce_key"]
        )

        # Render the prompt with a sample input
        prompt_template = Template(op_config["prompt"])
        sample_prompt = prompt_template.render(
            reduce_key=dict(zip(op_config["reduce_key"], sample_key)),
            inputs=[input_data[0]],
        )

        # Count tokens in the sample prompt
        prompt_tokens = count_tokens(sample_prompt, model)

        add_map_op = False
        if prompt_tokens * 2 > model_input_context_length:
            add_map_op = True
            self.console.log(
                f"[yellow]Warning: The reduce prompt exceeds the token limit for model {model}. "
                f"Token count: {prompt_tokens}, Limit: {model_input_context_length}. "
                f"Add a map operation to the pipeline.[/yellow]"
            )

        # # Also query an agent to look at a sample of the inputs and see if they think a map operation would be helpful
        # preprocessing_steps = ""
        # should_use_map, preprocessing_steps = self._should_use_map(
        #     op_config, input_data
        # )
        # if should_use_map or add_map_op:
        #     # Synthesize a map operation
        #     map_prompt, map_output_schema = self._synthesize_map_operation(
        #         op_config, preprocessing_steps, input_data
        #     )
        #     # Change the reduce operation prompt to use the map schema
        #     new_reduce_prompt = self._change_reduce_prompt_to_use_map_schema(
        #         op_config["prompt"], map_output_schema
        #     )
        #     op_config["prompt"] = new_reduce_prompt

        #     # Return unoptimized map and reduce operations
        #     return [map_prompt, op_config], input_data, 0.0

        original_output = self._run_operation(op_config, input_data)

        # Step 1: Synthesize a validator prompt
        validator_prompt = self._generate_validator_prompt(
            op_config, input_data, original_output
        )

        # Log the validator prompt
        self.console.log("[bold]Validator Prompt:[/bold]")
        self.console.log(validator_prompt)
        self.console.log("\n")  # Add a newline for better readability

        # Step 2: validate the output
        validator_inputs = self._create_validation_inputs(
            input_data, op_config["reduce_key"]
        )
        validation_results = self._validate_reduce_output(
            op_config, validator_inputs, original_output, validator_prompt
        )

        # Print the validation results
        self.console.log("[bold]Validation Results on Initial Sample:[/bold]")
        if validation_results["needs_improvement"]:
            self.console.log(
                "\n".join(
                    [
                        f"Issues: {result['issues']} Suggestions: {result['suggestions']}"
                        for result in validation_results["validation_results"]
                    ]
                )
            )

            # Step 3: Evaluate if decomposition is beneficial
            decomposition_result = self._evaluate_decomposition(
                op_config, input_data, level
            )

            if decomposition_result["should_decompose"]:
                return self._optimize_decomposed_reduce(
                    decomposition_result, op_config, input_data, level
                )

            return self._optimize_single_reduce(op_config, input_data, validator_prompt)
        else:
            self.console.log("No improvements identified.")
            return [op_config], original_output, 0.0

    def _should_use_map(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Determine if a map operation should be used based on the input data.
        """
        # Sample a random input item
        sample_input = random.choice(input_data)

        # Format the prompt with the sample input
        prompt_template = Template(op_config["prompt"])
        formatted_prompt = prompt_template.render(
            reduce_key=dict(
                zip(op_config["reduce_key"], sample_input[op_config["reduce_key"]])
            ),
            inputs=[sample_input],
        )

        # Prepare the message for the LLM
        messages = [{"role": "user", "content": formatted_prompt}]

        # Truncate the messages to fit the model's context window
        truncated_messages = truncate_messages(
            messages, self.config.get("model", self.default_model)
        )

        # Query the LLM for preprocessing suggestions
        preprocessing_prompt = (
            "Based on the following reduce operation prompt, should we do any preprocessing on the input data? "
            "Consider if we need to remove unnecessary context, or logically construct an output that will help in the task. "
            "If preprocessing would be beneficial, explain why and suggest specific steps. If not, explain why preprocessing isn't necessary.\n\n"
            f"Reduce operation prompt:\n{truncated_messages[0]['content']}"
        )

        preprocessing_response = self.llm_client.generate(
            model=self.config.get("model", self.default_model),
            messages=[{"role": "user", "content": preprocessing_prompt}],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "preprocessing_needed": {"type": "boolean"},
                        "rationale": {"type": "string"},
                        "suggested_steps": {"type": "string"},
                    },
                    "required": [
                        "preprocessing_needed",
                        "rationale",
                        "suggested_steps",
                    ],
                },
            },
        )

        preprocessing_result = preprocessing_response.choices[0].message.content

        should_preprocess = preprocessing_result["preprocessing_needed"]
        preprocessing_rationale = preprocessing_result["rationale"]

        self.console.log(f"[bold]Map-Reduce Decomposition Analysis:[/bold]")
        self.console.log(f"Should write a map operation: {should_preprocess}")
        self.console.log(f"Rationale: {preprocessing_rationale}")

        if should_preprocess:
            self.console.log(
                f"Suggested steps: {preprocessing_result['suggested_steps']}"
            )

        return should_preprocess, preprocessing_result["suggested_steps"]

    def _optimize_single_reduce(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Optimize a single reduce operation.

        This method performs the following steps:
        1. Determine and configure value sampling
        2. Determine if the reduce operation is associative
        3. Create and evaluate multiple reduce plans
        4. Run the best reduce plan

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            validator_prompt (str): The validator prompt for evaluating reduce plans.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]: A tuple containing a single-item list with the optimized configuration
            and a single-item list with the output from the optimized operation, and the cost of the operation due to synthesizing any resolve operations.
        """
        # Step 1: Determine and configure value sampling (TODO: re-enable this when the agent is more reliable)
        # value_sampling_config = self._determine_value_sampling(op_config, input_data)
        # if value_sampling_config["enabled"]:
        #     op_config["value_sampling"] = value_sampling_config
        #     self.console.log("[bold]Value Sampling Configuration:[/bold]")
        #     self.console.log(json.dumps(value_sampling_config, indent=2))

        # Step 2: Determine if the reduce operation is associative
        is_associative = self._is_associative(op_config, input_data)

        # Step 3: Create and evaluate multiple reduce plans
        self.console.log("[bold magenta]Generating batched plans...[/bold magenta]")
        reduce_plans = self._create_reduce_plans(op_config, input_data, is_associative)

        # Create gleaning plans
        self.console.log("[bold magenta]Generating gleaning plans...[/bold magenta]")
        gleaning_plans = self._generate_gleaning_plans(reduce_plans, validator_prompt)

        self.console.log("[bold magenta]Evaluating plans...[/bold magenta]")
        best_plan = self._evaluate_reduce_plans(
            op_config, reduce_plans + gleaning_plans, input_data, validator_prompt
        )

        # Step 4: Run the best reduce plan
        optimized_output = self._run_operation(best_plan, input_data)

        return [best_plan], optimized_output, 0.0

    def _generate_gleaning_plans(
        self,
        plans: List[Dict[str, Any]],
        validation_prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate plans that use gleaning for the given operation.

        Gleaning involves iteratively refining the output of an operation
        based on validation feedback. This method creates plans with different
        numbers of gleaning rounds.

        Args:
            plans (List[Dict[str, Any]]): The list of plans to use for gleaning.
            validation_prompt (str): The prompt used for validating the operation's output.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary of gleaning plans, where each key
            is a plan name and each value is a list containing a single operation configuration
            with gleaning parameters.

        """
        # Generate an op with gleaning num_rounds and validation_prompt
        gleaning_plans = []
        gleaning_rounds = [1]
        biggest_batch_size = max([plan["fold_batch_size"] for plan in plans])
        for plan in plans:
            if plan["fold_batch_size"] != biggest_batch_size:
                continue
            for gleaning_round in gleaning_rounds:
                plan_copy = copy.deepcopy(plan)
                plan_copy["gleaning"] = {
                    "num_rounds": gleaning_round,
                    "validation_prompt": validation_prompt,
                }
                plan_name = f"gleaning_{gleaning_round}_rounds_{plan['name']}"
                plan_copy["name"] = plan_name
                gleaning_plans.append(plan_copy)
        return gleaning_plans

    def _optimize_decomposed_reduce(
        self,
        decomposition_result: Dict[str, Any],
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        level: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Optimize a decomposed reduce operation.

        This method performs the following steps:
        1. Group the input data by the sub-group key.
        2. Optimize the first reduce operation.
        3. Run the optimized first reduce operation on all groups.
        4. Optimize the second reduce operation using the results of the first.
        5. Run the optimized second reduce operation.

        Args:
            decomposition_result (Dict[str, Any]): The result of the decomposition evaluation.
            op_config (Dict[str, Any]): The original reduce operation configuration.
            input_data (List[Dict[str, Any]]): The input data for the reduce operation.
            level (int): The current level of decomposition.
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]: A tuple containing the list of optimized configurations
            for both reduce operations and the final output of the second reduce operation, and the cost of the operation due to synthesizing any resolve operations.
        """
        sub_group_key = decomposition_result["sub_group_key"]
        first_reduce_prompt = decomposition_result["first_reduce_prompt"]
        second_reduce_prompt = decomposition_result["second_reduce_prompt"]
        pipeline = []
        all_cost = 0.0

        first_reduce_config = op_config.copy()
        first_reduce_config["prompt"] = first_reduce_prompt
        if isinstance(op_config["reduce_key"], list):
            first_reduce_config["reduce_key"] = [sub_group_key] + op_config[
                "reduce_key"
            ]
        else:
            first_reduce_config["reduce_key"] = [sub_group_key, op_config["reduce_key"]]
        first_reduce_config["pass_through"] = True

        if first_reduce_config.get("synthesize_resolve", True):
            resolve_config = {
                "type": "resolve",
                "empty": True,
                "embedding_model": "text-embedding-3-small",
                "resolution_model": self.config.get("default_model", "gpt-4o-mini"),
                "comparison_model": self.config.get("default_model", "gpt-4o-mini"),
                "_intermediates": {
                    "map_prompt": op_config.get("_intermediates", {}).get(
                        "last_map_prompt"
                    ),
                    "reduce_key": first_reduce_config["reduce_key"],
                },
            }
            optimized_resolve_config, resolve_cost = JoinOptimizer(
                self.config,
                resolve_config,
                self.console,
                self.llm_client,
                self.max_threads,
            ).optimize_resolve(input_data)
            all_cost += resolve_cost

            if not optimized_resolve_config.get("empty", False):
                # Add this to the pipeline
                pipeline += [optimized_resolve_config]

                # Run the resolver
                optimized_output = self._run_operation(
                    optimized_resolve_config, input_data
                )
                input_data = optimized_output

        first_optimized_configs, first_outputs, first_cost = self.optimize(
            first_reduce_config, input_data, level + 1
        )
        pipeline += first_optimized_configs
        all_cost += first_cost

        # Optimize second reduce operation
        second_reduce_config = op_config.copy()
        second_reduce_config["prompt"] = second_reduce_prompt
        second_reduce_config["pass_through"] = True

        second_optimized_configs, second_outputs, second_cost = self.optimize(
            second_reduce_config, first_outputs, level + 1
        )

        # Combine optimized configs and return with final output
        pipeline += second_optimized_configs
        all_cost += second_cost

        return pipeline, second_outputs, all_cost

    def _evaluate_decomposition(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        level: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate whether decomposing the reduce operation would be beneficial.

        This method first determines if decomposition would be helpful, and if so,
        it then determines the sub-group key and prompts for the decomposed operations.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            level (int): The current level of decomposition.

        Returns:
            Dict[str, Any]: A dictionary containing the decomposition decision and details.
        """
        should_decompose = self._should_decompose(op_config, input_data, level)

        # Log the decomposition decision
        if should_decompose["should_decompose"]:
            self.console.log(
                f"[bold green]Decomposition recommended:[/bold green] {should_decompose['explanation']}"
            )
        else:
            self.console.log(
                f"[bold yellow]Decomposition not recommended:[/bold yellow] {should_decompose['explanation']}"
            )

        # Return early if decomposition is not recommended
        if not should_decompose["should_decompose"]:
            return should_decompose

        # Temporarily stop the status
        if self.status:
            self.status.stop()

        # Ask user if they agree with the decomposition assessment
        user_agrees = Confirm.ask(
            f"Do you agree with the decomposition assessment? "
            f"[bold]{'Recommended' if should_decompose['should_decompose'] else 'Not recommended'}[/bold]",
            self.console,
        )

        # If user disagrees, invert the decomposition decision
        if not user_agrees:
            should_decompose["should_decompose"] = not should_decompose[
                "should_decompose"
            ]
            should_decompose["explanation"] = (
                "User disagreed with the initial assessment."
            )

        # Restart the status
        if self.status:
            self.status.start()

        # Return if decomposition is not recommended
        if not should_decompose["should_decompose"]:
            return should_decompose

        decomposition_details = self._get_decomposition_details(op_config, input_data)
        result = {**should_decompose, **decomposition_details}
        if decomposition_details["sub_group_key"] in op_config["reduce_key"]:
            result["should_decompose"] = False
            result[
                "explanation"
            ] += " However, the suggested sub-group key is already part of the current reduce key(s), so decomposition is not recommended."
            result["sub_group_key"] = ""

        return result

    def _should_decompose(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        level: int = 1,
    ) -> Dict[str, Any]:
        """
        Determine if decomposing the reduce operation would be beneficial.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            level (int): The current level of decomposition.

        Returns:
            Dict[str, Any]: A dictionary containing the decomposition decision and explanation.
        """
        # TODO: we have not enabled recursive decomposition yet
        if level > 1 and not op_config.get("recursively_optimize", False):
            return {
                "should_decompose": False,
                "explanation": "Recursive decomposition is not enabled.",
            }

        system_prompt = (
            "You are an AI assistant tasked with optimizing data processing pipelines."
        )

        # Sample a subset of input data for analysis
        sample_size = min(10, len(input_data))
        sample_input = random.sample(input_data, sample_size)

        # Get all keys from the input data
        all_keys = set().union(*(item.keys() for item in sample_input))
        reduce_key = op_config["reduce_key"]
        reduce_keys = [reduce_key] if isinstance(reduce_key, str) else reduce_key
        other_keys = [key for key in all_keys if key not in reduce_keys]

        # See if there's an input schema and constrain the sample_input to that schema
        input_schema = op_config.get("input", {}).get("schema", {})
        if input_schema:
            sample_input = [
                {key: item[key] for key in input_schema} for item in sample_input
            ]

        # Create a sample of values for other keys
        sample_values = {
            key: list(set(str(item.get(key))[:50] for item in sample_input))[:5]
            for key in other_keys
        }

        prompt = f"""Analyze the following reduce operation and determine if it should be decomposed into two reduce operations chained together:

        Reduce Operation Prompt:
        ```
        {op_config['prompt']}
        ```

        Current Reduce Key(s): {reduce_keys}
        Other Available Keys: {', '.join(other_keys)}

        Sample values for other keys:
        {json.dumps(sample_values, indent=2)}

        Based on this information, determine if it would be beneficial to decompose this reduce operation into a sub-reduce operation followed by a final reduce operation. Consider the following:

        1. Is there a natural hierarchy in the data (e.g., country -> state -> city) among the other available keys, with a key at a finer level of granularity than the current reduce key(s)?
        2. Are the current reduce key(s) some form of ID, and are there many different types of inputs for that ID among the other available keys?
        3. Does the prompt implicitly ask for sub-grouping based on the other available keys (e.g., "summarize policies by state, then by country")?
        4. Would splitting the operation improve accuracy (i.e., make sure information isn't lost when reducing)?
        5. Are all the keys of the potential hierarchy provided in the other available keys? If not, we should not decompose.
        6. Importantly, do not suggest decomposition using any key that is already part of the current reduce key(s). We are looking for a new key from the other available keys to use for sub-grouping.

        Provide your analysis in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "should_decompose": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["should_decompose", "explanation"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    def _get_decomposition_details(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Determine the sub-group key and prompts for decomposed reduce operations.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.

        Returns:
            Dict[str, Any]: A dictionary containing the sub-group key and prompts for decomposed operations.
        """
        system_prompt = (
            "You are an AI assistant tasked with optimizing data processing pipelines."
        )

        # Sample a subset of input data for analysis
        sample_size = min(10, len(input_data))
        sample_input = random.sample(input_data, sample_size)

        # Get all keys from the input data
        all_keys = set().union(*(item.keys() for item in sample_input))
        reduce_key = op_config["reduce_key"]
        reduce_keys = [reduce_key] if isinstance(reduce_key, str) else reduce_key
        other_keys = [key for key in all_keys if key not in reduce_keys]

        prompt = f"""Given that we've decided to decompose the following reduce operation, suggest a two-step reduce process:

        Reduce Operation Prompt:
        ```
        {op_config['prompt']}
        ```

        Reduce Key(s): {reduce_key}
        Other Keys: {', '.join(other_keys)}

        Provide the following:
        1. A sub-group key to use for the first reduce operation
        2. A prompt for the first reduce operation
        3. A prompt for the second (final) reduce operation

        For the reduce operation prompts, you should only minimally modify the original prompt. The prompts should be Jinja templates, and the only variables they can access are the `reduce_key` and `inputs` variables.

        Provide your suggestions in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "sub_group_key": {"type": "string"},
                "first_reduce_prompt": {"type": "string"},
                "second_reduce_prompt": {"type": "string"},
            },
            "required": [
                "sub_group_key",
                "first_reduce_prompt",
                "second_reduce_prompt",
            ],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    def _determine_value_sampling(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine whether value sampling should be enabled and configure its parameters.
        """
        system_prompt = (
            "You are an AI assistant helping to optimize data processing pipelines."
        )

        # Sample a subset of input data for analysis
        sample_size = min(100, len(input_data))
        sample_input = random.sample(input_data, sample_size)

        prompt = f"""
        Analyze the following reduce operation and determine if value sampling should be enabled:

        Reduce Operation Prompt:
        {op_config['prompt']}

        Sample Input Data (first 2 items):
        {json.dumps(sample_input[:2], indent=2)}

        Value sampling is appropriate for reduce operations that don't need to look at all the values for each key to produce a good result, such as generic summarization tasks.

        Based on the reduce operation prompt and the sample input data, determine if value sampling should be enabled.
        Answer with 'yes' if value sampling should be enabled or 'no' if it should not be enabled. Explain your reasoning briefly.
        """

        parameters = {
            "type": "object",
            "properties": {
                "enable_sampling": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["enable_sampling", "explanation"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        if not result["enable_sampling"]:
            return {"enabled": False}

        # Print the explanation for enabling value sampling
        self.console.log(f"Value sampling enabled: {result['explanation']}")

        # Determine sampling method
        prompt = f"""
        We are optimizing a reduce operation in a data processing pipeline. The reduce operation is defined by the following prompt:

        Reduce Operation Prompt:
        {op_config['prompt']}

        Sample Input Data (first 2 items):
        {json.dumps(sample_input[:2], indent=2)}

        We have determined that value sampling should be enabled for this reduce operation. Value sampling is a technique used to process only a subset of the input data for each reduce key, rather than processing all items. This can significantly reduce processing time and costs for very large datasets, especially when the reduce operation doesn't require looking at every single item to produce a good result (e.g., summarization tasks).

        Now we need to choose the most appropriate sampling method. The available methods are:

        1. "random": Randomly select a subset of values.
        Example: In a customer review analysis task, randomly selecting a subset of reviews to summarize the overall sentiment.

        2. "cluster": Use K-means clustering to select representative samples.
        Example: In a document categorization task, clustering documents based on their content and selecting representative documents from each cluster to determine the overall categories.

        3. "sem_sim": Use semantic similarity to select the most relevant samples to a query text.
        Example: In a news article summarization task, selecting articles that are semantically similar to a query like "Major economic events of {{reduce_key}}" to produce a focused summary.

        Based on the reduce operation prompt, the nature of the task, and the sample input data, which sampling method would be most appropriate?

        Provide your answer as either "random", "cluster", or "sem_sim", and explain your reasoning in detail. Consider the following in your explanation:
        - The nature of the reduce task (e.g., summarization, aggregation, analysis)
        - The structure and content of the input data
        - The potential benefits and drawbacks of each sampling method for this specific task
        """

        parameters = {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["random", "cluster", "sem_sim"]},
                "explanation": {"type": "string"},
            },
            "required": ["method", "explanation"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)
        method = result["method"]

        value_sampling_config = {
            "enabled": True,
            "method": method,
            "sample_size": 100,  # Default sample size
            "embedding_model": "text-embedding-3-small",
        }

        if method in ["cluster", "sem_sim"]:
            # Determine embedding keys
            prompt = f"""
            For the {method} sampling method, we need to determine which keys from the input data should be used for generating embeddings.

            Input data keys:
            {', '.join(sample_input[0].keys())}

            Sample Input Data:
            {json.dumps(sample_input[0], indent=2)[:1000]}...

            Based on the reduce operation prompt and the sample input data, which keys should be used for generating embeddings? Use keys that will create meaningful embeddings (i.e., not id-related keys).
            Provide your answer as a list of key names that is a subset of the input data keys. You should pick only the 1-3 keys that are necessary for generating meaningful embeddings, that have relatively short values.
            """

            parameters = {
                "type": "object",
                "properties": {
                    "embedding_keys": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                },
                "required": ["embedding_keys", "explanation"],
            }

            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            result = json.loads(response.choices[0].message.content)
            # TODO: validate that these exist
            embedding_keys = result["embedding_keys"]
            for key in result["embedding_keys"]:
                if key not in sample_input[0]:
                    embedding_keys.remove(key)

            if not embedding_keys:
                # Select the reduce key
                self.console.log(
                    "No embedding keys found, selecting reduce key for embedding key"
                )
                embedding_keys = (
                    op_config["reduce_key"]
                    if isinstance(op_config["reduce_key"], list)
                    else [op_config["reduce_key"]]
                )

            value_sampling_config["embedding_keys"] = embedding_keys

        if method == "sem_sim":
            # Determine query text
            prompt = f"""
            For the semantic similarity (sem_sim) sampling method, we need to determine the query text to compare against when selecting samples.

            Reduce Operation Prompt:
            {op_config['prompt']}

            The query text should be a Jinja template with access to the `reduce_key` variable.
            Based on the reduce operation prompt, what would be an appropriate query text for selecting relevant samples?
            """

            parameters = {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["query_text", "explanation"],
            }

            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            result = json.loads(response.choices[0].message.content)
            value_sampling_config["query_text"] = result["query_text"]

        return value_sampling_config

    def _is_associative(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if the reduce operation is associative.

        This method analyzes the reduce operation configuration and a sample of the input data
        to determine if the operation is associative (i.e., the order of combining elements
        doesn't affect the final result).

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.

        Returns:
            bool: True if the operation is determined to be associative, False otherwise.
        """
        system_prompt = (
            "You are an AI assistant helping to optimize data processing pipelines."
        )

        # Sample a subset of input data for analysis
        sample_size = min(5, len(input_data))
        sample_input = random.sample(input_data, sample_size)

        prompt = f"""
        Analyze the following reduce operation and determine if it is associative:

        Reduce Operation Prompt:
        {op_config['prompt']}

        Sample Input Data:
        {json.dumps(sample_input, indent=2)[:1000]}...

        Based on the reduce operation prompt, determine whether the order in which we process data matters.
        Answer with 'yes' if order matters or 'no' if order doesn't matter.
        Explain your reasoning briefly.

        For example:
        - Merging extracted key-value pairs from documents does not require order: combining {{"name": "John", "age": 30}} with {{"city": "New York", "job": "Engineer"}} yields the same result regardless of order
        - Generating a timeline of events requires order: the order of events matters for maintaining chronological accuracy.

        Consider these examples when determining whether the order in which we process data matters. You might also have to consider the specific data.
        """

        parameters = {
            "type": "object",
            "properties": {
                "order_matters": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["order_matters", "explanation"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)
        result["is_associative"] = not result["order_matters"]

        self.console.log(
            f"[yellow]Reduce operation {'is associative' if result['is_associative'] else 'is not associative'}.[/yellow] Analysis: {result['explanation']}"
        )
        return result["is_associative"]

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        original_output: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a custom validator prompt for assessing the quality of the reduce operation output.

        This method creates a prompt that will be used to validate the output of the reduce operation.
        It includes specific questions about the quality and completeness of the output.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            original_output (List[Dict[str, Any]]): Original output of the reduce operation.

        Returns:
            str: A custom validator prompt as a string.
        """
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for reduce operations in data processing pipelines."

        sample_input = random.choice(input_data)
        input_keys = op_config.get("input", {}).get("schema", {})
        if input_keys:
            sample_input = {k: sample_input[k] for k in input_keys}

        reduce_key = op_config.get("reduce_key")
        if reduce_key and original_output:
            if isinstance(reduce_key, list):
                key = next(
                    (
                        tuple(item[k] for k in reduce_key)
                        for item in original_output
                        if all(k in item for k in reduce_key)
                    ),
                    tuple(None for _ in reduce_key),
                )
                sample_output = next(
                    (
                        item
                        for item in original_output
                        if all(item.get(k) == v for k, v in zip(reduce_key, key))
                    ),
                    {},
                )
            else:
                key = next(
                    (
                        item[reduce_key]
                        for item in original_output
                        if reduce_key in item
                    ),
                    None,
                )
                sample_output = next(
                    (item for item in original_output if item.get(reduce_key) == key),
                    {},
                )
        else:
            sample_output = original_output[0] if original_output else {}

        output_keys = op_config.get("output", {}).get("schema", {})
        sample_output = {k: sample_output[k] for k in output_keys}

        prompt = f"""
        Analyze the following reduce operation and its input/output:

        Reduce Operation Prompt:
        {op_config["prompt"]}

        Sample Input (just one item):
        {json.dumps(sample_input, indent=2)}

        Sample Output:
        {json.dumps(sample_output, indent=2)}

        Create a custom validator prompt that will assess how well the reduce operation performed its intended task. The prompt should ask specific 2-3 questions about the quality of the output, such as:
        1. Does the output accurately reflect the aggregation method specified in the task? For example, if finding anomalies, are the identified anomalies actually anomalies?
        2. Are there any missing fields, unexpected null values, or data type mismatches in the output compared to the expected schema?
        3. Does the output maintain the key information from the input while appropriately condensing or summarizing it? For instance, in a text summarization task, are the main points preserved?
        4. How well does the output adhere to any specific formatting requirements mentioned in the original prompt, such as character limits for summaries or specific data types for aggregated values?

        Note that the output may reflect more than just the input provided, since we only provide a one-item sample input. Provide your response as a single string containing the custom validator prompt. The prompt should be tailored to the task and avoid generic criteria. The prompt should not reference a specific value in the sample input, but rather a general property.
        """

        parameters = {
            "type": "object",
            "properties": {"validator_prompt": {"type": "string"}},
            "required": ["validator_prompt"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["validator_prompt"]

    def _validate_reduce_output(
        self,
        op_config: Dict[str, Any],
        validation_inputs: Dict[Any, List[Dict[str, Any]]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        """
        Validate the output of the reduce operation using the generated validator prompt.

        This method assesses the quality of the reduce operation output by applying the validator prompt
        to multiple samples of the input and output data.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            validation_inputs (Dict[Any, List[Dict[str, Any]]]): Validation inputs for the reduce operation.
            output_data (List[Dict[str, Any]]): Output data from the reduce operation.
            validator_prompt (str): The validator prompt generated earlier.

        Returns:
            Dict[str, Any]: A dictionary containing validation results and a flag indicating if improvement is needed.
        """
        system_prompt = "You are an AI assistant tasked with validating the output of reduce operations in data processing pipelines."

        validation_results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for reduce_key, inputs in validation_inputs.items():
                if isinstance(op_config["reduce_key"], list):
                    sample_output = next(
                        (
                            item
                            for item in output_data
                            if all(
                                item[key] == reduce_key[i]
                                for i, key in enumerate(op_config["reduce_key"])
                            )
                        ),
                        None,
                    )
                else:
                    sample_output = next(
                        (
                            item
                            for item in output_data
                            if item[op_config["reduce_key"]] == reduce_key
                        ),
                        None,
                    )

                if sample_output is None:
                    self.console.log(
                        f"Warning: No output found for reduce key {reduce_key}"
                    )
                    continue

                input_str = json.dumps(inputs, indent=2)
                # truncate input_str to 40,000 words
                input_str = input_str.split()[:40000]
                input_str = " ".join(input_str) + "..."

                prompt = f"""{validator_prompt}

                Reduce Operation Task:
                {op_config["prompt"]}

                Input Data Samples:
                {input_str}

                Output Data Sample:
                {json.dumps(sample_output, indent=2)}

                Based on the validator prompt and the input/output samples, assess the quality (e.g., correctness, completeness) of the reduce operation output.
                Provide your assessment in the following format:
                """

                parameters = {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "suggestions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["is_valid", "issues", "suggestions"],
                }

                futures.append(
                    executor.submit(
                        self.llm_client.generate,
                        [{"role": "user", "content": prompt}],
                        system_prompt,
                        parameters,
                    )
                )

            for future, (reduce_key, inputs) in zip(futures, validation_inputs.items()):
                response = future.result()
                result = json.loads(response.choices[0].message.content)
                validation_results.append(result)

        # Determine if optimization is needed based on validation results
        invalid_count = sum(
            1 for result in validation_results if not result["is_valid"]
        )
        needs_improvement = invalid_count > 1

        return {
            "needs_improvement": needs_improvement,
            "validation_results": validation_results,
        }

    def _create_validation_inputs(
        self, input_data: List[Dict[str, Any]], reduce_key: Union[str, List[str]]
    ) -> Dict[Any, List[Dict[str, Any]]]:
        # Group input data by reduce_key
        grouped_data = {}
        for item in input_data:
            if isinstance(reduce_key, list):
                key = tuple(item[k] for k in reduce_key)
            else:
                key = item[reduce_key]
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(item)

        # Select a fixed number of reduce keys
        selected_keys = random.sample(
            list(grouped_data.keys()),
            min(self.num_samples_in_validation, len(grouped_data)),
        )

        # Create a new dict with only the selected keys
        validation_inputs = {key: grouped_data[key] for key in selected_keys}

        return validation_inputs

    def _create_reduce_plans(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        is_associative: bool,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple reduce plans based on the input data and operation configuration.

        This method generates various reduce plans by varying batch sizes and fold prompts.
        It takes into account the LLM's context window size to determine appropriate batch sizes.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            is_associative (bool): Flag indicating whether the reduce operation is associative.

        Returns:
            List[Dict[str, Any]]: A list of reduce plans, each with different batch sizes and fold prompts.
        """
        model = op_config.get("model", "gpt-4o-mini")
        model_input_context_length = model_cost.get(model, {}).get(
            "max_input_tokens", 8192
        )

        # Estimate tokens for prompt, input, and output
        prompt_tokens = count_tokens(op_config["prompt"], model)
        sample_input = input_data[:100]
        sample_output = self._run_operation(op_config, input_data[:100])

        prompt_vars = extract_jinja_variables(op_config["prompt"])
        prompt_vars = [var.split(".")[-1] for var in prompt_vars]
        avg_input_tokens = mean(
            [
                count_tokens(
                    json.dumps({k: item[k] for k in prompt_vars if k in item}), model
                )
                for item in sample_input
            ]
        )
        avg_output_tokens = mean(
            [
                count_tokens(
                    json.dumps({k: item[k] for k in prompt_vars if k in item}), model
                )
                for item in sample_output
            ]
        )

        # Calculate max batch size that fits in context window
        max_batch_size = (
            model_input_context_length - prompt_tokens - avg_output_tokens
        ) // avg_input_tokens

        # Generate 6 candidate batch sizes
        batch_sizes = [
            max(1, int(max_batch_size * ratio))
            for ratio in [0.1, 0.2, 0.4, 0.6, 0.75, 0.9]
        ]
        # Log the generated batch sizes
        self.console.log("[cyan]Generating plans for batch sizes:[/cyan]")
        for size in batch_sizes:
            self.console.log(f"  - {size}")
        batch_sizes = sorted(set(batch_sizes))  # Remove duplicates and sort

        plans = []

        # Generate multiple fold prompts
        max_retries = 5
        retry_count = 0
        fold_prompts = []

        while retry_count < max_retries and not fold_prompts:
            try:
                fold_prompts = self._synthesize_fold_prompts(
                    op_config,
                    sample_input,
                    sample_output,
                    num_prompts=self.num_fold_prompts,
                )
                if not fold_prompts:
                    raise ValueError("No fold prompts generated")
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise RuntimeError(
                        f"Failed to generate fold prompts after {max_retries} attempts: {str(e)}"
                    )
                self.console.log(
                    f"Retry {retry_count}/{max_retries}: Failed to generate fold prompts. Retrying..."
                )

        for batch_size in batch_sizes:
            for fold_idx, fold_prompt in enumerate(fold_prompts):
                plan = op_config.copy()
                plan["fold_prompt"] = fold_prompt
                plan["fold_batch_size"] = batch_size
                plan["associative"] = is_associative
                plan["name"] = f"{op_config['name']}_bs_{batch_size}_fp_{fold_idx}"
                plans.append(plan)

        return plans

    def _calculate_compression_ratio(
        self,
        op_config: Dict[str, Any],
        sample_input: List[Dict[str, Any]],
        sample_output: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate the compression ratio of the reduce operation.

        This method compares the size of the input data to the size of the output data
        to determine how much the data is being compressed by the reduce operation.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            sample_input (List[Dict[str, Any]]): Sample input data.
            sample_output (List[Dict[str, Any]]): Sample output data.

        Returns:
            float: The calculated compression ratio.
        """
        reduce_key = op_config["reduce_key"]
        input_schema = op_config.get("input", {}).get("schema", {})
        output_schema = op_config["output"]["schema"]
        model = op_config.get("model", "gpt-4o-mini")

        compression_ratios = {}

        # Handle both single key and list of keys
        if isinstance(reduce_key, list):
            distinct_keys = set(
                tuple(item[k] for k in reduce_key) for item in sample_input
            )
        else:
            distinct_keys = set(item[reduce_key] for item in sample_input)

        for key in distinct_keys:
            if isinstance(reduce_key, list):
                key_input = [
                    item
                    for item in sample_input
                    if tuple(item[k] for k in reduce_key) == key
                ]
                key_output = [
                    item
                    for item in sample_output
                    if tuple(item[k] for k in reduce_key) == key
                ]
            else:
                key_input = [item for item in sample_input if item[reduce_key] == key]
                key_output = [item for item in sample_output if item[reduce_key] == key]

            if input_schema:
                key_input_tokens = sum(
                    count_tokens(
                        json.dumps({k: item[k] for k in input_schema if k in item}),
                        model,
                    )
                    for item in key_input
                )
            else:
                key_input_tokens = sum(
                    count_tokens(json.dumps(item), model) for item in key_input
                )

            key_output_tokens = sum(
                count_tokens(
                    json.dumps({k: item[k] for k in output_schema if k in item}), model
                )
                for item in key_output
            )

            compression_ratios[key] = (
                key_output_tokens / key_input_tokens if key_input_tokens > 0 else 1
            )

        if not compression_ratios:
            return 1

        # Calculate importance weights based on the number of items for each key
        total_items = len(sample_input)
        if isinstance(reduce_key, list):
            importance_weights = {
                key: len(
                    [
                        item
                        for item in sample_input
                        if tuple(item[k] for k in reduce_key) == key
                    ]
                )
                / total_items
                for key in compression_ratios
            }
        else:
            importance_weights = {
                key: len([item for item in sample_input if item[reduce_key] == key])
                / total_items
                for key in compression_ratios
            }

        # Calculate weighted average of compression ratios
        weighted_sum = sum(
            compression_ratios[key] * importance_weights[key]
            for key in compression_ratios
        )
        return weighted_sum

    def _synthesize_fold_prompts(
        self,
        op_config: Dict[str, Any],
        sample_input: List[Dict[str, Any]],
        sample_output: List[Dict[str, Any]],
        num_prompts: int = 2,
    ) -> List[str]:
        """
        Synthesize fold prompts for the reduce operation. We generate multiple
        fold prompts in case one is bad.

        A fold operation is a higher-order function that iterates through a data structure,
        accumulating the results of applying a given combining operation to its elements.
        In the context of reduce operations, folding allows processing of data in batches,
        which can significantly improve performance for large datasets.

        This method generates multiple fold prompts that can be used to optimize the reduce operation
        by allowing it to run on batches of inputs. It uses the language model to create prompts
        that are variations of the original reduce prompt, adapted for folding operations.

        Args:
            op_config (Dict[str, Any]): The configuration of the reduce operation.
            sample_input (List[Dict[str, Any]]): A sample of the input data.
            sample_output (List[Dict[str, Any]]): A sample of the output data.
            num_prompts (int, optional): The number of fold prompts to generate. Defaults to 2.

        Returns:
            List[str]: A list of synthesized fold prompts.

        The method performs the following steps:
        1. Sets up the system prompt and parameters for the language model.
        2. Defines a function to get random examples from the sample data.
        3. Creates a prompt template for generating fold prompts.
        4. Uses multi-threading to generate multiple fold prompts in parallel.
        5. Returns the list of generated fold prompts.
        """
        system_prompt = "You are an AI assistant tasked with creating a fold prompt for reduce operations in data processing pipelines."
        original_prompt = op_config["prompt"]

        input_schema = op_config.get("input", {}).get("schema", {})
        output_schema = op_config["output"]["schema"]
        reduce_key = op_config["reduce_key"]

        def get_random_examples():
            if isinstance(reduce_key, list):
                random_key = tuple(
                    random.choice(
                        [
                            tuple(item[k] for k in reduce_key if k in item)
                            for item in sample_input
                            if all(k in item for k in reduce_key)
                        ]
                    )
                )
                input_example = random.choice(
                    [
                        item
                        for item in sample_input
                        if all(item.get(k) == v for k, v in zip(reduce_key, random_key))
                    ]
                )
                output_example = random.choice(
                    [
                        item
                        for item in sample_output
                        if all(item.get(k) == v for k, v in zip(reduce_key, random_key))
                    ]
                )
            else:
                random_key = random.choice(
                    [item[reduce_key] for item in sample_input if reduce_key in item]
                )
                input_example = random.choice(
                    [item for item in sample_input if item[reduce_key] == random_key]
                )
                output_example = random.choice(
                    [item for item in sample_output if item[reduce_key] == random_key]
                )

            if input_schema:
                input_example = {
                    k: input_example[k] for k in input_schema if k in input_example
                }
            output_example = {
                k: output_example[k] for k in output_schema if k in output_example
            }
            return input_example, output_example

        parameters = {
            "type": "object",
            "properties": {
                "fold_prompt": {
                    "type": "string",
                }
            },
            "required": ["fold_prompt"],
        }

        def generate_single_prompt():
            input_example, output_example = get_random_examples()
            prompt = f"""
            Original Reduce Operation Prompt:
            {original_prompt}

            Sample Input:
            {json.dumps(input_example, indent=2)}

            Sample Output:
            {json.dumps(output_example, indent=2)}

            Create a fold prompt for the reduce operation to run on batches of inputs. The fold prompt should:
            1. Minimally modify the original reduce prompt
            2. Describe how to combine the new values with the current reduced value
            3. Be designed to work iteratively, allowing for multiple fold operations. The first iteration will use the original prompt, and all successive iterations will use the fold prompt.

            The fold prompt should be a Jinja2 template with the following variables available:
            - {{ output }}: The current reduced value (a dictionary with the current output schema)
            - {{ inputs }}: A list of new values to be folded in
            - {{ reduce_key }}: The key used for grouping in the reduce operation

            Provide the fold prompt as a string.
            """
            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            fold_prompt = json.loads(response.choices[0].message.content)["fold_prompt"]

            # Run the operation with the fold prompt
            # Create a temporary plan with the fold prompt
            temp_plan = op_config.copy()
            temp_plan["fold_prompt"] = fold_prompt
            temp_plan["fold_batch_size"] = min(
                len(sample_input), 2
            )  # Use a small batch size for testing

            # Run the operation with the fold prompt
            self._run_operation(temp_plan, sample_input[: temp_plan["fold_batch_size"]])

            # If the operation runs successfully, return the fold prompt
            return fold_prompt

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            fold_prompts = list(
                executor.map(lambda _: generate_single_prompt(), range(num_prompts))
            )

        return fold_prompts

    def _evaluate_reduce_plans(
        self,
        op_config: Dict[str, Any],
        plans: List[Dict[str, Any]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple reduce plans and select the best one.

        This method takes a list of reduce plans, evaluates each one using the input data
        and a validator prompt, and selects the best plan based on the evaluation scores.
        It also attempts to create and evaluate a merged plan that enhances the runtime performance
        of the best plan.

        A merged plan is an optimization technique applied to the best-performing plan
        that uses the fold operation. It allows the best plan to run even faster by
        executing parallel folds and then merging the results of these individual folds
        together. We default to a merge batch size of 2, but one can increase this.

        Args:
            op_config (Dict[str, Any]): The configuration of the reduce operation.
            plans (List[Dict[str, Any]]): A list of reduce plans to evaluate.
            input_data (List[Dict[str, Any]]): The input data to use for evaluation.
            validator_prompt (str): The prompt to use for validating the output of each plan.

        Returns:
            Dict[str, Any]: The best reduce plan, either the top-performing original plan
                            or a merged plan if it performs well enough.

        The method performs the following steps:
        1. Evaluates each plan using multi-threading.
        2. Sorts the plans based on their evaluation scores.
        3. Selects the best plan and attempts to create a merged plan.
        4. Evaluates the merged plan and compares it to the best original plan.
        5. Returns either the merged plan or the best original plan based on their scores.
        """
        self.console.log("\n[bold]Evaluating Reduce Plans:[/bold]")
        for i, plan in enumerate(plans):
            self.console.log(f"Plan {i+1} (batch size: {plan['fold_batch_size']})")

        plan_scores = []
        plan_outputs = {}

        # Create a fixed random sample for evaluation
        sample_size = min(100, len(input_data))
        evaluation_sample = random.sample(input_data, sample_size)

        # Create a fixed set of validation samples
        validation_inputs = self._create_validation_inputs(
            evaluation_sample, plan["reduce_key"]
        )

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    self._evaluate_single_plan,
                    plan,
                    evaluation_sample,
                    validator_prompt,
                    validation_inputs,
                )
                for plan in plans
            ]
            for future in as_completed(futures):
                plan, score, output = future.result()
                plan_scores.append((plan, score))
                plan_outputs[id(plan)] = output

        # Sort plans by score in descending order, then by fold_batch_size in descending order
        sorted_plans = sorted(
            plan_scores, key=lambda x: (x[1], x[0]["fold_batch_size"]), reverse=True
        )

        self.console.log("\n[bold]Reduce Plan Scores:[/bold]")
        for i, (plan, score) in enumerate(sorted_plans):
            self.console.log(
                f"Plan {i+1} (batch size: {plan['fold_batch_size']}): {score:.2f}"
            )

        best_plan, best_score = sorted_plans[0]
        self.console.log(
            f"\n[green]Selected best plan with score: {best_score:.2f} and batch size: {best_plan['fold_batch_size']}[/green]"
        )

        if op_config.get("synthesize_merge", False):
            # Create a new plan with merge prompt and updated parameters
            merged_plan = best_plan.copy()

            # Synthesize merge prompt if it doesn't exist
            if "merge_prompt" not in merged_plan:
                merged_plan["merge_prompt"] = self._synthesize_merge_prompt(
                    merged_plan, plan_outputs[id(best_plan)]
                )
                # Print the synthesized merge prompt
                self.console.log("\n[bold]Synthesized Merge Prompt:[/bold]")
                self.console.log(merged_plan["merge_prompt"])

            # Set merge_batch_size to 2 and num_parallel_folds to 5
            merged_plan["merge_batch_size"] = 2

            # Evaluate the merged plan
            _, merged_plan_score, _, operation_instance = self._evaluate_single_plan(
                merged_plan,
                evaluation_sample,
                validator_prompt,
                validation_inputs,
                return_instance=True,
            )

            # Get the merge and fold times from the operation instance
            merge_times = operation_instance.merge_times
            fold_times = operation_instance.fold_times
            merge_avg_time = mean(merge_times) if merge_times else None
            fold_avg_time = mean(fold_times) if fold_times else None

            self.console.log("\n[bold]Scores:[/bold]")
            self.console.log(f"Original plan: {best_score:.2f}")
            self.console.log(f"Merged plan: {merged_plan_score:.2f}")

            # Compare scores and decide which plan to use
            if merged_plan_score >= best_score * 0.75:
                self.console.log(
                    f"\n[green]Using merged plan with score: {merged_plan_score:.2f}[/green]"
                )
                if merge_avg_time and fold_avg_time:
                    merged_plan["merge_time"] = merge_avg_time
                    merged_plan["fold_time"] = fold_avg_time
                return merged_plan
            else:
                self.console.log(
                    f"\n[yellow]Merged plan quality too low. Using original plan with score: {best_score:.2f}[/yellow]"
                )
                return best_plan
        else:
            return best_plan

    def _evaluate_single_plan(
        self,
        plan: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        validation_inputs: List[Dict[str, Any]],
        return_instance: bool = False,
    ) -> Union[
        Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
        Tuple[Dict[str, Any], float, List[Dict[str, Any]], BaseOperation],
    ]:
        """
        Evaluate a single reduce plan using the provided input data and validator prompt.

        This method runs the reduce operation with the given plan, validates the output,
        and calculates a score based on the validation results. The scoring works as follows:
        1. It counts the number of valid results from the validation.
        2. The score is calculated as the ratio of valid results to the total number of validation results.
        3. This produces a score between 0 and 1, where 1 indicates all results were valid, and 0 indicates none were valid.

        TODO: We should come up with a better scoring method here, maybe pairwise comparisons.

        Args:
            plan (Dict[str, Any]): The reduce plan to evaluate.
            input_data (List[Dict[str, Any]]): The input data to use for evaluation.
            validator_prompt (str): The prompt to use for validating the output.
            return_instance (bool, optional): Whether to return the operation instance. Defaults to False.

        Returns:
            Union[
                Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
                Tuple[Dict[str, Any], float, List[Dict[str, Any]], BaseOperation],
            ]: A tuple containing the plan, its score, the output data, and optionally the operation instance.

        The method performs the following steps:
        1. Runs the reduce operation with the given plan on the input data.
        2. Validates the output using the validator prompt.
        3. Calculates a score based on the validation results.
        4. Returns the plan, score, output data, and optionally the operation instance.
        """
        output = self._run_operation(plan, input_data, return_instance)
        if return_instance:
            output, operation_instance = output

        validation_result = self._validate_reduce_output(
            plan, validation_inputs, output, validator_prompt
        )

        # Calculate a score based on validation results
        valid_count = sum(
            1
            for result in validation_result["validation_results"]
            if result["is_valid"]
        )
        score = valid_count / len(validation_result["validation_results"])

        if return_instance:
            return plan, score, output, operation_instance
        else:
            return plan, score, output

    def _synthesize_merge_prompt(
        self, plan: Dict[str, Any], sample_outputs: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize a merge prompt for combining multiple folded outputs in a reduce operation.

        This method generates a merge prompt that can be used to combine the results of multiple
        parallel fold operations into a single output. It uses the language model to create a prompt
        that is consistent with the original reduce and fold prompts while addressing the specific
        requirements of merging multiple outputs.

        Args:
            plan (Dict[str, Any]): The reduce plan containing the original prompt and fold prompt.
            sample_outputs (List[Dict[str, Any]]): Sample outputs from the fold operation to use as examples.

        Returns:
            str: The synthesized merge prompt as a string.

        The method performs the following steps:
        1. Sets up the system prompt for the language model.
        2. Prepares a random sample output to use as an example.
        3. Creates a detailed prompt for the language model, including the original reduce prompt,
           fold prompt, sample output, and instructions for creating the merge prompt.
        4. Uses the language model to generate the merge prompt.
        5. Returns the generated merge prompt.
        """
        system_prompt = "You are an AI assistant tasked with creating a merge prompt for reduce operations in data processing pipelines. The pipeline has a reduce operation, and incrementally folds inputs into a single output. We want to optimize the pipeline for speed by running multiple folds on different inputs in parallel, and then merging the fold outputs into a single output."

        output_schema = plan["output"]["schema"]
        random_output = random.choice(sample_outputs)
        random_output = {
            k: random_output[k] for k in output_schema if k in random_output
        }

        prompt = f"""Reduce Operation Prompt (runs on the first batch of inputs):
        {plan["prompt"]}

        Fold Prompt (runs on the second and subsequent batches of inputs):
        {plan["fold_prompt"]}

        Sample output of the fold operation (an input to the merge operation):
        {json.dumps(random_output, indent=2)}

        Create a merge prompt for the reduce operation to combine 2+ folded outputs. The merge prompt should:
        1. Give context on the task & fold operations, describing that the prompt will be used to combine multiple outputs from the fold operation (as if the original prompt was run on all inputs at once)
        2. Describe how to combine multiple folded outputs into a single output
        3. Minimally deviate from the reduce and fold prompts

        The merge prompt should be a Jinja2 template with the following variables available:
        - {{ outputs }}: A list of reduced outputs to be merged (each following the output schema). You can access the first output with {{ outputs[0] }} and the second with {{ outputs[1] }}

        Output Schema:
        {json.dumps(output_schema, indent=2)}

        Provide the merge prompt as a string.
        """

        parameters = {
            "type": "object",
            "properties": {
                "merge_prompt": {
                    "type": "string",
                }
            },
            "required": ["merge_prompt"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["merge_prompt"]
