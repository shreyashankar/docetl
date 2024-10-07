from collections import defaultdict
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import hashlib
from rich.console import Console

from docetl.dataset import Dataset, create_parsing_tool_map
from docetl.operations import get_operation
from docetl.operations.utils import flush_cache
from docetl.config_wrapper import ConfigWrapper

load_dotenv()


class DSLRunner(ConfigWrapper):
    """
    A class for executing Domain-Specific Language (DSL) configurations.

    This class is responsible for loading, validating, and executing DSL configurations
    defined in YAML files. It manages datasets, executes pipeline steps, and tracks
    the cost of operations.

    Attributes:
        config (Dict): The loaded configuration from the YAML file.
        default_model (str): The default language model to use for operations.
        max_threads (int): Maximum number of threads for parallel processing.
        console (Console): Rich console for output formatting.
        datasets (Dict): Storage for loaded datasets.
    """

    def __init__(self, config: Dict, max_threads: int = None):
        """
        Initialize the DSLRunner with a YAML configuration file.

        Args:
            max_threads (int, optional): Maximum number of threads to use. Defaults to None.
        """
        ConfigWrapper.__init__(self, config, max_threads)
        self.datasets = {}

        self.intermediate_dir = (
            self.config.get("pipeline", {}).get("output", {}).get("intermediate_dir")
        )

        # Create parsing tool map
        self.parsing_tool_map = create_parsing_tool_map(
            self.config.get("parsing_tools", None)
        )

        # Check if output path is correctly formatted as JSON
        output_path = self.config.get("pipeline", {}).get("output", {}).get("path")
        if output_path:
            if not output_path.lower().endswith(".json"):
                raise ValueError(
                    f"Output path '{output_path}' is not a JSON file. Please provide a path ending with '.json'."
                )
        else:
            raise ValueError(
                "No output path specified in the configuration. Please provide an output path ending with '.json' in the configuration."
            )

        self.syntax_check()

        op_map = {op["name"]: op for op in self.config["operations"]}

        # Hash each pipeline step/operation
        # for each step op, hash the code of each op up until and (including that op)
        self.step_op_hashes = defaultdict(dict)
        for step in self.config["pipeline"]["steps"]:
            for idx, op in enumerate(step["operations"]):
                op_name = op if isinstance(op, str) else list(op.keys())[0]

                all_ops_until_and_including_current = [
                    op_map[prev_op] for prev_op in step["operations"][:idx]
                ] + [op_map[op_name]]
                all_ops_str = json.dumps(all_ops_until_and_including_current)
                self.step_op_hashes[step["name"]][op_name] = hashlib.sha256(
                    all_ops_str.encode()
                ).hexdigest()

    def syntax_check(self):
        """
        Perform a syntax check on all operations defined in the configuration.

        This method validates each operation by attempting to instantiate it.
        If any operation fails to instantiate, a ValueError is raised.

        Raises:
            ValueError: If any operation fails the syntax check.
        """
        self.console.rule("[yellow]Syntax Check[/yellow]")
        self.console.print(
            "[yellow]Performing syntax check on all operations...[/yellow]"
        )

        for operation_config in self.config["operations"]:
            operation = operation_config["name"]
            operation_type = operation_config["type"]

            try:
                operation_class = get_operation(operation_type)
                operation_class(
                    self,
                    operation_config,
                    self.default_model,
                    self.max_threads,
                    self.console,
                )
            except Exception as e:
                raise ValueError(
                    f"Syntax check failed for operation '{operation}': {str(e)}"
                )

        self.console.print("[green]Syntax check passed for all operations.[/green]")

    def find_operation(self, op_name: str) -> Dict:
        for operation_config in self.config["operations"]:
            if operation_config["name"] == op_name:
                return operation_config
        raise ValueError(f"Operation '{op_name}' not found in configuration.")

    def run(self) -> float:
        """
        Execute the entire pipeline defined in the configuration.

        This method loads datasets, executes each step in the pipeline, saves the output,
        and returns the total cost of execution.

        Returns:
            float: The total cost of executing the pipeline.
        """
        self.console.rule("[bold blue]Pipeline Execution[/bold blue]")
        start_time = time.time()
        self.load_datasets()
        total_cost = 0
        for step in self.config["pipeline"]["steps"]:
            step_name = step["name"]
            input_data = (
                self.datasets[step["input"]].load() if "input" in step else None
            )
            output_data, step_cost = self.execute_step(step, input_data)
            self.datasets[step_name] = Dataset(self, "memory", output_data)
            flush_cache(self.console)
            total_cost += step_cost
            self.console.log(
                f"Step [cyan]{step_name}[/cyan] completed. Cost: [green]${step_cost:.2f}[/green]"
            )

        self.save_output(
            self.datasets[self.config["pipeline"]["steps"][-1]["name"]].load()
        )

        # Save the self.step_op_hashes to a file if self.intermediate_dir exists
        if self.intermediate_dir:
            with open(
                os.path.join(self.intermediate_dir, ".docetl_intermediate_config.json"),
                "w",
            ) as f:
                json.dump(self.step_op_hashes, f)

        self.console.rule("[bold green]Execution Summary[/bold green]")
        self.console.print(f"[bold green]Total cost: [green]${total_cost:.2f}[/green]")
        self.console.print(
            f"[bold green]Total time: [green]{time.time() - start_time:.2f} seconds[/green]"
        )

        return total_cost

    def load_datasets(self):
        """
        Load all datasets defined in the configuration.

        This method creates Dataset objects for each dataset in the configuration.

        Raises:
            ValueError: If an unsupported dataset type is encountered.
        """
        self.console.rule("[cyan]Loading Datasets[/cyan]")
        for name, dataset_config in self.config["datasets"].items():
            if dataset_config["type"] == "file":
                self.datasets[name] = Dataset(
                    self,
                    "file",
                    dataset_config["path"],
                    source="local",
                    parsing=dataset_config.get("parsing", []),
                    user_defined_parsing_tool_map=self.parsing_tool_map,
                )
                self.console.print(f"Loaded dataset: [bold]{name}[/bold]")
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

    def save_output(self, data: List[Dict]):
        """
        Save the final output of the pipeline.

        Args:
            data (List[Dict]): The data to be saved.

        Raises:
            ValueError: If an unsupported output type is specified in the configuration.
        """
        self.console.rule("[cyan]Saving Output[/cyan]")
        output_config = self.config["pipeline"]["output"]
        if output_config["type"] == "file":
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_config["path"]), exist_ok=True)

            with open(output_config["path"], "w") as file:
                json.dump(data, file, indent=2)
            self.console.print(
                f"[green italic]💾 Output saved to {output_config['path']}[/green italic]"
            )
        else:
            raise ValueError(f"Unsupported output type: {output_config['type']}")

    def execute_step(
        self, step: Dict, input_data: Optional[List[Dict]]
    ) -> Tuple[List[Dict], float]:
        """
        Execute a single step in the pipeline.

        This method runs all operations defined for a step, updating the progress
        and calculating the cost.

        Args:
            step (Dict): The step configuration.
            input_data (Optional[List[Dict]]): Input data for the step.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the output data and the total cost of the step.
        """
        self.console.rule(f"[bold blue]Executing Step: {step['name']}[/bold blue]")
        total_cost = 0
        for operation in step["operations"]:
            if isinstance(operation, dict):
                operation_name = list(operation.keys())[0]
                operation_config = self.find_operation(operation_name)
            else:
                operation_name = operation
                operation_config = {}

            # Load from checkpoint if it exists
            attempted_input_data = self._load_from_checkpoint_if_exists(
                step["name"], operation_name
            )
            if attempted_input_data is not None:
                input_data = attempted_input_data
                self.console.print(
                    f"[green]✓ [italic]Loaded saved data for operation '{operation_name}' in step '{step['name']}'[/italic][/green]"
                )
                continue

            op_object = self.find_operation(operation_name).copy()
            op_object.update(operation_config)

            # If sample is set, sample the input data
            if op_object.get("sample"):
                input_data = self.datasets[step["input"]].sample(op_object["sample"])

            with self.console.status("[bold]Running Operation:[/bold]") as status:
                status.update(f"Type: [cyan]{op_object['type']}[/cyan]")
                status.update(f"Name: [cyan]{op_object.get('name', 'Unnamed')}[/cyan]")
                self.status = status

                operation_class = get_operation(op_object["type"])
                operation_instance = operation_class(
                    self,
                    op_object,
                    self.default_model,
                    self.max_threads,
                    self.console,
                    self.status,
                )
                if op_object["type"] == "equijoin":
                    left_data = self.datasets[op_object["left"]].load()
                    right_data = self.datasets[op_object["right"]].load()
                    input_data, cost = operation_instance.execute(left_data, right_data)
                else:
                    input_data, cost = operation_instance.execute(input_data)
                total_cost += cost
                self.console.log(
                    f"\tOperation [cyan]{operation_name}[/cyan] completed. Cost: [green]${cost:.2f}[/green]"
                )

            # Checkpoint after each operation
            if self.intermediate_dir:
                self._save_checkpoint(step["name"], operation_name, input_data)

        return input_data, total_cost

    def _load_from_checkpoint_if_exists(
        self, step_name: str, operation_name: str
    ) -> Optional[List[Dict]]:
        if self.intermediate_dir is None:
            return None

        intermediate_config_path = os.path.join(
            self.intermediate_dir, ".docetl_intermediate_config.json"
        )
        if not os.path.exists(intermediate_config_path):
            return None

        # See if the checkpoint config is the same as the current step op hash
        with open(intermediate_config_path, "r") as f:
            intermediate_config = json.load(f)

        if (
            intermediate_config.get(step_name, {}).get(operation_name, "")
            != self.step_op_hashes[step_name][operation_name]
        ):
            return None

        checkpoint_path = os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.json"
        )
        # check if checkpoint exists
        if os.path.exists(checkpoint_path):
            if f"{step_name}_{operation_name}" not in self.datasets:
                self.datasets[f"{step_name}_{operation_name}"] = Dataset(
                    self, "file", checkpoint_path, "local"
                )
            return self.datasets[f"{step_name}_{operation_name}"].load()
        return None

    def _save_checkpoint(self, step_name: str, operation_name: str, data: List[Dict]):
        """
        Save a checkpoint of the current data after an operation.

        This method creates a JSON file containing the current state of the data
        after an operation has been executed. The checkpoint is saved in a directory
        structure that reflects the step and operation names.

        Args:
            step_name (str): The name of the current step in the pipeline.
            operation_name (str): The name of the operation that was just executed.
            data (List[Dict]): The current state of the data to be checkpointed.

        Note:
            The checkpoint is saved only if a checkpoint directory has been specified
            when initializing the DSLRunner.
        """
        checkpoint_path = os.path.join(
            self.intermediate_dir, step_name, f"{operation_name}.json"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

        self.console.print(
            f"[green]✓ [italic]Intermediate saved for operation '{operation_name}' in step '{step_name}' at {checkpoint_path}[/italic][/green]"
        )


if __name__ == "__main__":
    runner = DSLRunner("workloads/medical/map_opt.yaml")
    runner.run()
