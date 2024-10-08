import json
import re
from typing import Any, Dict, List

import tiktoken
import yaml
from jinja2 import Environment, meta
from litellm import completion_cost as lcc


def extract_jinja_variables(template_string: str) -> List[str]:
    """
    Extract variables from a Jinja2 template string.

    This function uses both Jinja2's AST parsing and regex to find all variables
    referenced in the given template string, including nested variables.

    Args:
        template_string (str): The Jinja2 template string to analyze.

    Returns:
        List[str]: A list of unique variable names found in the template.
    """
    # Create a Jinja2 environment
    env = Environment(autoescape=True)

    # Parse the template
    ast = env.parse(template_string)

    # Find all the variables referenced in the template
    variables = meta.find_undeclared_variables(ast)

    # Use regex to find any additional variables that might be missed
    # This regex looks for {{ variable }} patterns, including nested ones
    regex_variables = set(re.findall(r"{{\s*([\w.]+)\s*}}", template_string))

    # Combine both sets of variables
    all_variables = variables.union(regex_variables)

    # Special-case: remove "input"
    all_variables.discard("input")

    return list(all_variables)


def completion_cost(response) -> float:
    try:
        return lcc(response)
    except Exception:
        return 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in a string using the specified model.
    """
    model_name = model.replace("azure/", "")
    try:
        encoder = tiktoken.encoding_for_model(model_name)
        return len(encoder.encode(text))
    except Exception:
        # Use gpt-4o-mini to count tokens for other models
        encoder = tiktoken.encoding_for_model("gpt-4o")
        return len(encoder.encode(text))


def truncate_sample_data(
    data: Dict[str, Any], available_tokens: int, key_lists: List[List[str]], model: str
) -> Dict[str, Any]:
    """
    Truncate sample data to fit within available tokens.

    Args:
        data (Dict[str, Any]): The original data dictionary to truncate.
        available_tokens (int): The maximum number of tokens allowed.
        key_lists (List[List[str]]): Lists of keys to prioritize in the truncation process.
        model (str): The name of the model to use for token counting.

    Returns:
        Dict[str, Any]: A new dictionary containing truncated data that fits within the token limit.
    """
    truncated_data = {}
    current_tokens = 0

    for key_list in key_lists:
        for key in sorted(
            key_list, key=lambda k: len(str(data.get(k, ""))), reverse=True
        ):
            if key in data:
                field_tokens = count_tokens(f'"{key}": {json.dumps(data[key])}', model)
                if current_tokens + field_tokens <= available_tokens:
                    truncated_data[key] = data[key]
                    current_tokens += field_tokens
                else:
                    # Calculate remaining tokens
                    remaining_tokens = available_tokens - current_tokens

                    # Encode the value
                    try:
                        encoder = tiktoken.encoding_for_model(model)
                    except Exception:
                        encoder = tiktoken.encoding_for_model("gpt-4o")
                    encoded_value = encoder.encode(str(data[key]))

                    # Calculate how many tokens to keep
                    tokens_to_keep = (
                        remaining_tokens - 20
                    )  # Reserve 20 tokens for truncation message
                    start_tokens = min(tokens_to_keep // 2, field_tokens // 2)
                    end_tokens = min(
                        tokens_to_keep - start_tokens, field_tokens - start_tokens
                    )

                    # Truncate the encoded value
                    truncated_encoded = (
                        encoded_value[:start_tokens]
                        + encoder.encode("[....truncated content...]")
                        + encoded_value[-end_tokens:]
                    )

                    # Decode the truncated value
                    truncated_value = encoder.decode(truncated_encoded)

                    # Add the truncated value to the result
                    truncated_data[key] = truncated_value
                    current_tokens += len(truncated_encoded)

                    return truncated_data

    return truncated_data
