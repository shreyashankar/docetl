import uuid
from typing import Dict, List, Tuple

import tiktoken

from docetl.operations.base import BaseOperation


class SplitOperation(BaseOperation):
    """
    A class that implements a split operation on input data, dividing it into manageable chunks.

    This class extends BaseOperation to:
    1. Split input data into chunks of specified size based on the 'split_key' and 'token_count' configuration.
    2. Assign unique identifiers to each original document and number chunks sequentially.
    3. Return results containing:
       - {split_key}_chunk: The content of the split chunk.
       - {name}_id: A unique identifier for each original document.
       - {name}_chunk_num: The sequential number of the chunk within its original document.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.config["name"]

    def syntax_check(self) -> None:
        required_keys = ["split_key", "method", "method_kwargs"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in SplitOperation configuration"
                )

        if not isinstance(self.config["split_key"], str):
            raise TypeError("'split_key' must be a string")

        if self.config["method"] not in ["token_count", "delimiter"]:
            raise ValueError(f"Invalid method '{self.config['method']}'")

        if self.config["method"] == "token_count":
            if (
                not isinstance(self.config["method_kwargs"]["num_tokens"], int)
                or self.config["method_kwargs"]["num_tokens"] <= 0
            ):
                raise ValueError("'num_tokens' must be a positive integer")
        elif self.config["method"] == "delimiter":
            if not isinstance(self.config["method_kwargs"]["delimiter"], str):
                raise ValueError("'delimiter' must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        split_key = self.config["split_key"]
        method = self.config["method"]
        method_kwargs = self.config["method_kwargs"]
        try:
            encoder = tiktoken.encoding_for_model(
                self.config["method_kwargs"]
                .get("model", self.default_model)
                .split("/")[-1]
            )
        except Exception:
            encoder = tiktoken.encoding_for_model("gpt-4o")

        results = []
        cost = 0.0

        for item in input_data:
            if split_key not in item:
                raise KeyError(f"Split key '{split_key}' not found in item")

            content = item[split_key]
            doc_id = str(uuid.uuid4())

            if method == "token_count":
                token_count = method_kwargs["num_tokens"]
                tokens = encoder.encode(content)

                for chunk_num, i in enumerate(
                    range(0, len(tokens), token_count), start=1
                ):
                    chunk_tokens = tokens[i : i + token_count]
                    chunk = encoder.decode(chunk_tokens)

                    result = item.copy()
                    result.update(
                        {
                            f"{split_key}_chunk": chunk,
                            f"{self.name}_id": doc_id,
                            f"{self.name}_chunk_num": chunk_num,
                        }
                    )
                    results.append(result)

            elif method == "delimiter":
                delimiter = method_kwargs["delimiter"]
                num_splits_to_group = method_kwargs.get("num_splits_to_group", 1)
                chunks = content.split(delimiter)

                # Get rid of empty chunks
                chunks = [chunk for chunk in chunks if chunk.strip()]

                for chunk_num, i in enumerate(
                    range(0, len(chunks), num_splits_to_group), start=1
                ):
                    grouped_chunks = chunks[i : i + num_splits_to_group]
                    joined_chunk = delimiter.join(grouped_chunks).strip()

                    result = item.copy()
                    result.update(
                        {
                            f"{split_key}_chunk": joined_chunk,
                            f"{self.name}_id": doc_id,
                            f"{self.name}_chunk_num": chunk_num,
                        }
                    )
                    results.append(result)

        return results, cost
