import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from docetl.base_schemas import ParsingTool
from docetl.parsing_tools import get_parser, get_parsing_tools


def _read_file_as_text(path: str) -> str | None:
    """Extract text from one file for directory datasets.

    Dispatches on extension: PDFs via PyMuPDF, Word/PowerPoint/Excel via
    their parsers, everything else as UTF-8 text. Returns ``None`` (and
    logs a warning) for files that are binary with no known extractor.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            import pymupdf

            with pymupdf.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        if ext == ".docx":
            from docx import Document

            return "\n".join(p.text for p in Document(path).paragraphs)
        if ext == ".pptx":
            from pptx import Presentation

            slides = []
            for slide in Presentation(path).slides:
                slides.append(
                    "\n".join(
                        shape.text_frame.text
                        for shape in slide.shapes
                        if shape.has_text_frame
                    )
                )
            return "\n\n".join(slides)
        if ext == ".xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows = []
            for sheet in wb:
                for row in sheet.iter_rows(values_only=True):
                    rows.append("\t".join("" if c is None else str(c) for c in row))
            return "\n".join(rows)

        with open(path, "rb") as f:
            head = f.read(1024)
        if b"\x00" in head:
            from docetl.console import DOCETL_CONSOLE

            DOCETL_CONSOLE.log(
                f"[yellow]Skipping binary file with no text extractor: {path}[/yellow]"
            )
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        from docetl.console import DOCETL_CONSOLE

        DOCETL_CONSOLE.log(f"[yellow]Skipping unreadable file {path}: {e}[/yellow]")
        return None


def create_parsing_tool_map(
    parsing_tools: list[ParsingTool] | None,
) -> dict[str, ParsingTool]:
    """
    Create a mapping of parsing tool names to their corresponding ParsingTool objects.

    Args:
        parsing_tools (list[ParsingTool] | None): A list of ParsingTool objects.

    Returns:
        dict[str, ParsingTool]: A dictionary mapping tool names to ParsingTool objects.
    """
    if not parsing_tools:
        return {}

    if not isinstance(parsing_tools[0], ParsingTool):
        parsing_tools = [ParsingTool(**tool) for tool in parsing_tools]

    return {tool.name: tool for tool in parsing_tools}


class DataLoader:
    """
    A class representing a dataset with various loading and parsing capabilities.

    Attributes:
        type (str): The type of the dataset ('file' or 'memory').
        source (str): The source of the dataset (currently only 'local' is supported).
        path_or_data (str | list[dict]): The file path or in-memory data.
        parsing (list[dict[str, str]]): A list of parsing tools to apply to the data.
        user_defined_parsing_tool_map (dict[str, ParsingTool]): A map of user-defined parsing tools.
    """

    class schema(BaseModel):
        """
        Represents a dataset configuration in the pipeline.

        Attributes:
            type (str): The type of the dataset. Must be either 'file' or 'memory'.
            path (str): The path to the dataset file or the in-memory data, depending on the type.
            source (str): The source of the dataset. Currently, only 'local' is supported. Defaults to 'local'.
            parsing (list[dict[str, str]] | None): A list of parsing tools to apply to the data. Each parsing tool
                                                      is represented by a dictionary with 'input_key', 'function', and
                                                      'output_key' keys. Defaults to None.

        Example:
            ```yaml
            datasets:
              my_dataset:
                type: file
                path: input.json
                parsing:
                  - input_key: file_path
                    function: txt_to_string
                    output_key: content
            ```

        Note:
            The parsing tools are applied in the order they are listed. Each parsing tool takes the output
            of the previous tool as its input, allowing for chained processing of the data.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        type: Literal["file", "memory"]
        path: str | list[dict] | pd.DataFrame
        source: str = "local"
        parsing: list[dict[str, str]] | None = None

    def __init__(
        self,
        runner,
        type: str,
        path_or_data: str | list[dict],
        source: str = "local",
        parsing: list[dict[str, str]] = None,
        user_defined_parsing_tool_map: dict[str, ParsingTool] = {},
    ):
        """
        Initialize a Dataset object.

        Args:
            type (str): The type of the dataset ('file' or 'memory').
            source (str): The source of the dataset (currently only 'local' is supported).
            path_or_data (str | list[dict]): The file path or in-memory data.
            parsing (list[dict[str, str]] | None): A list of parsing tools to apply to the data.
            user_defined_parsing_tool_map (dict[str, ParsingTool], optional): A map of user-defined parsing tools.
        """
        self.runner = runner
        self.type = self._validate_type(type)
        self.source = self._validate_source(source)
        self.path_or_data = self._validate_path_or_data(path_or_data)
        self.parsing = self._validate_parsing(parsing)
        self.user_defined_parsing_tool_map = user_defined_parsing_tool_map

    def _validate_type(self, type: str) -> str:
        """
        Validate the dataset type.

        Args:
            type (str): The type to validate.

        Returns:
            str: The validated type.

        Raises:
            ValueError: If the type is not 'file' or 'memory'.
        """
        if type not in ["file", "memory"]:
            raise ValueError("Type must be 'file' or 'memory'")
        return type

    def _validate_source(self, source: str) -> str:
        """
        Validate the dataset source.

        Args:
            source (str): The source to validate.

        Returns:
            str: The validated source.

        Raises:
            ValueError: If the source is not 'local'.
        """
        if source != "local":
            raise ValueError("Source must be 'local'")
        return source

    def _validate_path_or_data(
        self, path_or_data: str | list[dict]
    ) -> str | list[dict]:
        """
        Validate the path or data of the dataset.

        Args:
            path_or_data (str | list[dict]): The path or data to validate.

        Returns:
            str | list[dict]: The validated path or data.

        Raises:
            ValueError: If the path or data is invalid for the given type.
        """
        if self.type == "file":
            if not isinstance(path_or_data, str):
                raise ValueError("For type 'file', path_or_data must be a string")
            valid_extensions = (".json", ".csv", ".parquet")
            if not path_or_data.lower().endswith(
                valid_extensions
            ) and not os.path.isdir(path_or_data):
                raise ValueError(
                    f"Path must end with one of {valid_extensions}, "
                    "or be a directory (every file in it is read as text)"
                )
        elif self.type == "memory":
            if not isinstance(path_or_data, (list, pd.DataFrame)):
                raise ValueError(
                    "For type 'memory', path_or_data must be a list of dictionaries, or a pandas DataFrame"
                )
        return path_or_data

    def _validate_parsing(
        self, parsing_tools: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        """
        Validate the parsing tools.

        Args:
            parsing_tools (list[dict[str, str]] | None): The parsing tools to validate.

        Returns:
            list[dict[str, str]]: The validated parsing tools.

        Raises:
            ValueError: If any parsing tool is invalid.
        """
        if parsing_tools is None:
            return []

        for tool in parsing_tools:
            if not isinstance(tool, dict) or "function" not in tool:
                raise ValueError(
                    "Each parsing tool must be a dictionary with a 'function' key and any arguments required by that function"
                )
            if not isinstance(tool["function"], str):
                raise ValueError("'function' in parsing tools must be a string")
            if "function_kwargs" in tool and not isinstance(
                tool["function_kwargs"], dict
            ):
                raise ValueError("'function_kwargs', if present, must be a dictionary")

        return parsing_tools

    def __repr__(self):
        """
        Return a string representation of the Dataset object.

        Returns:
            str: A string representation of the Dataset object.
        """
        return f"DataLoader(type='{self.type}', source='{self.source}', path_or_data='{self.path_or_data}', parsing={self.parsing})"

    def load(self, limit: int | None = None) -> list[dict]:
        """
        Load the dataset from the specified path or return the in-memory data.

        Args:
            limit (int | None): If given, load only the first *limit* raw
                rows (streaming for CSV/Parquet, so a small sample of a
                large file is cheap). Parsing tools are applied after the
                limit, i.e. to just the sampled rows.

        Returns:
            list[dict]: A list of dictionaries representing the dataset.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        if self.type == "memory":
            data = self.path_or_data
            if limit is not None:
                data = data[:limit]
            return self._apply_parsing_tools(data)

        if os.path.isdir(self.path_or_data):
            data = []
            for path in self._dir_files(limit):
                text = _read_file_as_text(path)
                if text is None:
                    continue
                data.append(
                    {
                        "path": str(path),
                        "filename": os.path.basename(path),
                        "text": text,
                    }
                )
            return self._apply_parsing_tools(data)

        _, ext = os.path.splitext(self.path_or_data.lower())

        if ext == ".json":
            import json

            with open(self.path_or_data, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if limit is not None:
                data = data[:limit]
        elif ext == ".csv":
            import csv
            import itertools

            with open(self.path_or_data, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                data = list(
                    itertools.islice(reader, limit) if limit is not None else reader
                )
        elif ext == ".parquet":
            data = self._read_parquet(limit)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return self._apply_parsing_tools(data)

    def _read_parquet(self, limit: int | None) -> list[dict]:
        if limit is not None:
            try:
                import pyarrow.parquet as pq

                batches = []
                rows = 0
                for batch in pq.ParquetFile(self.path_or_data).iter_batches():
                    batches.append(batch)
                    rows += batch.num_rows
                    if rows >= limit:
                        break
                if batches:
                    import pyarrow as pa

                    table = pa.Table.from_batches(batches)
                    return table.to_pylist()[:limit]
                return []
            except ImportError:
                pass
            return (
                pd.read_parquet(self.path_or_data).head(limit).to_dict(orient="records")
            )
        return pd.read_parquet(self.path_or_data).to_dict(orient="records")

    def _dir_files(self, limit: int | None = None) -> list[str]:
        """All non-hidden files under a directory dataset, sorted, recursive."""
        files = []
        for root, dirs, names in os.walk(self.path_or_data):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            files.extend(os.path.join(root, n) for n in names if not n.startswith("."))
        files.sort()
        return files[:limit] if limit is not None else files

    def count(self) -> int:
        """Number of rows the dataset loads to, computed cheaply when possible.

        With parsing tools configured the data must be fully loaded and
        parsed (parsing can change the row count). Otherwise Parquet uses
        file metadata and CSV streams line counts without building dicts.
        """
        if self.parsing:
            return len(self.load())

        if self.type == "memory":
            return len(self.path_or_data)

        if os.path.isdir(self.path_or_data):
            return len(self._dir_files())

        _, ext = os.path.splitext(self.path_or_data.lower())
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq

                return pq.ParquetFile(self.path_or_data).metadata.num_rows
            except ImportError:
                pass
        elif ext == ".csv":
            import csv

            with open(self.path_or_data, "r", encoding="utf-8-sig", newline="") as f:
                return sum(1 for _ in csv.DictReader(f))

        return len(self.load())

    def _process_item(
        self,
        item: dict[str, Any],
        func: Callable,
        **function_kwargs: dict[str, Any],
    ):
        result = func(item, **function_kwargs)
        return [item.copy() | res for res in result]

    def _apply_parsing_tools(self, data: list[dict] | pd.DataFrame) -> list[dict]:
        """
        Apply parsing tools to the data.

        Args:
            data (list[dict] | pd.DataFrame): The data to apply parsing tools to.

        Returns:
            list[dict]: The data with parsing tools applied.

        Raises:
            ValueError: If a parsing tool is not found or if an input key is missing from an item.
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        for tool in self.parsing:
            function_kwargs = dict(tool)
            function_kwargs.pop("function")
            # FIXME: The following is just for backwards compatibility
            # with the existing yaml format...
            if "function_kwargs" in function_kwargs:
                function_kwargs.update(function_kwargs.pop("function_kwargs"))

            try:
                func = get_parser(tool["function"])
            except KeyError:
                if (
                    self.user_defined_parsing_tool_map
                    and tool["function"] in self.user_defined_parsing_tool_map
                ):
                    # Define the custom function in an explicit namespace to reliably capture it
                    _namespace: dict[str, Any] = {}
                    exec(
                        "from typing import List, Dict\n"
                        + self.user_defined_parsing_tool_map[
                            tool["function"]
                        ].function_code,
                        _namespace,
                        _namespace,
                    )
                    # Get the function object from the exec namespace
                    func = _namespace[tool["function"]]
                else:
                    raise ValueError(
                        f"Parsing tool {tool['function']} not found. Please define it or use one of our existing parsing tools: {get_parsing_tools()}"
                    )

            new_data = []

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._process_item,
                        item,
                        func,
                        **function_kwargs,
                    )
                    for item in data
                ]
                for future in as_completed(futures):
                    new_data.extend(future.result())

            data = new_data

        return data
