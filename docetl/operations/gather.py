from typing import Any, Dict, List, Tuple

from docetl.operations.base import BaseOperation


class GatherOperation(BaseOperation):
    """
    A class that implements a gather operation on input data, adding contextual information from surrounding chunks.

    This class extends BaseOperation to:
    1. Group chunks by their document ID.
    2. Order chunks within each group.
    3. Add peripheral context to each chunk based on the configuration.
    4. Include headers for each chunk and its upward hierarchy.
    5. Return results containing the rendered chunks with added context, including information about skipped characters and headers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the GatherOperation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """
        Perform a syntax check on the operation configuration.

        Raises:
            ValueError: If required keys are missing or if there are configuration errors.
            TypeError: If main_chunk_start or main_chunk_end are not strings.
        """
        required_keys = ["content_key", "doc_id_key", "order_key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in GatherOperation configuration"
                )

        peripheral_config = self.config.get("peripheral_chunks", {})
        for direction in ["previous", "next"]:
            if direction not in peripheral_config:
                continue
            for section in ["head", "middle", "tail"]:
                if section in peripheral_config[direction]:
                    section_config = peripheral_config[direction][section]
                    if section != "middle" and "count" not in section_config:
                        raise ValueError(
                            f"Missing 'count' in {direction}.{section} configuration"
                        )

        if "main_chunk_start" in self.config and not isinstance(
            self.config["main_chunk_start"], str
        ):
            raise TypeError("'main_chunk_start' must be a string")
        if "main_chunk_end" in self.config and not isinstance(
            self.config["main_chunk_end"], str
        ):
            raise TypeError("'main_chunk_end' must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the gather operation on the input data.

        Args:
            input_data (List[Dict]): The input data to process.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed results and the cost of the operation.
        """
        content_key = self.config["content_key"]
        doc_id_key = self.config["doc_id_key"]
        order_key = self.config["order_key"]
        peripheral_config = self.config.get("peripheral_chunks", {})
        main_chunk_start = self.config.get(
            "main_chunk_start", "--- Begin Main Chunk ---"
        )
        main_chunk_end = self.config.get("main_chunk_end", "--- End Main Chunk ---")
        doc_header_key = self.config.get("doc_header_key", None)
        results = []
        cost = 0.0

        # Group chunks by document ID
        grouped_chunks = {}
        for item in input_data:
            doc_id = item[doc_id_key]
            if doc_id not in grouped_chunks:
                grouped_chunks[doc_id] = []
            grouped_chunks[doc_id].append(item)

        # Process each group of chunks
        for chunks in grouped_chunks.values():
            # Sort chunks by their order within the document
            chunks.sort(key=lambda x: x[order_key])

            # Process each chunk with its peripheral context and headers
            for i, chunk in enumerate(chunks):
                rendered_chunk = self.render_chunk_with_context(
                    chunks,
                    i,
                    peripheral_config,
                    content_key,
                    order_key,
                    main_chunk_start,
                    main_chunk_end,
                    doc_header_key,
                )

                result = chunk.copy()
                result[f"{content_key}_rendered"] = rendered_chunk
                results.append(result)

        return results, cost

    def render_chunk_with_context(
        self,
        chunks: List[Dict],
        current_index: int,
        peripheral_config: Dict,
        content_key: str,
        order_key: str,
        main_chunk_start: str,
        main_chunk_end: str,
        doc_header_key: str,
    ) -> str:
        """
        Render a chunk with its peripheral context and headers.

        Args:
            chunks (List[Dict]): List of all chunks in the document.
            current_index (int): Index of the current chunk being processed.
            peripheral_config (Dict): Configuration for peripheral chunks.
            content_key (str): Key for the content in each chunk.
            order_key (str): Key for the order of each chunk.
            main_chunk_start (str): String to mark the start of the main chunk.
            main_chunk_end (str): String to mark the end of the main chunk.
            doc_header_key (str): The key for the headers in the current chunk.

        Returns:
            str: Renderted chunk with context and headers.
        """
        combined_parts = ["--- Previous Context ---"]

        combined_parts.extend(
            self.process_peripheral_chunks(
                chunks[:current_index],
                peripheral_config.get("previous", {}),
                content_key,
                order_key,
            )
        )
        combined_parts.append("--- End Previous Context ---\n")

        # Process main chunk
        main_chunk = chunks[current_index]
        if headers := self.render_hierarchy_headers(
            main_chunk, chunks[: current_index + 1], doc_header_key
        ):
            combined_parts.append(headers)
        combined_parts.extend(
            (
                f"{main_chunk_start}",
                f"{main_chunk[content_key]}",
                f"{main_chunk_end}",
                "\n--- Next Context ---",
            )
        )
        combined_parts.extend(
            self.process_peripheral_chunks(
                chunks[current_index + 1 :],
                peripheral_config.get("next", {}),
                content_key,
                order_key,
            )
        )
        combined_parts.append("--- End Next Context ---")

        return "\n".join(combined_parts)

    def process_peripheral_chunks(
        self,
        chunks: List[Dict],
        config: Dict,
        content_key: str,
        order_key: str,
        reverse: bool = False,
    ) -> List[str]:
        """
        Process peripheral chunks according to the configuration.

        Args:
            chunks (List[Dict]): List of chunks to process.
            config (Dict): Configuration for processing peripheral chunks.
            content_key (str): Key for the content in each chunk.
            order_key (str): Key for the order of each chunk.
            reverse (bool, optional): Whether to process chunks in reverse order. Defaults to False.

        Returns:
            List[str]: List of processed chunk strings.
        """
        if reverse:
            chunks = list(reversed(chunks))

        processed_parts = []
        included_chunks = []
        total_chunks = len(chunks)

        head_config = config.get("head", {})
        tail_config = config.get("tail", {})

        head_count = int(head_config.get("count", 0))
        tail_count = int(tail_config.get("count", 0))
        in_skip = False
        skip_char_count = 0

        for i, chunk in enumerate(chunks):
            if i < head_count:
                section = "head"
            elif i >= total_chunks - tail_count:
                section = "tail"
            elif "middle" in config:
                section = "middle"
            else:
                # Show number of characters skipped
                skipped_chars = len(chunk[content_key])
                if not in_skip:
                    skip_char_count = skipped_chars
                    in_skip = True
                else:
                    skip_char_count += skipped_chars

                continue

            if in_skip:
                processed_parts.append(
                    f"[... {skip_char_count} characters skipped ...]"
                )
                in_skip = False
                skip_char_count = 0

            section_config = config.get(section, {})
            section_content_key = section_config.get("content_key", content_key)

            is_summary = section_content_key != content_key
            summary_suffix = " (Summary)" if is_summary else ""

            chunk_prefix = f"[Chunk {chunk[order_key]}{summary_suffix}]"
            processed_parts.extend((chunk_prefix, f"{chunk[section_content_key]}"))
            included_chunks.append(chunk)

        if in_skip:
            processed_parts.append(f"[... {skip_char_count} characters skipped ...]")

        if reverse:
            processed_parts = list(reversed(processed_parts))

        return processed_parts

    def render_hierarchy_headers(
        self,
        current_chunk: Dict,
        chunks: List[Dict],
        doc_header_key: str,
    ) -> str:
        """
        Render headers for the current chunk's hierarchy.

        Args:
            current_chunk (Dict): The current chunk being processed.
            chunks (List[Dict]): List of chunks up to and including the current chunk.
            doc_header_key (str): The key for the headers in the current chunk.
        Returns:
            str: Renderted headers in the current chunk's hierarchy.
        """
        current_hierarchy = {}

        if doc_header_key is None:
            return ""

        # Find the largest/highest level in the current chunk
        current_chunk_headers = current_chunk.get(doc_header_key, [])
        highest_level = float("inf")  # Initialize with positive infinity
        for header_info in current_chunk_headers:
            level = header_info.get("level")
            if level is not None and level < highest_level:
                highest_level = level

        # If no headers found in the current chunk, set highest_level to None
        if highest_level == float("inf"):
            highest_level = None

        for chunk in chunks:
            for header_info in chunk.get(doc_header_key, []):
                header = header_info["header"]
                level = header_info["level"]
                if header and level:
                    current_hierarchy[level] = header
                    # Clear lower levels when a higher level header is found
                    for lower_level in range(level + 1, len(current_hierarchy) + 1):
                        if lower_level in current_hierarchy:
                            current_hierarchy[lower_level] = None

        rendered_headers = [
            f"{'#' * level} {header}"
            for level, header in sorted(current_hierarchy.items())
            if header is not None and (highest_level is None or level < highest_level)
        ]
        rendered_headers = " > ".join(rendered_headers)
        return f"_Current Section:_ {rendered_headers}" if rendered_headers else ""
