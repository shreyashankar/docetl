"""Runs every pipeline example in the repository offline, in both modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from litellm import EmbeddingResponse, ModelResponse

from docetl.execution import BatchResult
from docetl.runner import DSLRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pipeline_examples() -> list[Path]:
    examples = []
    for path in PROJECT_ROOT.rglob("*.yaml"):
        if any(part.startswith(".") for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        contents = path.read_text(encoding="utf-8")
        if "pipeline:" in contents and "operations:" in contents:
            examples.append(path)
    return sorted(examples)


def _value_for_schema(name: str, schema: dict[str, Any]) -> Any:
    if name == "should_refine":
        return False
    special_values = {
        "category": "billing",
        "priority": "high",
        "sentiment": "neutral",
        "date": "2025-01-01",
        "role": "witness",
    }
    if name in special_values:
        return special_values[name]
    kind = schema.get("type")
    if kind == "boolean":
        return True
    if kind == "integer":
        return 1
    if kind == "number":
        return 0.5
    if kind == "array":
        return [_value_for_schema(name, schema.get("items", {"type": "string"}))]
    if kind == "object":
        return {
            key: _value_for_schema(key, value)
            for key, value in schema.get("properties", {}).items()
        }
    return f"sample {name}"


def _fake_completion(**kwargs: Any) -> ModelResponse:
    tools = kwargs.get("tools") or []
    response_format = kwargs.get("response_format")
    if tools:
        function = tools[0]["function"]
        properties = function["parameters"].get("properties", {})
        payload = {
            name: _value_for_schema(name, schema) for name, schema in properties.items()
        }
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-example",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(payload),
                    },
                }
            ],
        }
    elif response_format:
        schema = response_format.get("json_schema", {}).get("schema", {})
        payload = {
            name: _value_for_schema(name, value)
            for name, value in schema.get("properties", {}).items()
        }
        message = {"role": "assistant", "content": json.dumps(payload)}
    else:
        message = {"role": "assistant", "content": "sample response"}
    return ModelResponse(
        id="example-smoke",
        object="chat.completion",
        created=0,
        model="gpt-4o-mini",
        choices=[{"index": 0, "finish_reason": "stop", "message": message}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _fake_embedding(
    *, model: str, input: list[str], **kwargs: Any
) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        data=[
            {
                "object": "embedding",
                "index": index,
                "embedding": [1.0, 0.01 * index, 0.0],
            }
            for index, _ in enumerate(input)
        ],
        usage={"prompt_tokens": len(input), "total_tokens": len(input)},
    )


def _synthetic_rows(example: Path) -> list[dict[str, Any]]:
    marker = str(example.relative_to(PROJECT_ROOT))
    base = {
        "id": marker + "-1",
        "hearing_id": "H1",
        "text": "Sample text. Second sentence.",
        "contents": "Sample contract text.",
        "content": "First paragraph.\n\nSecond paragraph.",
        "transcript": "Speaker: Sample testimony about health policy.",
        "title": "Sample title",
        "committee": "Sample committee",
        "date": "2025-01-01",
        "system": "SampleSystem",
        "prompt_text": "Always verify inputs before acting.",
        "email_text": "From: Alice <alice@example.com>\nSubject: Test\n\nHello Bob.",
        "filename": "sample.txt",
        "file_name": "sample.txt",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "response": "Sample response to an AI policy request.",
        "author": "Alice",
        "organization": "Example University",
    }
    return [base, {**base, "id": marker + "-2", "name": "A. Smith"}]


@pytest.mark.parametrize(
    "example_path",
    _pipeline_examples(),
    ids=lambda path: str(path.relative_to(PROJECT_ROOT)),
)
@pytest.mark.parametrize("execution_mode", ["legacy", "compiled_batch"])
def test_pipeline_example_executes_offline(
    example_path: Path,
    execution_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import docetl.execution.runtime as runtime_module
    import docetl.operations.utils.api as api_module

    monkeypatch.setattr(api_module, "completion", _fake_completion)
    monkeypatch.setattr(api_module, "embedding", _fake_embedding)

    def fake_execute_batch(requests, config):
        return [
            BatchResult(
                request.custom_id,
                _fake_completion(**request.body).model_dump(),
                status_code=200,
            )
            for request in requests
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    rows = _synthetic_rows(example_path)
    for name in config.get("datasets", {}):
        config["datasets"][name] = {"type": "memory", "path": rows}
    config.pop("fallback_models", None)
    config.pop("fallback_embedding_models", None)
    config["bypass_cache"] = True
    for index, operation in enumerate(config.get("operations", [])):
        operation["bypass_cache"] = True
        incompatible = any(
            operation.get(field)
            for field in (
                "agent",
                "batch_prompt",
                "calibrate",
                "cascade",
                "fold_prompt",
                "gleaning",
                "merge_prompt",
                "pdf_url_key",
                "validate",
            )
        )
        if (
            execution_mode == "compiled_batch"
            and operation.get("type") in {"map", "filter", "reduce"}
            and not incompatible
        ):
            operation["model"] = "gpt-4o-mini"
            operation["execution"] = {
                "mode": "batch",
                "backend": "litellm",
                "provider": "openai",
                "work_dir": str(tmp_path / f"batch-{index}"),
            }
    config["pipeline"]["output"] = {
        "type": "file",
        "path": str(tmp_path / "output.json"),
        "intermediate_dir": str(tmp_path / "intermediate"),
    }

    runner = DSLRunner(config, max_threads=2)
    output, _ = runner.run()

    assert isinstance(output, list)
