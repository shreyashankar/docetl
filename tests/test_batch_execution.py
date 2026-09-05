import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from docetl.execution import (
    BatchExecutionError,
    BatchRequest,
    BatchResult,
    ExecutionPolicy,
    ExecutionTarget,
    LogicalLLMRequest,
    compile_execution_plan,
    execute_batch,
    execute_plan,
    partition_batch_requests,
)
from docetl.operations.map import MapOperation
from docetl.operations.reduce import ReduceOperation
from docetl.runner import DSLRunner


def _response_row(custom_id: str, value: str) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"completion-{custom_id}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{custom_id}",
                                    "type": "function",
                                    "function": {
                                        "name": "send_output",
                                        "arguments": json.dumps({"label": value}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        },
        "error": None,
    }


def test_openai_batch_persists_and_orders_results(tmp_path, monkeypatch):
    import litellm

    calls = {"upload": 0, "create": 0, "retrieve": 0, "content": 0}

    def create_file(**kwargs):
        calls["upload"] += 1
        assert kwargs["purpose"] == "batch"
        return SimpleNamespace(id="file-input")

    def create_batch(**kwargs):
        calls["create"] += 1
        assert kwargs["endpoint"] == "/v1/chat/completions"
        return SimpleNamespace(id="batch-1")

    def retrieve_batch(**kwargs):
        calls["retrieve"] += 1
        return SimpleNamespace(
            status="completed", output_file_id="file-output", error_file_id=None
        )

    def file_content(**kwargs):
        calls["content"] += 1
        # Providers do not guarantee output order.
        return SimpleNamespace(
            text="".join(
                json.dumps(row) + "\n"
                for row in [
                    _response_row("second", "B"),
                    _response_row("first", "A"),
                ]
            )
        )

    monkeypatch.setattr(litellm, "create_file", create_file)
    monkeypatch.setattr(litellm, "create_batch", create_batch)
    monkeypatch.setattr(litellm, "retrieve_batch", retrieve_batch)
    monkeypatch.setattr(litellm, "file_content", file_content)

    requests = [
        BatchRequest("first", {"model": "gpt-4o-mini", "messages": []}),
        BatchRequest("second", {"model": "gpt-4o-mini", "messages": []}),
    ]
    config = {"backend": "litellm", "work_dir": str(tmp_path)}
    results = execute_batch(requests, config)

    assert [result.custom_id for result in results] == ["first", "second"]
    assert calls == {"upload": 1, "create": 1, "retrieve": 1, "content": 1}

    # Completed output is a durable checkpoint; a rerun must not resubmit.
    assert execute_batch(requests, config) == results
    assert calls == {"upload": 1, "create": 1, "retrieve": 1, "content": 1}


def test_openai_batch_resumes_existing_provider_job(tmp_path, monkeypatch):
    import litellm

    from docetl.execution import batch as batch_module

    requests = [BatchRequest("one", {"model": "gpt-4o-mini", "messages": []})]
    config = {"backend": "litellm", "work_dir": str(tmp_path)}
    job_dir = batch_module._job_directory(requests, config)
    job_dir.mkdir(parents=True)
    (job_dir / "manifest.json").write_text(
        json.dumps({"batch_id": "existing-batch"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        litellm,
        "create_file",
        lambda **kwargs: pytest.fail("resume must not upload a second file"),
    )
    monkeypatch.setattr(
        litellm,
        "create_batch",
        lambda **kwargs: pytest.fail("resume must not create a second batch"),
    )
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda **kwargs: SimpleNamespace(
            status="completed", output_file_id="output", error_file_id=None
        ),
    )
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda **kwargs: SimpleNamespace(
            text=json.dumps(_response_row("one", "done")) + "\n"
        ),
    )

    assert execute_batch(requests, config)[0].custom_id == "one"


def test_provider_batch_recovers_after_partial_result_download(tmp_path, monkeypatch):
    """A downloaded success file must not poison retries if errors are missing."""
    import litellm

    calls = {"create": 0, "content": []}
    requests = [
        BatchRequest("ok", {"model": "m", "messages": []}),
        BatchRequest("failed", {"model": "m", "messages": []}),
    ]

    monkeypatch.setattr(
        litellm, "create_file", lambda **kwargs: SimpleNamespace(id="input")
    )

    def create_batch(**kwargs):
        calls["create"] += 1
        return SimpleNamespace(id="batch-partial")

    monkeypatch.setattr(litellm, "create_batch", create_batch)
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda **kwargs: SimpleNamespace(
            status="completed", output_file_id="successes", error_file_id="errors"
        ),
    )

    fail_error_download = True

    def file_content(*, file_id, **kwargs):
        nonlocal fail_error_download
        calls["content"].append(file_id)
        if file_id == "successes":
            return SimpleNamespace(text=json.dumps(_response_row("ok", "done")) + "\n")
        if fail_error_download:
            fail_error_download = False
            raise ConnectionError("connection dropped while downloading errors")
        return SimpleNamespace(
            text=json.dumps(
                {
                    "custom_id": "failed",
                    "response": None,
                    "error": {"code": "invalid_request", "message": "bad input"},
                }
            )
            + "\n"
        )

    monkeypatch.setattr(litellm, "file_content", file_content)
    config = {"backend": "litellm", "work_dir": str(tmp_path)}

    with pytest.raises(ConnectionError, match="downloading errors"):
        execute_batch(requests, config)

    results = execute_batch(requests, config)

    assert calls["create"] == 1
    assert calls["content"] == ["successes", "errors", "successes", "errors"]
    assert [result.custom_id for result in results] == ["ok", "failed"]
    assert results[1].body is None
    assert results[1].error["code"] == "invalid_request"


def test_identical_concurrent_runners_submit_only_one_provider_job(
    tmp_path, monkeypatch
):
    import litellm

    calls = {"upload": 0, "create": 0}

    def create_file(**kwargs):
        calls["upload"] += 1
        # Make an unlocked implementation deterministically overlap here.
        time.sleep(0.05)
        return SimpleNamespace(id="input")

    def create_batch(**kwargs):
        calls["create"] += 1
        return SimpleNamespace(id="one-job")

    monkeypatch.setattr(litellm, "create_file", create_file)
    monkeypatch.setattr(litellm, "create_batch", create_batch)
    monkeypatch.setattr(
        litellm,
        "retrieve_batch",
        lambda **kwargs: SimpleNamespace(
            status="completed", output_file_id="output", error_file_id=None
        ),
    )
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda **kwargs: SimpleNamespace(
            text=json.dumps(_response_row("one", "done")) + "\n"
        ),
    )
    request = BatchRequest("one", {"model": "m", "messages": []})
    config = {"backend": "litellm", "work_dir": str(tmp_path)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute_batch, [request], config) for _ in range(2)]
        results = [future.result() for future in futures]

    assert calls == {"upload": 1, "create": 1}
    assert results[0] == results[1]


def test_job_lock_serializes_independent_processes(tmp_path):
    import multiprocessing
    import queue

    from docetl.execution import batch as batch_module

    if batch_module.fcntl is None:
        pytest.skip("cross-process advisory lock requires POSIX fcntl")
    context = multiprocessing.get_context("fork")
    entered = context.Queue()
    release = context.Event()

    def hold_lock():
        with batch_module._job_lock(tmp_path / "same-job"):
            entered.put("entered")
            release.wait(timeout=5)

    first = context.Process(target=hold_lock)
    second = context.Process(target=hold_lock)
    try:
        first.start()
        assert entered.get(timeout=2) == "entered"
        second.start()
        with pytest.raises(queue.Empty):
            entered.get(timeout=0.2)
        release.set()
        assert entered.get(timeout=2) == "entered"
    finally:
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)
        if first.is_alive():
            first.terminate()
        if second.is_alive():
            second.terminate()

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_provider_batch_timeout_resumes_without_resubmission(tmp_path, monkeypatch):
    import litellm

    calls = {"create": 0, "retrieve": 0}
    complete = False
    monkeypatch.setattr(
        litellm, "create_file", lambda **kwargs: SimpleNamespace(id="input")
    )

    def create_batch(**kwargs):
        calls["create"] += 1
        return SimpleNamespace(id="slow-job")

    def retrieve_batch(**kwargs):
        calls["retrieve"] += 1
        if complete:
            return SimpleNamespace(
                status="completed", output_file_id="output", error_file_id=None
            )
        return SimpleNamespace(status="in_progress")

    monkeypatch.setattr(litellm, "create_batch", create_batch)
    monkeypatch.setattr(litellm, "retrieve_batch", retrieve_batch)
    monkeypatch.setattr(
        litellm,
        "file_content",
        lambda **kwargs: SimpleNamespace(
            text=json.dumps(_response_row("one", "done")) + "\n"
        ),
    )
    request = BatchRequest("one", {"model": "m", "messages": []})
    config = {
        "backend": "litellm",
        "work_dir": str(tmp_path),
        "timeout_seconds": 0,
        "poll_interval_seconds": 0,
    }

    with pytest.raises(TimeoutError, match="rerun the pipeline to resume"):
        execute_batch([request], config)
    complete = True

    assert execute_batch([request], config)[0].custom_id == "one"
    assert calls["create"] == 1
    assert calls["retrieve"] == 2


def test_batch_job_identity_ignores_scheduler_policy(tmp_path):
    from docetl.execution import batch as batch_module

    requests = [BatchRequest("one", {"model": "m", "messages": []})]
    first = batch_module._job_directory(
        requests,
        {
            "backend": "litellm",
            "work_dir": str(tmp_path),
            "deadline_seconds": 60,
            "fallback": {"mode": "direct", "concurrency": 2},
        },
    )
    second = batch_module._job_directory(
        requests,
        {
            "backend": "litellm",
            "work_dir": str(tmp_path),
            "deadline_seconds": 600,
            "fallback": {"mode": "direct", "concurrency": 20},
        },
    )

    assert first == second


def test_batch_rejects_duplicate_output_ids(tmp_path):
    from docetl.execution import batch as batch_module

    requests = [
        BatchRequest("one", {"model": "m", "messages": []}),
        BatchRequest("two", {"model": "m", "messages": []}),
    ]
    config = {"backend": "vllm", "work_dir": str(tmp_path), "model": "m"}
    job_dir = batch_module._job_directory(requests, config)
    job_dir.mkdir(parents=True)
    duplicate = _response_row("one", "x")
    duplicate["response"] = duplicate["response"]["body"]
    (job_dir / "output.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BatchExecutionError, match="duplicate custom_id"):
        execute_batch(requests, config)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_response_row("foreign", "x")], "unexpected request"),
        ([{"response": {}}], "string custom_id"),
        ([{"custom_id": 7, "response": {}}], "string custom_id"),
        ([{"custom_id": "one", "response": "invalid"}], "response must be"),
    ],
)
def test_batch_rejects_foreign_or_malformed_output_rows(tmp_path, rows, message):
    from docetl.execution import batch as batch_module

    requests = [BatchRequest("one", {"model": "m", "messages": []})]
    config = {"backend": "vllm", "work_dir": str(tmp_path), "model": "m"}
    job_dir = batch_module._job_directory(requests, config)
    job_dir.mkdir(parents=True)
    (job_dir / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(BatchExecutionError, match=message):
        execute_batch(requests, config)


def test_vllm_batch_uses_safe_argv_and_openai_jsonl(tmp_path, monkeypatch):
    from docetl.execution import batch as batch_module

    captured = {}

    def fake_run(argv, check):
        captured["argv"] = argv
        assert check is True
        input_path = Path(argv[argv.index("-i") + 1])
        output_path = Path(argv[argv.index("-o") + 1])
        request = json.loads(input_path.read_text(encoding="utf-8"))
        assert request["body"]["model"] == "Qwen/Qwen3-8B"
        assert request["body"]["max_tokens"] == 512
        response = _response_row("one", "local")
        # vLLM writes the endpoint body directly rather than OpenAI's wrapper.
        response["response"] = response["response"]["body"]
        output_path.write_text(json.dumps(response) + "\n", encoding="utf-8")

    monkeypatch.setattr(batch_module.subprocess, "run", fake_run)
    results = execute_batch(
        [BatchRequest("one", {"model": "ignored", "messages": []})],
        {
            "backend": "vllm",
            "work_dir": str(tmp_path),
            "model": "Qwen/Qwen3-8B",
            "request_defaults": {"max_tokens": 512},
            "engine_args": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            },
        },
    )

    assert results[0].body["model"] == "gpt-4o-mini"
    assert captured["argv"][:2] == ["vllm", "run-batch"]
    assert "--tensor-parallel-size" in captured["argv"]
    assert "--enforce-eager" in captured["argv"]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError(), "command was not found"),
        (subprocess.CalledProcessError(17, ["vllm"]), "exited with status 17"),
    ],
)
def test_vllm_process_failures_are_actionable(tmp_path, monkeypatch, failure, message):
    from docetl.execution import batch as batch_module

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(batch_module.subprocess, "run", fail)
    with pytest.raises(BatchExecutionError, match=message):
        execute_batch(
            [BatchRequest("one", {"model": "m", "messages": []})],
            {"backend": "vllm", "work_dir": str(tmp_path), "model": "m"},
        )


def test_vllm_rejects_successful_process_without_output(tmp_path, monkeypatch):
    from docetl.execution import batch as batch_module

    monkeypatch.setattr(batch_module.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(BatchExecutionError, match="did not create its output"):
        execute_batch(
            [BatchRequest("one", {"model": "m", "messages": []})],
            {"backend": "vllm", "work_dir": str(tmp_path), "model": "m"},
        )


def test_batch_partition_respects_count_and_encoded_bytes():
    requests = [
        BatchRequest(str(index), {"model": "m", "messages": [], "value": "x" * 20})
        for index in range(3)
    ]
    one_size = len(
        (json.dumps(requests[0].as_dict(), sort_keys=True) + "\n").encode("utf-8")
    )

    assert [
        len(chunk) for chunk in partition_batch_requests(requests, max_requests=2)
    ] == [2, 1]
    assert [
        len(chunk)
        for chunk in partition_batch_requests(
            requests, max_requests=10, max_bytes=one_size * 2 - 1
        )
    ] == [1, 1, 1]


def test_map_compiles_one_request_per_row_and_preserves_order(monkeypatch):
    import docetl.execution.runtime as runtime_module

    seen = {}

    def fake_execute_batch(requests, config):
        seen["requests"] = requests
        seen["config"] = config
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, value)["response"]["body"],
            )
            for request, value in zip(requests, ["A", "B"])
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        },
        max_threads=2,
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute([{"text": "one"}, {"text": "two"}])

    assert output == [
        {"text": "one", "label": "A"},
        {"text": "two", "label": "B"},
    ]
    assert len(seen["requests"]) == 2
    assert seen["requests"][0].body["tools"][0]["function"]["name"] == "send_output"


def test_map_splits_provider_request_limit(monkeypatch):
    import docetl.execution.runtime as runtime_module

    batch_sizes = []

    def fake_execute_batch(requests, config):
        batch_sizes.append(len(requests))
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, "ok")["response"]["body"],
            )
            for request in requests
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {
                "mode": "batch",
                "backend": "litellm",
                "max_batch_requests": 2,
            },
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute([{"text": str(i)} for i in range(5)])

    assert len(output) == 5
    assert batch_sizes == [2, 2, 1]


def test_litellm_provider_is_inferred_from_model(monkeypatch):
    import docetl.execution.runtime as runtime_module

    captured = {}

    def fake_execute_batch(requests, config):
        captured["provider"] = config["provider"]
        captured["model"] = requests[0].body["model"]
        return [
            BatchResult(
                requests[0].custom_id,
                _response_row(requests[0].custom_id, "ok")["response"]["body"],
            )
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "vertex_ai/gemini-2.5-flash",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "model": "vertex_ai/gemini-2.5-flash",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "bypass_cache": True,
        },
        "vertex_ai/gemini-2.5-flash",
        1,
    )

    operation.execute([{"text": "one"}])

    assert captured == {"provider": "vertex_ai", "model": "gemini-2.5-flash"}


def test_vllm_deepseek_string_request_preserves_no_tool_call_path():
    runner = DSLRunner(
        {
            "default_model": "deepseek-r1",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )

    body = runner.api._build_completion_request(
        model="deepseek-r1",
        op_type="map",
        messages=[{"role": "user", "content": "Summarize"}],
        output_schema={"summary": "string"},
        litellm_completion_kwargs={},
        op_config={"output": {"schema": {"summary": "string"}}},
        execution_config={
            "mode": "batch",
            "backend": "vllm",
            "model": "deepseek-ai/DeepSeek-R1",
        },
    )

    assert body["model"] == "deepseek-ai/DeepSeek-R1"
    assert "tools" not in body
    assert "tool_choice" not in body


def test_inferred_unsupported_litellm_batch_provider_is_rejected():
    runner = DSLRunner(
        {
            "default_model": "anthropic/claude-sonnet-4-5",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )

    with pytest.raises(ValueError, match="currently supports provider values"):
        runner.api.call_llm_compiled(
            [
                {
                    "model": "anthropic/claude-sonnet-4-5",
                    "op_type": "map",
                    "messages": [{"role": "user", "content": "Classify"}],
                    "output_schema": {"label": "string"},
                    "op_config": {"output": {"schema": {"label": "string"}}},
                    "bypass_cache": True,
                }
            ],
            {"mode": "batch", "backend": "litellm"},
        )


@pytest.mark.parametrize("field", ["agent", "gleaning", "validate", "calibrate"])
def test_batch_execution_rejects_multi_round_map_features(field):
    config = {
        "name": "classify",
        "type": "map",
        "prompt": "Classify {{ input.text }}",
        "output": {"schema": {"label": "string"}},
        "execution": {"mode": "batch", "backend": "litellm"},
    }
    config[field] = (
        {"num_rounds": 1, "validation_prompt": "check"}
        if field == "gleaning"
        else ["output['label']"] if field == "validate" else True
    )
    with pytest.raises(ValueError, match="cannot yet be combined"):
        MapOperation.schema.model_validate(config)


def test_execution_compiler_produces_stable_direct_plan():
    request = LogicalLLMRequest(
        request_id="row-1",
        operation_name="classify",
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Classify this"}],
        output_schema={"label": "string"},
    )
    first_policy = ExecutionPolicy.from_config(
        {"mode": "direct", "concurrency": 4, "poll_interval_seconds": 1}
    )
    second_policy = ExecutionPolicy.from_config(
        {"mode": "direct", "concurrency": 99, "poll_interval_seconds": 99}
    )

    def build_body(logical, policy):
        return {"model": logical.model, "messages": logical.messages}

    first = compile_execution_plan([request], first_policy, build_body)
    second = compile_execution_plan([request], second_policy, build_body)

    assert first.target is ExecutionTarget.DIRECT
    assert first.plan_id == second.plan_id
    assert first.requests[0].logical is request


def test_direct_execution_uses_same_compiled_map_path(monkeypatch):
    import litellm

    seen = []

    def fake_completion(**kwargs):
        seen.append(kwargs)
        value = "A" if "one" in kwargs["messages"][-1]["content"] else "B"
        return _response_row("direct", value)["response"]["body"]

    monkeypatch.setattr(litellm, "completion", fake_completion)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "direct", "concurrency": 2},
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute([{"text": "one"}, {"text": "two"}])

    assert output == [
        {"text": "one", "label": "A"},
        {"text": "two", "label": "B"},
    ]
    assert len(seen) == 2
    assert {call["model"] for call in seen} == {"gpt-4o-mini"}


def test_runtime_reports_direct_target_for_direct_plan(monkeypatch):
    import litellm

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: _response_row("one", "ok")["response"]["body"],
    )
    policy = ExecutionPolicy.from_config({"mode": "direct"})
    logical = LogicalLLMRequest("one", "map", "gpt-4o-mini", [], {"label": "string"})
    plan = compile_execution_plan(
        [logical],
        policy,
        lambda request, selected_policy: {
            "model": request.model,
            "messages": request.messages,
        },
    )

    target, results = execute_plan(plan)

    assert target is ExecutionTarget.DIRECT
    assert results[0].request_id == "one"


def test_reduce_compiles_one_request_per_group(monkeypatch):
    import docetl.execution.runtime as runtime_module

    seen = {}

    def fake_execute_batch(requests, config):
        seen["requests"] = requests
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, value)["response"]["body"],
            )
            for request, value in zip(requests, ["first summary", "second summary"])
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = ReduceOperation(
        runner,
        {
            "name": "summarize_by_team",
            "type": "reduce",
            "reduce_key": "team",
            "prompt": "Summarize {{ inputs }} for {{ reduce_key.team }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "enable_observability": True,
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    output, _ = operation.execute(
        [
            {"team": "red", "text": "one"},
            {"team": "red", "text": "two"},
            {"team": "blue", "text": "three"},
        ]
    )

    assert [(row["team"], row["label"]) for row in output] == [
        ("red", "first summary"),
        ("blue", "second summary"),
    ]
    assert output[0]["_counts_prereduce_summarize_by_team"] == 2
    assert len(seen["requests"]) == 2
    assert (
        "many inputs:one output" in seen["requests"][0].body["messages"][0]["content"]
    )


def test_compiled_reduce_rejects_dependent_fold_rounds():
    with pytest.raises(ValueError, match="one request per group"):
        ReduceOperation.schema.model_validate(
            {
                "name": "fold",
                "type": "reduce",
                "reduce_key": "team",
                "prompt": "Summarize {{ inputs }}",
                "fold_prompt": "Fold {{ inputs }} into {{ output }}",
                "fold_batch_size": 2,
                "output": {"schema": {"label": "string"}},
                "execution": {"mode": "batch", "backend": "litellm"},
            }
        )


def test_provider_deadline_fallback_is_durable_and_atomic(tmp_path, monkeypatch):
    import litellm

    import docetl.execution.runtime as runtime_module

    calls = {"batch": 0, "direct": 0}

    def timed_out_batch(requests, config):
        calls["batch"] += 1
        assert 0 <= config["timeout_seconds"] <= 0.01
        raise TimeoutError("provider deadline exceeded")

    def direct_completion(**kwargs):
        calls["direct"] += 1
        assert kwargs["model"] == "openai/gpt-4o-mini"
        return _response_row("fallback", "direct winner")["response"]["body"]

    monkeypatch.setattr(runtime_module, "execute_batch", timed_out_batch)
    monkeypatch.setattr(litellm, "completion", direct_completion)
    policy = ExecutionPolicy.from_config(
        {
            "mode": "batch",
            "backend": "litellm",
            "provider": "openai",
            "work_dir": str(tmp_path),
            "deadline_seconds": 0.01,
            "fallback": {"mode": "direct", "concurrency": 2},
        }
    )
    logical = LogicalLLMRequest(
        "one",
        "classify",
        "openai/gpt-4o-mini",
        [{"role": "user", "content": "classify"}],
        {"label": "string"},
    )
    plan = compile_execution_plan(
        [logical],
        policy,
        lambda request, selected_policy: {
            "model": "gpt-4o-mini",
            "messages": request.messages,
        },
    )

    target, results = execute_plan(plan)

    assert target is ExecutionTarget.DIRECT
    assert results[0].response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] == json.dumps({"label": "direct winner"})
    assert calls == {"batch": 1, "direct": 1}

    # A rerun obeys the persisted winner and result. It must neither wait for
    # the late provider job nor publish a second direct result.
    monkeypatch.setattr(
        runtime_module,
        "execute_batch",
        lambda *args, **kwargs: pytest.fail("provider must not be read after fallback"),
    )
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: pytest.fail("completed direct fallback must be replayed"),
    )
    replay_target, replay_results = execute_plan(plan)

    assert replay_target is ExecutionTarget.DIRECT
    assert replay_results == results


def test_fallback_requires_explicit_deadline():
    with pytest.raises(ValueError, match="deadline_seconds is required"):
        ExecutionPolicy.from_config(
            {
                "mode": "batch",
                "backend": "litellm",
                "fallback": "direct",
            }
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "unknown"}, "execution must use mode"),
        ({"mode": "direct", "concurrency": 0}, "greater than zero"),
        (
            {"mode": "direct", "deadline_seconds": 1, "fallback": "direct"},
            "only for provider batch",
        ),
        (
            {
                "mode": "batch",
                "backend": "litellm",
                "deadline_seconds": -1,
            },
            "deadline_seconds must be greater than zero",
        ),
    ],
)
def test_execution_policy_rejects_invalid_scheduler_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        ExecutionPolicy.from_config(config)


def test_atomic_winner_publication_has_one_consistent_winner(tmp_path):
    import docetl.execution.runtime as runtime_module

    winner_path = tmp_path / "winner.json"
    barrier = threading.Barrier(16)

    def claim(index):
        barrier.wait()
        target = ExecutionTarget.DIRECT if index % 2 else ExecutionTarget.PROVIDER_BATCH
        return runtime_module._claim_winner(winner_path, target)

    with ThreadPoolExecutor(max_workers=16) as executor:
        winners = list(executor.map(claim, range(16)))

    assert len(set(winners)) == 1
    assert (
        json.loads(winner_path.read_text(encoding="utf-8"))["winner"]
        == winners[0].value
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_provider_result_wins_before_deadline_without_direct_calls(
    tmp_path, monkeypatch
):
    import litellm

    import docetl.execution.runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "execute_batch",
        lambda requests, config: [
            BatchResult(
                requests[0].custom_id,
                _response_row(requests[0].custom_id, "provider winner")["response"][
                    "body"
                ],
                status_code=200,
            )
        ],
    )
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: pytest.fail("direct fallback must not run"),
    )
    policy = ExecutionPolicy.from_config(
        {
            "mode": "batch",
            "backend": "litellm",
            "provider": "openai",
            "work_dir": str(tmp_path),
            "deadline_seconds": 60,
            "fallback": "direct",
        }
    )
    logical = LogicalLLMRequest(
        "one", "classify", "openai/gpt-4o-mini", [], {"label": "string"}
    )
    plan = compile_execution_plan(
        [logical],
        policy,
        lambda request, selected_policy: {
            "model": "gpt-4o-mini",
            "messages": request.messages,
        },
    )

    target, results = execute_plan(plan)

    assert target is ExecutionTarget.PROVIDER_BATCH
    assert results[0].status_code == 200


def test_chunk_deadline_falls_back_the_whole_plan(tmp_path, monkeypatch):
    import litellm

    import docetl.execution.runtime as runtime_module

    batch_calls = []

    def provider_batch(requests, config):
        batch_calls.append([request.custom_id for request in requests])
        if len(batch_calls) == 2:
            raise TimeoutError("second chunk missed the deadline")
        return [
            BatchResult(
                request.custom_id,
                _response_row(request.custom_id, "provider partial")["response"][
                    "body"
                ],
                status_code=200,
            )
            for request in requests
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", provider_batch)
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: _response_row("direct", "direct whole plan")["response"][
            "body"
        ],
    )
    policy = ExecutionPolicy.from_config(
        {
            "mode": "batch",
            "backend": "litellm",
            "provider": "openai",
            "max_batch_requests": 2,
            "work_dir": str(tmp_path),
            "deadline_seconds": 60,
            "fallback": {"mode": "direct", "concurrency": 3},
        }
    )
    logical_requests = [
        LogicalLLMRequest(
            str(index),
            "classify",
            "openai/gpt-4o-mini",
            [{"role": "user", "content": str(index)}],
            {"label": "string"},
        )
        for index in range(3)
    ]
    plan = compile_execution_plan(
        logical_requests,
        policy,
        lambda request, selected_policy: {
            "model": "gpt-4o-mini",
            "messages": request.messages,
        },
    )

    target, results = execute_plan(plan)

    assert batch_calls == [["0", "1"], ["2"]]
    assert target is ExecutionTarget.DIRECT
    assert len(results) == 3
    assert all(
        json.loads(
            result.response["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"
            ]
        )["label"]
        == "direct whole plan"
        for result in results
    )


@pytest.mark.parametrize("skip_on_error", [False, True])
def test_compiled_map_has_explicit_per_row_failure_semantics(
    monkeypatch, skip_on_error
):
    import docetl.execution.runtime as runtime_module

    def fake_execute_batch(requests, config):
        return [
            BatchResult(
                requests[0].custom_id,
                None,
                error={"code": "rate_limit"},
                status_code=429,
            ),
            BatchResult(
                requests[1].custom_id,
                _response_row(requests[1].custom_id, "survivor")["response"]["body"],
                status_code=200,
            ),
        ]

    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    operation = MapOperation(
        runner,
        {
            "name": "classify",
            "type": "map",
            "prompt": "Classify {{ input.text }}",
            "output": {"schema": {"label": "string"}},
            "execution": {"mode": "batch", "backend": "litellm"},
            "skip_on_error": skip_on_error,
            "bypass_cache": True,
        },
        "gpt-4o-mini",
        2,
    )

    if not skip_on_error:
        with pytest.raises(Exception, match="Batch request failed"):
            operation.execute([{"text": "bad"}, {"text": "good"}])
        return

    output, _ = operation.execute([{"text": "bad"}, {"text": "good"}])
    assert output == [{"text": "good", "label": "survivor"}]


def test_execution_policy_does_not_fragment_semantic_llm_cache(tmp_path, monkeypatch):
    from diskcache import Cache
    from litellm import ModelResponse

    import docetl.execution.runtime as runtime_module
    import docetl.operations.utils.api as api_module

    isolated_cache = Cache(str(tmp_path / "cache"))
    monkeypatch.setattr(api_module, "cache", isolated_cache)
    calls = {"legacy": 0}

    def legacy_completion(**kwargs):
        calls["legacy"] += 1
        return ModelResponse(**_response_row("legacy", "same")["response"]["body"])

    monkeypatch.setattr(api_module, "completion", legacy_completion)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    base_config = {
        "name": "classify",
        "type": "map",
        "prompt": "Classify {{ input.text }}",
        "output": {"schema": {"label": "string"}},
    }
    legacy = MapOperation(runner, base_config, "gpt-4o-mini", 1)
    first, _ = legacy.execute([{"text": "one"}])

    monkeypatch.setattr(
        runtime_module,
        "execute_batch",
        lambda *args, **kwargs: pytest.fail(
            "compiled execution should reuse the semantic cache"
        ),
    )
    compiled = MapOperation(
        runner,
        {
            **base_config,
            "execution": {"mode": "batch", "backend": "litellm"},
        },
        "gpt-4o-mini",
        1,
    )
    second, _ = compiled.execute([{"text": "one"}])

    assert first == second == [{"text": "one", "label": "same"}]
    assert calls == {"legacy": 1}
    isolated_cache.close()


def test_compiled_reduce_preserves_legacy_group_metadata(monkeypatch):
    from litellm import ModelResponse

    import docetl.execution.runtime as runtime_module
    import docetl.operations.utils.api as api_module

    def label_for_messages(messages):
        user_text = str(messages[-1].get("content", ""))
        return "red summary" if user_text.endswith("for red") else "blue summary"

    def legacy_completion(**kwargs):
        return ModelResponse(
            **_response_row("legacy", label_for_messages(kwargs["messages"]))[
                "response"
            ]["body"]
        )

    def fake_execute_batch(requests, config):
        return [
            BatchResult(
                request.custom_id,
                _response_row(
                    request.custom_id, label_for_messages(request.body["messages"])
                )["response"]["body"],
                status_code=200,
            )
            for request in requests
        ]

    monkeypatch.setattr(api_module, "completion", legacy_completion)
    monkeypatch.setattr(runtime_module, "execute_batch", fake_execute_batch)
    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/out.json"}},
        }
    )
    base_config = {
        "name": "summarize_by_team",
        "type": "reduce",
        "reduce_key": "team",
        "prompt": "Summarize {{ inputs }} for {{ reduce_key.team }}",
        "output": {"schema": {"label": "string"}, "lineage": ["id"]},
        "pass_through": True,
        "enable_observability": True,
        "bypass_cache": True,
    }
    rows = [
        {"id": 1, "team": "red", "text": "one", "owner": "Alice"},
        {"id": 2, "team": "red", "text": "two", "owner": "Bob"},
        {"id": 3, "team": "blue", "text": "three", "owner": "Carol"},
    ]
    legacy = ReduceOperation(runner, base_config, "gpt-4o-mini", 2)
    compiled = ReduceOperation(
        runner,
        {
            **base_config,
            "execution": {"mode": "batch", "backend": "litellm"},
        },
        "gpt-4o-mini",
        2,
    )

    legacy_output, _ = legacy.execute(rows)
    compiled_output, _ = compiled.execute(rows)

    assert {row["team"]: row for row in compiled_output} == {
        row["team"]: row for row in legacy_output
    }
