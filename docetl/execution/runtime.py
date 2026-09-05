"""Runtime for executing compiled LLM plans through a common contract."""

from __future__ import annotations

import json
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .batch import BatchRequest, atomic_write, execute_batch, partition_batch_requests
from .compiler import direct_runtime_parameters
from .models import (
    ExecutionPolicy,
    ExecutionTarget,
    LogicalLLMResult,
    PhysicalExecutionPlan,
    PhysicalLLMRequest,
)


def execute_plan(
    plan: PhysicalExecutionPlan,
) -> tuple[ExecutionTarget, list[LogicalLLMResult]]:
    """Execute a compiled plan; return the backend used and ordered results.

    The returned target can differ from ``plan.target`` when a deadline
    fallback demotes a provider batch to direct execution.
    """
    if not plan.requests:
        return plan.target, []
    if plan.target is ExecutionTarget.DIRECT:
        return ExecutionTarget.DIRECT, _execute_direct(plan)
    if (
        plan.target is ExecutionTarget.PROVIDER_BATCH
        and plan.policy.config.get("fallback") is not None
    ):
        return _execute_with_deadline_fallback(plan)
    return plan.target, _execute_deferred(plan)


def _execute_direct(plan: PhysicalExecutionPlan) -> list[LogicalLLMResult]:
    workers = min(plan.policy.concurrency, len(plan.requests))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # executor.map preserves request order even when calls finish out of order.
        return list(executor.map(_complete_one, plan.requests))


def _complete_one(request: PhysicalLLMRequest) -> LogicalLLMResult:
    from litellm import completion

    try:
        completion_kwargs = {**request.body, **request.runtime_parameters}
        response = completion(**completion_kwargs)
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if not isinstance(response, dict):
            raise TypeError("LiteLLM completion did not return a mapping response")
        return LogicalLLMResult(
            request.logical.request_id,
            response,
            status_code=200,
        )
    except Exception as exc:
        return LogicalLLMResult(
            request.logical.request_id,
            None,
            error=exc,
        )


def _execute_deferred(plan: PhysicalExecutionPlan) -> list[LogicalLLMResult]:
    config = dict(plan.policy.config)
    deadline_monotonic = config.pop("_deadline_monotonic", None)
    batch_requests = [
        BatchRequest(request.logical.request_id, request.body, request.url)
        for request in plan.requests
    ]
    default_batch_size = (
        50_000 if plan.target is ExecutionTarget.PROVIDER_BATCH else len(batch_requests)
    )
    max_requests = int(config.get("max_batch_requests", default_batch_size))
    default_max_bytes: int | None = (
        200_000_000 if plan.target is ExecutionTarget.PROVIDER_BATCH else None
    )
    max_bytes_value: Any = config.get("max_batch_bytes", default_max_bytes)
    max_bytes = int(max_bytes_value) if max_bytes_value is not None else None

    results = []
    for chunk in partition_batch_requests(
        batch_requests, max_requests=max_requests, max_bytes=max_bytes
    ):
        chunk_config = dict(config)
        if deadline_monotonic is not None:
            chunk_config["timeout_seconds"] = max(
                0.0, float(deadline_monotonic) - time.monotonic()
            )
        results.extend(execute_batch(chunk, chunk_config))
    return [
        LogicalLLMResult(
            request_id=result.custom_id,
            response=result.body,
            error=result.error,
            status_code=result.status_code,
        )
        for result in results
    ]


def _execute_with_deadline_fallback(
    plan: PhysicalExecutionPlan,
) -> tuple[ExecutionTarget, list[LogicalLLMResult]]:
    state_dir = Path(plan.policy.config.get("work_dir", ".docetl/batches")) / (
        "plans-" + plan.plan_id
    )
    winner_path = state_dir / "winner.json"
    deadline_path = state_dir / "deadline.json"
    fallback_output_path = state_dir / "direct-output.json"
    winner = _read_winner(winner_path)
    if winner is ExecutionTarget.DIRECT:
        return ExecutionTarget.DIRECT, _resume_direct_fallback(
            plan, fallback_output_path
        )

    batch_config = dict(plan.policy.config)
    remaining_seconds = _remaining_deadline_seconds(
        deadline_path, float(batch_config["deadline_seconds"])
    )
    batch_config["_deadline_monotonic"] = time.monotonic() + max(0.0, remaining_seconds)
    try:
        batch_results = _execute_deferred_with_config(plan, batch_config)
    except TimeoutError:
        winner = _claim_winner(winner_path, ExecutionTarget.DIRECT)
        if winner is not ExecutionTarget.DIRECT:
            # Another runner completed and claimed the provider output while this
            # process crossed the deadline. Resume that durable provider job.
            return winner, _execute_deferred(plan)
        return ExecutionTarget.DIRECT, _resume_direct_fallback(
            plan, fallback_output_path
        )

    winner = _claim_winner(winner_path, ExecutionTarget.PROVIDER_BATCH)
    if winner is ExecutionTarget.DIRECT:
        return ExecutionTarget.DIRECT, _resume_direct_fallback(
            plan, fallback_output_path
        )
    return ExecutionTarget.PROVIDER_BATCH, batch_results


def _resume_direct_fallback(
    plan: PhysicalExecutionPlan, output_path: Path
) -> list[LogicalLLMResult]:
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return [
            LogicalLLMResult(
                request_id=row["request_id"],
                response=row["response"],
                error=row.get("error"),
                status_code=row.get("status_code"),
            )
            for row in payload
        ]

    fallback = plan.policy.config.get("fallback")
    fallback_config = fallback if isinstance(fallback, dict) else {}
    fallback_policy = ExecutionPolicy.from_config(
        {
            "mode": "direct",
            "concurrency": fallback_config.get(
                "concurrency", plan.policy.config.get("fallback_concurrency", 1)
            ),
            "direct_kwargs": fallback_config.get(
                "direct_kwargs", plan.policy.config.get("direct_kwargs", {})
            ),
        }
    )
    direct_requests = [
        PhysicalLLMRequest(
            logical=request.logical,
            body={**request.body, "model": request.logical.model},
            runtime_parameters=direct_runtime_parameters(
                request.logical.inference_parameters,
                fallback_policy.config.get("direct_kwargs", {}),
            ),
            url=request.url,
        )
        for request in plan.requests
    ]
    direct_plan = PhysicalExecutionPlan.create(
        target=ExecutionTarget.DIRECT,
        requests=direct_requests,
        policy=fallback_policy,
    )
    results = _execute_direct(direct_plan)
    if all(result.error is None and result.response is not None for result in results):
        atomic_write(
            output_path,
            json.dumps(
                [
                    {
                        "request_id": result.request_id,
                        "response": result.response,
                        "error": None,
                        "status_code": result.status_code,
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    return results


def _read_winner(path: Path) -> ExecutionTarget | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExecutionTarget(payload["winner"])


def _claim_winner(path: Path, target: ExecutionTarget) -> ExecutionTarget:
    payload = json.dumps({"winner": target.value}, sort_keys=True) + "\n"
    if not _publish_once(path, payload):
        winner = _read_winner(path)
        if winner is None:  # pragma: no cover - defensive filesystem race guard
            raise RuntimeError(f"Execution winner file disappeared: {path}")
        return winner
    return target


def _remaining_deadline_seconds(path: Path, deadline_seconds: float) -> float:
    started_at = time.time()
    payload = json.dumps({"started_at": started_at}, sort_keys=True) + "\n"
    if not _publish_once(path, payload):
        persisted = json.loads(path.read_text(encoding="utf-8"))
        started_at = float(persisted["started_at"])
    return deadline_seconds - (time.time() - started_at)


def _publish_once(path: Path, contents: str) -> bool:
    """Atomically publish fully written contents only when *path* is absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _execute_deferred_with_config(
    plan: PhysicalExecutionPlan, config: dict[str, Any]
) -> list[LogicalLLMResult]:
    policy = ExecutionPolicy(
        target=plan.policy.target,
        config=config,
        concurrency=plan.policy.concurrency,
    )
    configured_plan = PhysicalExecutionPlan(
        plan_id=plan.plan_id,
        target=plan.target,
        requests=plan.requests,
        policy=policy,
    )
    return _execute_deferred(configured_plan)
