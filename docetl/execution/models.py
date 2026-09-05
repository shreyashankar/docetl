"""Provider-neutral models for compiling and executing LLM calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Settings that change where or how often a plan runs, but not what it
# computes. They are excluded from plan and job identity so that tuning them
# never creates (and bills) a second provider job for the same requests.
SCHEDULING_ONLY_CONFIG_KEYS = frozenset(
    {
        "concurrency",
        "direct_kwargs",
        "fallback_concurrency",
        "poll_interval_seconds",
        "timeout_seconds",
        "work_dir",
    }
)


class ExecutionTarget(str, Enum):
    """Physical backend selected for a compiled execution plan."""

    DIRECT = "direct"
    PROVIDER_BATCH = "provider_batch"
    VLLM_BATCH = "vllm_batch"


@dataclass(frozen=True)
class LogicalLLMRequest:
    """One model invocation before a physical backend has been selected."""

    request_id: str
    operation_name: str
    model: str
    messages: list[dict[str, Any]]
    output_schema: dict[str, str]
    inference_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Operation-level policy used by the execution compiler."""

    target: ExecutionTarget
    config: dict[str, Any] = field(default_factory=dict)
    concurrency: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExecutionPolicy":
        normalized = dict(config)
        mode = normalized.get("mode", "direct")
        backend = normalized.get("backend", "litellm")
        if mode == "direct":
            target = ExecutionTarget.DIRECT
        elif mode == "batch" and backend == "litellm":
            target = ExecutionTarget.PROVIDER_BATCH
        elif mode == "batch" and backend == "vllm":
            target = ExecutionTarget.VLLM_BATCH
        else:
            raise ValueError(
                "execution must use mode 'direct', or mode 'batch' with "
                "backend 'litellm' or 'vllm'"
            )
        concurrency = int(normalized.get("concurrency", 1))
        if concurrency <= 0:
            raise ValueError("execution.concurrency must be greater than zero")
        fallback = normalized.get("fallback")
        if fallback is not None:
            if target is not ExecutionTarget.PROVIDER_BATCH:
                raise ValueError(
                    "execution.fallback is currently supported only for provider batch"
                )
            if normalized.get("deadline_seconds") is None:
                raise ValueError(
                    "execution.deadline_seconds is required when fallback is configured"
                )
            if fallback != "direct" and not (
                isinstance(fallback, dict)
                and fallback.get("mode", "direct") == "direct"
            ):
                raise ValueError("execution.fallback currently supports only direct")
        if (
            normalized.get("deadline_seconds") is not None
            and float(normalized["deadline_seconds"]) <= 0
        ):
            raise ValueError("execution.deadline_seconds must be greater than zero")
        return cls(target=target, config=normalized, concurrency=concurrency)


@dataclass(frozen=True)
class PhysicalLLMRequest:
    """A logical request lowered to an OpenAI-compatible endpoint body."""

    logical: LogicalLLMRequest
    body: dict[str, Any]
    runtime_parameters: dict[str, Any] = field(default_factory=dict)
    url: str = "/v1/chat/completions"


@dataclass(frozen=True)
class PhysicalExecutionPlan:
    """Executable plan for one group of independent LLM requests."""

    plan_id: str
    target: ExecutionTarget
    requests: tuple[PhysicalLLMRequest, ...]
    policy: ExecutionPolicy

    @classmethod
    def create(
        cls,
        *,
        target: ExecutionTarget,
        requests: list[PhysicalLLMRequest],
        policy: ExecutionPolicy,
    ) -> "PhysicalExecutionPlan":
        identity_config = {
            key: value
            for key, value in policy.config.items()
            if key not in SCHEDULING_ONLY_CONFIG_KEYS
        }
        if isinstance(identity_config.get("fallback"), dict):
            identity_config["fallback"] = {
                key: value
                for key, value in identity_config["fallback"].items()
                if key not in {"concurrency", "direct_kwargs"}
            }
        identity = {
            "target": target.value,
            "config": identity_config,
            "requests": [
                {
                    "request_id": request.logical.request_id,
                    "operation_name": request.logical.operation_name,
                    "body": request.body,
                    "url": request.url,
                }
                for request in requests
            ],
        }
        plan_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        return cls(plan_id, target, tuple(requests), policy)


@dataclass(frozen=True)
class LogicalLLMResult:
    """Provider-neutral result returned to DocETL's existing parser."""

    request_id: str
    response: dict[str, Any] | None
    error: Any | None = None
    status_code: int | None = None
