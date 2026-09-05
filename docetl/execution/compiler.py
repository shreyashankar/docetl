"""Lower logical LLM requests into backend-specific physical plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import (
    ExecutionPolicy,
    ExecutionTarget,
    LogicalLLMRequest,
    PhysicalExecutionPlan,
    PhysicalLLMRequest,
)

RequestBodyBuilder = Callable[[LogicalLLMRequest, ExecutionPolicy], dict[str, Any]]


def compile_execution_plan(
    requests: list[LogicalLLMRequest],
    policy: ExecutionPolicy,
    build_body: RequestBodyBuilder,
) -> PhysicalExecutionPlan:
    """Compile independent logical requests into a physical execution plan."""
    ids = [request.request_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Logical request_id values must be unique within a plan")

    physical_requests = []
    for request in requests:
        body = build_body(request, policy)
        if not isinstance(body, dict):
            raise TypeError("build_body must return a dictionary")
        if "model" not in body or "messages" not in body:
            raise ValueError("A compiled chat request requires model and messages")
        physical_requests.append(
            PhysicalLLMRequest(
                logical=request,
                body=body,
                runtime_parameters=_runtime_parameters(request, policy),
            )
        )

    _validate_target(policy.target, physical_requests)
    return PhysicalExecutionPlan.create(
        target=policy.target,
        requests=physical_requests,
        policy=policy,
    )


def _runtime_parameters(
    request: LogicalLLMRequest, policy: ExecutionPolicy
) -> dict[str, Any]:
    if policy.target is not ExecutionTarget.DIRECT:
        return {}
    return direct_runtime_parameters(
        request.inference_parameters, policy.config.get("direct_kwargs", {})
    )


def direct_runtime_parameters(
    inference_parameters: dict[str, Any], direct_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Keep client-only LiteLLM parameters out of serialized request bodies."""
    client_keys = {
        "api_base",
        "base_url",
        "api_key",
        "custom_llm_provider",
        "allowed_openai_params",
    }
    parameters = {
        key: value for key, value in inference_parameters.items() if key in client_keys
    }
    parameters.update(direct_kwargs)
    return parameters


def _validate_target(
    target: ExecutionTarget, requests: list[PhysicalLLMRequest]
) -> None:
    if target in {ExecutionTarget.PROVIDER_BATCH, ExecutionTarget.VLLM_BATCH}:
        endpoints = {request.url for request in requests}
        if len(endpoints) > 1:
            raise ValueError("A physical batch plan may target only one endpoint")
    if target is ExecutionTarget.VLLM_BATCH:
        models = {request.body["model"] for request in requests}
        if len(models) > 1:
            raise ValueError("A vLLM batch plan requires exactly one model")
