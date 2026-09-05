"""Execution backends for compiled LLM request batches."""

from .batch import (
    BatchExecutionError,
    BatchRequest,
    BatchResult,
    ExecutionError,
    SUPPORTED_LITELLM_BATCH_PROVIDERS,
    execute_batch,
    partition_batch_requests,
)
from .compiler import compile_execution_plan
from .models import (
    ExecutionPolicy,
    ExecutionTarget,
    LogicalLLMRequest,
    LogicalLLMResult,
    PhysicalExecutionPlan,
    PhysicalLLMRequest,
)
from .runtime import execute_plan

__all__ = [
    "BatchExecutionError",
    "BatchRequest",
    "BatchResult",
    "ExecutionError",
    "SUPPORTED_LITELLM_BATCH_PROVIDERS",
    "execute_batch",
    "partition_batch_requests",
    "compile_execution_plan",
    "execute_plan",
    "ExecutionPolicy",
    "ExecutionTarget",
    "LogicalLLMRequest",
    "LogicalLLMResult",
    "PhysicalExecutionPlan",
    "PhysicalLLMRequest",
]
