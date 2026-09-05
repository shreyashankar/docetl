import ast
import asyncio
import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from enum import Enum
from typing import Any

from litellm import (
    APIConnectionError,
    ModelResponse,
    RateLimitError,
    ServiceUnavailableError,
    completion,
    embedding,
    get_llm_provider,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from rich import print as rprint
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from docetl.execution import (
    SUPPORTED_LITELLM_BATCH_PROVIDERS,
    ExecutionPolicy,
    ExecutionTarget,
    LogicalLLMRequest,
    compile_execution_plan,
    execute_plan,
)
from docetl.utils import completion_cost

from .cache import cache, cache_key, freezeargs
from .llm import (
    InvalidOutputError,
    LLMResult,
    approx_count_tokens,
    timeout,
    truncate_messages,
)
from .openai_agents_runner import run_openai_agent
from .validation import (
    convert_dict_schema_to_list_schema,
    convert_val,
    get_user_input_for_schema,
    safe_eval,
    strict_render,
)

BASIC_MODELS = ["gpt-4o-mini", "gpt-4o"]


def _raise_with_provider_hint(model: str, exc: Exception) -> None:
    """Re-raise with a hint to prefix the model name if it looks bare."""
    if model not in BASIC_MODELS and "/" not in model:
        raise ValueError(
            "Note: You may also need to prefix your model name with the "
            "provider, e.g. 'openai/gpt-4o-mini' or "
            "'gemini/gemini-1.5-flash' to conform to LiteLLM API "
            f"standards. Original error: {exc}"
        ) from exc


class OutputMode(Enum):
    """Enumeration of output modes for LLM calls."""

    TOOLS = "tools"
    STRUCTURED_OUTPUT = "structured_output"


def is_deepseek_r1(model: str) -> bool:
    model = model.lower()
    return "deepseek-r1" in model or "deepseek-reasoner" in model


def is_snowflake(model: str) -> bool:
    model = model.lower()
    return "snowflake" in model


class APIWrapper(object):
    def __init__(self, runner):
        self.runner = runner
        self.default_lm_api_base = runner.config.get("default_lm_api_base", None)
        self.default_embedding_api_base = runner.config.get(
            "default_embedding_api_base", None
        )
        # Use routers as instance variables (for fallback models)
        self.router = getattr(runner, "router", None)
        self.embedding_router = getattr(runner, "embedding_router", None)
        # Store fallback configs and router cache from runner
        self.fallback_models_config = getattr(runner, "fallback_models_config", [])
        self.runner_router_cache = getattr(runner, "_router_cache", {})

    def _get_router_with_operation_model(self, operation_model: str) -> Any:
        """
        Get Router completion function with operation's model first, then fallbacks.
        Uses cached Router from runner if available.
        """
        # Return cached Router if available
        if operation_model in self.runner_router_cache:
            return self.runner_router_cache[operation_model].completion

        from litellm import Router

        # Build model list: operation model first, then fallbacks
        model_list = [
            {
                "model_name": operation_model,
                "litellm_params": {
                    "model": operation_model,
                    **(
                        {"api_base": self.default_lm_api_base}
                        if self.default_lm_api_base
                        else {}
                    ),
                },
            }
        ]
        model_names = [operation_model]

        # Add fallback models, skipping duplicates
        seen = {operation_model}
        for cfg in self.fallback_models_config:
            name = (
                cfg.get("model_name")
                if isinstance(cfg, dict)
                else (cfg if isinstance(cfg, str) else None)
            )
            if not name or name in seen:
                continue
            seen.add(name)
            params = (
                cfg.get("litellm_params", {}).copy() if isinstance(cfg, dict) else {}
            )
            params["model"] = name
            if self.default_lm_api_base and "api_base" not in params:
                params["api_base"] = self.default_lm_api_base
            model_list.append({"model_name": name, "litellm_params": params})
            model_names.append(name)

        # Build fallbacks list: operation model falls back to all fallback models
        router_kwargs = {"model_list": model_list}
        if len(model_names) > 1:
            # fallbacks should be a list of dicts: [{"model1": ["fallback1", "fallback2"]}]
            router_kwargs["fallbacks"] = [{operation_model: model_names[1:]}]

        router = Router(**router_kwargs)
        self.runner_router_cache[operation_model] = router
        return router.completion

    def _track_token_usage(self, model: str, response) -> None:
        """Extract token usage from a LiteLLM response and accumulate on the runner."""
        if response and hasattr(response, "usage") and response.usage:
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.runner.total_token_usage[model]["prompt_tokens"] += prompt_tokens
            self.runner.total_token_usage[model][
                "completion_tokens"
            ] += completion_tokens

            # Track cached/cache-creation tokens if available
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0
            # Anthropic reports cache reads separately
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cached = max(cached, cache_read)
            if cached:
                self.runner.total_token_usage[model]["cached_tokens"] = (
                    self.runner.total_token_usage[model].get("cached_tokens", 0)
                    + cached
                )
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            if cache_creation:
                self.runner.total_token_usage[model]["cache_creation_tokens"] = (
                    self.runner.total_token_usage[model].get("cache_creation_tokens", 0)
                    + cache_creation
                )

    @freezeargs
    def gen_embedding(self, model: str, input: list[str]) -> list[float]:
        """
        A cached wrapper around litellm.embedding function.

        This function uses LRU (Least Recently Used) cache to store and retrieve
        embeddings for given inputs. It can significantly speed up repeated calls
        with the same model and input.

        Args:
            model (str): The name of the embedding model to use.
            input (list[str]): The input text to generate an embedding for.

        Returns:
            list[float]: The embedding vector as a list of floats.

        Note:
            The cache size is set to 1000. Adjust this value based on your memory
            constraints and usage patterns.
        """
        # Create a unique key for the cache
        key = hashlib.md5(f"{model}_{input}".encode()).hexdigest()
        input = json.loads(input)

        # If the model starts with "gpt" and there is no openai key, prefix the model with "azure"
        if (
            model.startswith("text-embedding")
            and not os.environ.get("OPENAI_API_KEY")
            and self.runner.config.get("from_docwrangler", False)
        ):
            model = "azure/" + model

        with cache as c:
            # Try to get the result from cache
            result = c.get(key)
            if result is None:
                # If not in cache, compute the embedding
                if not isinstance(input[0], str):
                    input = [json.dumps(item) for item in input]

                input = [item if item else "None" for item in input]

                # FIXME: Should we use a different limit for embedding?
                self.runner.blocking_acquire("embedding_call", weight=1)
                if self.runner.is_cancelled:
                    raise asyncio.CancelledError("Operation was cancelled")

                extra_kwargs = {}
                if self.default_embedding_api_base:
                    extra_kwargs["api_base"] = self.default_embedding_api_base

                # Use embedding router if available (for fallback models)
                embedding_fn = (
                    self.embedding_router.embedding
                    if self.embedding_router
                    else embedding
                )
                result = embedding_fn(model=model, input=input, **extra_kwargs)
                self._track_token_usage(model, result)
                # Cache the result
                c.set(key, result)

        return result

    def call_llm_batch(
        self,
        model: str,
        op_type: str,
        messages: list[dict[str, str]],
        output_schema: dict[str, str],
        verbose: bool = False,
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
        bypass_cache: bool = False,
        litellm_completion_kwargs: dict[str, Any] = {},
        op_config: dict[str, Any] = {},
    ) -> LLMResult:
        # Turn the output schema into a list of schemas
        output_schema = convert_dict_schema_to_list_schema(output_schema)

        # Invoke the LLM call
        return self.call_llm(
            model,
            op_type,
            messages,
            output_schema,
            verbose=verbose,
            timeout_seconds=timeout_seconds,
            max_retries_per_timeout=max_retries_per_timeout,
            bypass_cache=bypass_cache,
            litellm_completion_kwargs=litellm_completion_kwargs,
            op_config=op_config,
        )

    def _build_completion_request(
        self,
        model: str,
        op_type: str,
        messages: list[dict[str, Any]],
        output_schema: dict[str, str],
        litellm_completion_kwargs: dict[str, Any],
        op_config: dict[str, Any],
        execution_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one OpenAI-compatible request body without executing it.

        Batch backends deliberately support the non-agentic, row-local call
        shape first. Validation, gleaning, and agent loops need more than one
        dependent round and are rejected by ``MapOperation`` before this point.
        """
        props = {key: convert_val(value) for key, value in output_schema.items()}
        output_mode = op_config.get("output", {}).get("mode", OutputMode.TOOLS.value)
        use_structured_output = output_mode == OutputMode.STRUCTURED_OUTPUT.value
        use_tools = not (
            len(props) == 1
            and next(iter(props.values())).get("type") == "string"
            and ("sagemaker" in model or is_deepseek_r1(model))
        )

        persona = self.runner.config.get("system_prompt", {}).get(
            "persona", "a helpful assistant"
        )
        dataset_description = self.runner.config.get("system_prompt", {}).get(
            "dataset_description", "a collection of unstructured documents"
        )
        operation_shape = (
            "many inputs:one output" if op_type == "reduce" else "one input:one output"
        )
        base_prompt = (
            f"You are a {persona}, helping the user make sense of their data. "
            f"The dataset description is: {dataset_description}. You will be "
            f"performing a {op_type} operation ({operation_shape}). You will "
            "perform the specified task on the provided data, as precisely and "
            "exhaustively (i.e., high recall) as possible."
        )

        body: dict[str, Any] = dict(litellm_completion_kwargs)
        # These configure a LiteLLM client, not an endpoint request body.
        for client_key in (
            "api_base",
            "base_url",
            "api_key",
            "custom_llm_provider",
            "allowed_openai_params",
        ):
            body.pop(client_key, None)
        if "n" in op_config.get("output", {}):
            body["n"] = op_config["output"]["n"]
        if "gpt-5" in model:
            body.pop("temperature", None)

        if execution_config.get("mode", "batch") == "direct":
            body["model"] = model
        elif execution_config.get("backend", "litellm") == "litellm":
            normalized_model, provider, _, _ = get_llm_provider(model)
            configured_provider = execution_config.get("provider", "openai")
            if provider != configured_provider:
                raise ValueError(
                    f"Batch backend provider {configured_provider!r} cannot execute "
                    f"model {model!r} (LiteLLM provider {provider!r})"
                )
            body["model"] = normalized_model
        else:
            body["model"] = execution_config.get("model", model)

        system_prompt = base_prompt
        if use_structured_output:
            system_prompt += (
                " Respond with a JSON object that follows the required schema."
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": {
                        "type": "object",
                        "properties": props,
                        "required": list(props),
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        elif use_tools:
            system_prompt += (
                " The result should be a structured output that you will send "
                "back to the user, with the `send_output` function. Do not "
                "influence your answers too much based on the `send_output` "
                "function parameter names; just use them to send the result "
                "back to the user."
            )
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": props,
                "required": list(props),
            }
            if "gemini" not in model and "claude" not in model:
                parameters["additionalProperties"] = False
            tool: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": "send_output",
                    "description": "Send output back to the user",
                    "parameters": parameters,
                },
            }
            if "claude" not in model:
                tool["additionalProperties"] = False
                tool["strict"] = True
            body["tools"] = [tool]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": "send_output"},
            }
        else:
            system_prompt = base_prompt

        body["messages"] = truncate_messages(
            [{"role": "system", "content": system_prompt}] + messages,
            model,
        )
        return body

    def call_llm_compiled(
        self,
        calls: list[dict[str, Any]],
        execution_config: dict[str, Any],
    ) -> list[LLMResult]:
        """Compile a list of independent LLM calls and execute them as one plan."""
        execution_config = dict(execution_config)
        policy = ExecutionPolicy.from_config(execution_config)
        if policy.target is ExecutionTarget.PROVIDER_BATCH:
            if "provider" not in execution_config and calls:
                providers = {get_llm_provider(call["model"])[1] for call in calls}
                if len(providers) != 1:
                    raise ValueError(
                        "A provider batch requires models from exactly one "
                        "LiteLLM provider"
                    )
                execution_config["provider"] = providers.pop()
                policy = replace(policy, config=execution_config)
            if (
                execution_config.get("provider", "openai")
                not in SUPPORTED_LITELLM_BATCH_PROVIDERS
            ):
                raise ValueError(
                    "LiteLLM batch execution currently supports provider values "
                    + ", ".join(sorted(SUPPORTED_LITELLM_BATCH_PROVIDERS))
                )
        results: list[LLMResult | None] = [None] * len(calls)
        logical_requests: list[LogicalLLMRequest] = []
        pending: list[tuple[int, str, str]] = []

        with cache as c:
            for index, call in enumerate(calls):
                # Execution policy changes where and when an identical logical
                # call runs; it must not create a second semantic cache entry.
                cache_op_config = {
                    key: value
                    for key, value in call["op_config"].items()
                    if key != "execution"
                }
                key = cache_key(
                    call["model"],
                    call["op_type"],
                    call["messages"],
                    call["output_schema"],
                    None,
                    self.runner.config.get("system_prompt", {}),
                    cache_op_config,
                )
                cached = c.get(key)
                if cached is not None and not call.get("bypass_cache", False):
                    results[index] = LLMResult(
                        response=cached, total_cost=0.0, validated=True
                    )
                    continue
                custom_id = f"request-{index}"
                logical_requests.append(
                    LogicalLLMRequest(
                        request_id=custom_id,
                        operation_name=call["op_config"].get("name", call["op_type"]),
                        model=call["model"],
                        messages=call["messages"],
                        output_schema=call["output_schema"],
                        inference_parameters=call.get("litellm_completion_kwargs", {}),
                        metadata={
                            "op_type": call["op_type"],
                            "op_config": call["op_config"],
                        },
                    )
                )
                pending.append((index, key, call["model"]))

        if logical_requests:
            plan = compile_execution_plan(
                logical_requests,
                policy,
                lambda request, selected_policy: self._build_completion_request(
                    request.model,
                    request.metadata["op_type"],
                    request.messages,
                    request.output_schema,
                    request.inference_parameters,
                    request.metadata["op_config"],
                    selected_policy.config,
                ),
            )
            effective_target, physical_results = execute_plan(plan)
            # completion_cost prices synchronous calls. Apply the batch discount
            # only where it is published and uniform (OpenAI's 50%); self-hosted
            # vLLM has no per-token price at all.
            if "cost_multiplier" in execution_config:
                multiplier = float(execution_config["cost_multiplier"])
            elif effective_target is ExecutionTarget.VLLM_BATCH:
                multiplier = 0.0
            elif (
                effective_target is ExecutionTarget.PROVIDER_BATCH
                and execution_config.get("provider", "openai") == "openai"
            ):
                multiplier = 0.5
            else:
                multiplier = 1.0
            with cache as c:
                for (index, key, model), physical_result in zip(
                    pending, physical_results
                ):
                    if (
                        physical_result.error is not None
                        or physical_result.response is None
                        or (
                            physical_result.status_code is not None
                            and not 200 <= physical_result.status_code < 300
                        )
                    ):
                        results[index] = LLMResult(
                            response=None, total_cost=0.0, validated=False
                        )
                        continue
                    response = ModelResponse(**physical_result.response)
                    estimated_cost = completion_cost(response) * multiplier
                    setattr(response, "_completion_cost", estimated_cost)
                    self._track_token_usage(model, response)
                    c.set(key, response)
                    results[index] = LLMResult(
                        response=response,
                        total_cost=estimated_cost,
                        validated=True,
                    )

        return [
            (
                result
                if result is not None
                else LLMResult(response=None, total_cost=0.0, validated=False)
            )
            for result in results
        ]

    def _cached_call_llm(
        self,
        cache_key: str,
        model: str,
        op_type: str,
        messages: list[dict[str, str]],
        output_schema: dict[str, str],
        tools: str | None = None,
        scratchpad: str | None = None,
        validation_config: dict[str, Any] | None = None,
        gleaning_config: dict[str, Any] | None = None,
        verbose: bool = False,
        bypass_cache: bool = False,
        initial_result: Any | None = None,
        litellm_completion_kwargs: dict[str, Any] = {},
        op_config: dict[str, Any] = {},
        agent_config: Any | None = None,
    ) -> LLMResult:
        """
        Cached version of the call_llm function.

        This function serves as a cached wrapper around _call_llm_with_cache. It uses
        the @freezeargs decorator to ensure immutable arguments and @functools.lru_cache
        for caching results.

        Args:
            cache_key (str): A unique key for caching.
            model (str): The model name.
            op_type (str): The operation type.
            messages (list[dict[str, str]]): The messages to send to the LLM.
            output_schema (dict[str, str]): The output schema dictionary.
            tools (str | None): The tools to pass to the LLM.
            scratchpad (str | None): The scratchpad to use for the operation.
            validation_config (dict[str, Any] | None): The validation configuration.
            gleaning_config (dict[str, Any] | None): The gleaning configuration.
            verbose (bool): Whether to print verbose output.
            bypass_cache (bool): Whether to bypass the cache.
            initial_result (Any | None): The initial result to use for the operation, if exists.
            op_config (dict[str, Any]): The operation configuration.
        Returns:
            LLMResult: The response from _call_llm_with_cache.
        """
        # Determine output mode using central enum
        output_mode_str = op_config.get("output", {}).get(
            "mode", OutputMode.TOOLS.value
        )
        use_structured_output = output_mode_str == OutputMode.STRUCTURED_OUTPUT.value
        if agent_config is not None and use_structured_output:
            raise ValueError(
                "agent cannot be combined with output.mode='structured_output'; "
                "agentic operations use the OpenAI Agents SDK final output schema."
            )
        if agent_config is not None and gleaning_config:
            raise ValueError("agent cannot be combined with gleaning.")
        if (
            model.startswith("gpt")
            and not os.environ.get("OPENAI_API_KEY")
            and self.runner.config.get("from_docwrangler", False)
        ):
            model = "azure/" + model

        # Pop off temperature if it's gpt-5 in the model name
        if "gpt-5" in model:
            litellm_completion_kwargs.pop("temperature", None)

        total_cost = 0.0
        validated = False
        with cache as c:
            response = c.get(cache_key)
            if response is not None and not bypass_cache:
                validated = True
            else:
                if not initial_result:
                    response = self._call_llm_with_cache(
                        model,
                        op_type,
                        messages,
                        output_schema,
                        tools,
                        scratchpad,
                        litellm_completion_kwargs,
                        op_config=op_config,
                        use_structured_output=use_structured_output,
                        agent_config=agent_config,
                    )
                    if isinstance(response, ModelResponse):
                        total_cost += completion_cost(response)
                        self._track_token_usage(model, response)
                    else:
                        total_cost += getattr(response, "_docetl_total_cost", 0.0)
                else:
                    response = initial_result

                if gleaning_config:
                    # Retry gleaning prompt + regular LLM
                    num_gleaning_rounds = gleaning_config.get("num_rounds", 2)

                    parsed_output = (
                        self.parse_llm_response(
                            response,
                            output_schema,
                            json.loads(tools) if tools else None,
                            False,
                            use_structured_output,
                        )[0]
                        if isinstance(response, ModelResponse)
                        else response
                    )

                    validator_messages = (
                        [
                            {
                                "role": "system",
                                "content": f"You are a helpful assistant, intelligently processing data. This is a {op_type} operation.",
                            }
                        ]
                        + messages
                        + [{"role": "assistant", "content": json.dumps(parsed_output)}]
                    )

                    for rnd in range(num_gleaning_rounds):
                        # Break early if gleaning condition is not met
                        if not self.should_glean(gleaning_config, parsed_output):
                            break
                        # Prepare validator prompt
                        validator_prompt = strict_render(
                            gleaning_config["validation_prompt"],
                            {"output": parsed_output},
                        )
                        self.runner.blocking_acquire("llm_call", weight=1)
                        # Approx the number of tokens in the messages
                        approx_num_tokens = approx_count_tokens(
                            validator_messages
                            + [{"role": "user", "content": validator_prompt}]
                        )
                        self.runner.blocking_acquire(
                            "llm_tokens", weight=approx_num_tokens
                        )

                        # Pop off temperature if it's gpt-5 in the model name
                        gleaning_model = gleaning_config.get("model", model)
                        validator_kwargs = litellm_completion_kwargs.copy()
                        if "gpt-5" in gleaning_model:
                            validator_kwargs.pop("temperature", None)

                        # Get params for should refine
                        should_refine_params = {
                            "type": "object",
                            "properties": {
                                "should_refine": {"type": "boolean"},
                                "improvements": {"type": "string"},
                            },
                            "required": ["should_refine", "improvements"],
                        }
                        if "gemini" not in gleaning_model:
                            should_refine_params["additionalProperties"] = False

                        # Add extra kwargs
                        extra_kwargs = {}
                        if self.default_lm_api_base:
                            extra_kwargs["api_base"] = self.default_lm_api_base
                        if is_snowflake(gleaning_model):
                            extra_kwargs["allowed_openai_params"] = [
                                "tools",
                                "tool_choice",
                            ]

                        # Use router if available (for fallback models), otherwise use direct completion
                        # When using router, ensure gleaning model is tried first, then fallback models
                        if self.router and self.fallback_models_config:
                            completion_fn = self._get_router_with_operation_model(
                                gleaning_model
                            )
                        else:
                            completion_fn = completion

                        validator_response = completion_fn(
                            model=gleaning_model,
                            messages=truncate_messages(
                                validator_messages
                                + [{"role": "user", "content": validator_prompt}],
                                gleaning_model,
                            ),
                            tools=[
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "should_refine_answer",
                                        "description": "Determine if the output should be refined based on the validation feedback",
                                        "strict": True,
                                        "parameters": should_refine_params,
                                        "additionalProperties": False,
                                    },
                                }
                            ],
                            tool_choice="required",
                            **validator_kwargs,
                            **extra_kwargs,
                        )
                        total_cost += completion_cost(validator_response)
                        self._track_token_usage(gleaning_model, validator_response)

                        # Parse the validator response
                        suggestion = json.loads(
                            validator_response.choices[0]
                            .message.tool_calls[0]
                            .function.arguments
                        )
                        if not suggestion["should_refine"]:
                            break

                        if verbose:
                            self.runner.console.log(
                                f"Validator improvements (gleaning round {rnd + 1}): {suggestion['improvements']}"
                            )

                        # Prompt for improvement
                        improvement_prompt = f"""Based on the validation feedback:

                        ```
                        {suggestion["improvements"]}
                        ```

                        Please improve your previous response. Ensure that the output adheres to the required schema and addresses any issues raised in the validation."""
                        messages.append({"role": "user", "content": improvement_prompt})

                        # Call LLM again

                        response = self._call_llm_with_cache(
                            model,
                            op_type,
                            messages,
                            output_schema,
                            tools,
                            scratchpad,
                            litellm_completion_kwargs,
                            op_config=op_config,
                            use_structured_output=use_structured_output,
                            agent_config=agent_config,
                        )
                        parsed_output = self.parse_llm_response(
                            response, output_schema, tools, False, use_structured_output
                        )[0]
                        validator_messages[-1] = {
                            "role": "assistant",
                            "content": json.dumps(parsed_output),
                        }

                        total_cost += completion_cost(response)
                        self._track_token_usage(model, response)

                    validated = True

                # If there's validation, handle it here
                elif validation_config:
                    num_tries = validation_config.get("num_retries", 2) + 1
                    validation_fn = validation_config.get("validation_fn")
                    val_rule = validation_config.get("val_rule")

                    # Try validation
                    i = 0
                    validation_result = False
                    while not validation_result and i < num_tries:
                        parsed_output, validation_result = validation_fn(response)
                        if validation_result:
                            validated = True
                            break

                        # Append the validation result to messages
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(parsed_output),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Your output {parsed_output} either failed to match my specified schema or failed one or more of the following validation rules: {str(val_rule)}\n\nPlease try again.",
                            }
                        )
                        self.runner.console.log(
                            f"[bold red]Validation failed:[/bold red] {val_rule}\n"
                            f"\t[yellow]Output:[/yellow] {parsed_output}\n"
                            f"\t({i + 1}/{num_tries})"
                        )
                        i += 1

                        response = self._call_llm_with_cache(
                            model,
                            op_type,
                            messages,
                            output_schema,
                            tools,
                            scratchpad,
                            litellm_completion_kwargs,
                            op_config=op_config,
                            use_structured_output=use_structured_output,
                            agent_config=agent_config,
                        )
                        if isinstance(response, ModelResponse):
                            total_cost += completion_cost(response)
                            self._track_token_usage(model, response)
                        else:
                            total_cost += getattr(response, "_docetl_total_cost", 0.0)

                else:
                    # No validation, so we assume the result is valid
                    validated = True

                # Only set the cache if the result tool calls or output is not empty
                if validated:
                    c.set(cache_key, response)

        return LLMResult(response=response, total_cost=total_cost, validated=validated)

    def call_llm(
        self,
        model: str,
        op_type: str,
        messages: list[dict[str, str]],
        output_schema: dict[str, str],
        tools: list[dict[str, str]] | None = None,
        scratchpad: str | None = None,
        timeout_seconds: int = 120,
        max_retries_per_timeout: int = 2,
        validation_config: dict[str, Any] | None = None,
        gleaning_config: dict[str, Any] | None = None,
        verbose: bool = False,
        bypass_cache: bool = False,
        initial_result: Any | None = None,
        litellm_completion_kwargs: dict[str, Any] = {},
        op_config: dict[str, Any] = {},
        agent_config: Any | None = None,
    ) -> LLMResult:
        """
        Wrapper function that uses caching for LLM calls.

        This function generates a cache key and calls the cached version of call_llm.
        It retries the call if it times out after 60 seconds.

        Args:
            model (str): The model name.
            op_type (str): The operation type.
            messages (list[dict[str, str]]): The messages to send to the LLM.
            output_schema (dict[str, str]): The output schema dictionary.
            tools (list[dict[str, str]] | None): The tools to pass to the LLM.
            scratchpad (str | None): The scratchpad to use for the operation.
            timeout_seconds (int): The timeout for the LLM call.
            max_retries_per_timeout (int): The maximum number of retries per timeout.
            bypass_cache (bool): Whether to bypass the cache.
            initial_result (Any | None): The initial result to use for the operation, if exists.
            op_config (dict[str, Any]): Operation configuration, may contain output.mode.
        Returns:
            LLMResult: The result from the cached LLM call.

        Raises:
            TimeoutError: If the call times out after retrying.
        """
        # Determine output mode using central enum
        output_mode_str = op_config.get("output", {}).get(
            "mode", OutputMode.TOOLS.value
        )
        if output_mode_str not in [mode.value for mode in OutputMode]:
            raise ValueError(
                f"Invalid output mode '{output_mode_str}'. Must be 'tools' or 'structured_output'."
            )
        if agent_config is not None and output_mode_str == OutputMode.STRUCTURED_OUTPUT.value:
            raise ValueError(
                "agent cannot be combined with output.mode='structured_output'; "
                "agentic operations use the OpenAI Agents SDK final output schema."
            )
        if agent_config is not None and gleaning_config:
            raise ValueError("agent cannot be combined with gleaning.")
        if agent_config is not None and not getattr(agent_config, "cache", False):
            bypass_cache = True

        key = cache_key(
            model,
            op_type,
            messages,
            output_schema,
            scratchpad,
            self.runner.config.get("system_prompt", {}),
            op_config,
        )

        max_retries = max_retries_per_timeout
        attempt = 0
        rate_limited_attempt = 0
        while attempt <= max_retries:
            try:
                output = timeout(timeout_seconds)(self._cached_call_llm)(
                    key,
                    model,
                    op_type,
                    messages,
                    output_schema,
                    json.dumps(tools) if tools else None,
                    scratchpad,
                    validation_config=validation_config,
                    gleaning_config=gleaning_config,
                    verbose=verbose,
                    bypass_cache=bypass_cache,
                    initial_result=initial_result,
                    litellm_completion_kwargs=litellm_completion_kwargs,
                    op_config=op_config,
                    agent_config=agent_config,
                )
                # Log input and output if verbose
                if verbose:
                    # Truncate messages to 500 chars
                    messages_str = str(messages)
                    truncated_messages = (
                        messages_str[:500] + "..."
                        if len(messages_str) > 500
                        else messages_str
                    )

                    # Log with nice formatting
                    self.runner.console.print(
                        Panel(
                            Group(
                                Text("Input:", style="bold cyan"),
                                Text(truncated_messages),
                                Text("\nOutput:", style="bold cyan"),
                                Text(str(output)),
                            ),
                            title="[bold green]LLM Call Details[/bold green]",
                            border_style="green",
                        )
                    )

                return output
            except RateLimitError:
                # TODO: this is a really hacky way to handle rate limits
                # we should implement a more robust retry mechanism
                backoff_time = 4 * (2**rate_limited_attempt)  # Exponential backoff
                max_backoff = 120  # Maximum backoff time of 60 seconds
                sleep_time = min(backoff_time, max_backoff)
                self.runner.console.log(
                    f"[yellow]Rate limit hit. Retrying in {sleep_time:.2f} seconds...[/yellow]"
                )
                time.sleep(sleep_time)
                rate_limited_attempt += 1
            except APIConnectionError as e:
                self.runner.console.log(
                    f"[bold red]API connection error. Retrying...[/bold red] {e}"
                )
                time.sleep(1)
            except ServiceUnavailableError:
                self.runner.console.log(
                    "[bold red]Service unavailable. Retrying...[/bold red]"
                )
                time.sleep(1)
            except TimeoutError:
                if attempt == max_retries:
                    self.runner.console.log(
                        f"[bold red]LLM call timed out after {max_retries + 1} attempts[/bold red]"
                    )
                    # TODO: HITL
                    return LLMResult(response=None, total_cost=0.0, validated=False)
                attempt += 1

    def _call_llm_with_cache(
        self,
        model: str,
        op_type: str,
        messages: list[dict[str, str]],
        output_schema: dict[str, str],
        tools: str | None = None,
        scratchpad: str | None = None,
        litellm_completion_kwargs: dict[str, Any] = {},
        op_config: dict[str, Any] = {},
        use_structured_output: bool = False,
        agent_config: Any | None = None,
    ) -> Any:
        """
        Make an LLM call with caching.

        This function prepares the necessary parameters and makes a call to the LLM
        using the provided model, operation type, prompt, and output schema.

        Args:
            model (str): The model name.
            op_type (str): The operation type.
            messages (list[dict[str, str]]): The messages to send to the LLM.
            output_schema (dict[str, str]): The output schema dictionary.
            tools (str | None): The tools to pass to the LLM.
            scratchpad (str | None): The scratchpad to use for the operation.
        Returns:
            str: The response from the LLM.
        """
        props = {key: convert_val(value) for key, value in output_schema.items()}
        use_tools = True

        if (
            len(props) == 1
            and list(props.values())[0].get("type") == "string"
            and scratchpad is None
            and ("sagemaker" in model or is_deepseek_r1(model))
        ):
            use_tools = False

        # For structured output mode, override use_tools
        if use_structured_output:
            use_tools = False

        if tools is None and use_tools and not use_structured_output:
            if scratchpad is not None:
                props["updated_scratchpad"] = {"type": "string"}

            parameters = {"type": "object", "properties": props}
            parameters["required"] = list(props.keys())

            # TODO: this is a hack to get around the fact that gemini doesn't support additionalProperties
            if "gemini" not in model and "claude" not in model:
                parameters["additionalProperties"] = False

            if is_snowflake(model):
                tools = [
                    {
                        "tool_spec": {
                            "type": "generic",
                            "name": "send_output",
                            "description": "Send output back to the user",
                            "input_schema": parameters,
                        }
                    }
                ]
            else:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "send_output",
                            "description": "Send output back to the user",
                            "parameters": parameters,
                        },
                    }
                ]
            if "claude" not in model:
                tools[0]["additionalProperties"] = False
                tools[0]["strict"] = True

            tool_choice = {"type": "function", "function": {"name": "send_output"}}

        elif tools is not None:
            tools = json.loads(tools)
            tool_choice = (
                "required"
                if any(tool.get("required", False) for tool in tools)
                else "auto"
            )
            tools = [
                {"type": "function", "function": tool["function"]} for tool in tools
            ]

        else:
            tools = None
            tool_choice = None

        # Prepare structured output schema if using structured output mode
        response_format = None
        if use_structured_output:
            if scratchpad is not None:
                props["updated_scratchpad"] = {"type": "string"}

            schema = {
                "type": "object",
                "properties": props,
                "required": list(props.keys()),
                "additionalProperties": False,
            }

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                },
            }

        persona = self.runner.config.get("system_prompt", {}).get(
            "persona", "a helpful assistant"
        )
        dataset_description = self.runner.config.get("system_prompt", {}).get(
            "dataset_description", "a collection of unstructured documents"
        )
        parethetical_op_instructions = (
            "many inputs:one output" if op_type == "reduce" else "one input:one output"
        )

        # Different system prompts based on model type
        base_prompt = f"You are a {persona}, helping the user make sense of their data. The dataset description is: {dataset_description}. You will be performing a {op_type} operation ({parethetical_op_instructions}). You will perform the specified task on the provided data, as precisely and exhaustively (i.e., high recall) as possible."

        if use_structured_output:
            system_prompt = (
                base_prompt
                + " Respond with a JSON object that follows the required schema."
            )
        elif "sagemaker" in model or is_deepseek_r1(model):
            system_prompt = base_prompt
        else:
            system_prompt = (
                base_prompt
                + " The result should be a structured output that you will send back to the user, with the `send_output` function. Do not influence your answers too much based on the `send_output` function parameter names; just use them to send the result back to the user."
            )

        if scratchpad:
            system_prompt += f"""

You are incrementally processing data across multiple batches. You will see:
1. The current batch of data to process
2. The intermediate output so far (what you returned last time)
3. A scratchpad for tracking additional state: {scratchpad}

IMPORTANT: Only use the scratchpad if your task specifically requires tracking items that appear multiple times across batches. If you only need to track distinct/unique items, leave the scratchpad empty and set updated_scratchpad to null.

The intermediate output contains the result that directly answers the user's task, for **all** the data processed so far, including the current batch. You must return this via the send_output function.

Example task that NEEDS scratchpad - counting words that appear >2 times:
- Call send_output with: {{"frequent_words": ["the", "and"]}} # Words seen 3+ times - this is your actual result
- Set updated_scratchpad to: {{"pending": {{"cat": 2, "dog": 1}}}} # Must track words seen 1-2 times

Example task that does NOT need scratchpad - collecting unique locations:
- Call send_output with: {{"locations": ["New York", "Paris"]}} # Just the unique items
- Set updated_scratchpad to: null # No need to track counts since we only want distinct items

As you process each batch:
1. Use both the previous output and scratchpad (if needed) to inform your processing
2. Call send_output with your result that combines the current batch with previous output
3. Set updated_scratchpad only if you need to track counts/frequencies between batches

If you use the scratchpad, keep it concise (~500 chars) and easily parsable using:
- Key-value pairs
- JSON-like format
- Simple counters/tallies

Your main result must be sent via send_output. The updated_scratchpad is only for tracking state between batches, and should be null unless you specifically need to track frequencies."""

        # Truncate messages if they exceed the model's context length
        messages_with_system_prompt = truncate_messages(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ]
            + messages,
            model,
        )

        self.runner.blocking_acquire("llm_call", weight=1)

        # Approx the number of tokens in the messages
        approx_num_tokens = approx_count_tokens(messages)
        self.runner.blocking_acquire("llm_tokens", weight=approx_num_tokens)
        if self.runner.is_cancelled:
            raise asyncio.CancelledError("Operation was cancelled")

        extra_litellm_kwargs = {}
        extra_litellm_kwargs.update(litellm_completion_kwargs)
        if "n" in op_config.get("output", {}).keys():
            extra_litellm_kwargs["n"] = op_config["output"]["n"]
        if is_snowflake(model):
            extra_litellm_kwargs["allowed_openai_params"] = ["tools", "tool_choice"]
        if self.default_lm_api_base:
            extra_litellm_kwargs["api_base"] = self.default_lm_api_base

        if agent_config is not None:
            return run_openai_agent(
                runner=self.runner,
                model=model,
                op_type=op_type,
                messages=messages,
                output_schema=output_schema,
                agent_config=agent_config,
                system_prompt=system_prompt,
                scratchpad=scratchpad,
                litellm_completion_kwargs=extra_litellm_kwargs,
            )

        # Use router if available (for fallback models), otherwise use direct completion
        # When using router, ensure operation's model is tried first, then fallback models
        if self.router and self.fallback_models_config:
            # Build model list with operation's model first, then fallback models
            completion_fn = self._get_router_with_operation_model(model)
        else:
            completion_fn = completion

        # Pop off temperature if it's gpt-5 in the model name
        if "gpt-5" in model:
            extra_litellm_kwargs.pop("temperature", None)

        if use_structured_output:
            try:
                response = completion_fn(
                    model=model,
                    messages=messages_with_system_prompt,
                    response_format=response_format,
                    **extra_litellm_kwargs,
                )
            except Exception as e:
                _raise_with_provider_hint(model, e)
                raise e
        elif tools is not None:
            try:
                response = completion_fn(
                    model=model,
                    messages=messages_with_system_prompt,
                    tools=tools,
                    tool_choice=tool_choice,
                    **extra_litellm_kwargs,
                )
            except Exception as e:
                _raise_with_provider_hint(model, e)
                raise e
        else:
            try:
                response = completion_fn(
                    model=model,
                    messages=messages_with_system_prompt,
                    **extra_litellm_kwargs,
                )
            except Exception as e:
                _raise_with_provider_hint(model, e)
                raise e

        return response

    def classify_with_logprob(
        self,
        model: str,
        messages: list[dict[str, str]],
        labels: list[Any],
        *,
        system_prompt: str | None = None,
        top_logprobs: int | None = None,
        litellm_completion_kwargs: dict[str, Any] = {},
    ) -> "tuple[Any, float]":
        """
        Single-token categorical classification with a calibrated confidence.

        Thin wrapper over :meth:`_classify_with_logprob_with_cost` that drops the
        cost component; see that method for the full contract. Returns
        ``(label, prob)`` -- the predicted label and its confidence in
        ``[0, 1]`` -- and is the cheap **proxy** path for model cascades (see
        ``docetl/operations/utils/cascade.py``).
        """
        label, prob, _cost = self._classify_with_logprob_with_cost(
            model,
            messages,
            labels,
            system_prompt=system_prompt,
            top_logprobs=top_logprobs,
            litellm_completion_kwargs=litellm_completion_kwargs,
        )
        return label, prob

    def _classify_with_logprob_with_cost(
        self,
        model: str,
        messages: list[dict[str, str]],
        labels: list[Any],
        *,
        system_prompt: str | None = None,
        top_logprobs: int | None = None,
        litellm_completion_kwargs: dict[str, Any] = {},
    ) -> "tuple[Any, float, float]":
        """
        Like :meth:`classify_with_logprob` but also returns the call's cost.

        Renders ``labels`` as a single-token numeric menu (``1 = <label0>``
        ...), asks the model to answer with just the menu number, requests
        token logprobs, and maps the answer token's probability back to a
        label. Cascade adapters use the returned cost for accounting; the
        public wrapper drops it.

        Rendering the labels as digits means booleans and enums both decode to
        a single token regardless of the label text, so one logprob lookup
        yields the answer. The returned probability is the softmax of the
        chosen label over the menu tokens present in ``top_logprobs`` and lies
        in ``[0, 1]``. Unlike the structured/tool paths, this uses
        ``litellm.completion`` directly (no router fallback): proxies are
        intentionally a single cheap model.

        Returns:
            ``(label, prob, cost)``.

        Raises:
            ValueError: if ``labels`` is empty or has more than 9 entries, or
                the provider does not return usable logprobs for the answer.
        """
        if not labels:
            raise ValueError("classify_with_logprob requires a non-empty label set")
        # Digits 1..9 are a single token in common tokenizers; beyond that we
        # cannot guarantee single-token decoding, which the logprob mapping
        # relies on.
        if len(labels) > 9:
            raise ValueError(
                "classify_with_logprob supports at most 9 labels (single-token "
                f"menu); got {len(labels)}. Split the task or use the oracle path."
            )

        token_to_label = {str(i + 1): label for i, label in enumerate(labels)}
        menu = "\n".join(f"{i + 1} = {label}" for i, label in enumerate(labels))
        instruction = (
            "Answer with ONLY the single number of the best option below -- no "
            "words, no punctuation, just the digit.\n\n" + menu
        )
        sys_msg = system_prompt or (
            "You are a precise classifier. Read the input and choose the single "
            "best option from the menu."
        )
        call_messages = (
            [{"role": "system", "content": sys_msg}]
            + list(messages)
            + [{"role": "user", "content": instruction}]
        )

        k = (
            top_logprobs
            if top_logprobs is not None
            else min(max(len(labels) + 4, 10), 20)
        )

        extra = dict(litellm_completion_kwargs)
        extra.setdefault("temperature", 0)
        if "gpt-5" in model:
            extra.pop("temperature", None)
        if self.default_lm_api_base and "api_base" not in extra:
            extra["api_base"] = self.default_lm_api_base

        self.runner.blocking_acquire("llm_call", weight=1)
        if self.runner.is_cancelled:
            raise asyncio.CancelledError("Operation was cancelled")

        try:
            response = completion(
                model=model,
                messages=call_messages,
                logprobs=True,
                top_logprobs=k,
                max_tokens=1,
                **extra,
            )
        except Exception as e:
            _raise_with_provider_hint(model, e)
            raise

        try:
            cost = completion_cost(response)
        except Exception:
            cost = 0.0
        self._track_token_usage(model, response)
        label, prob = self._parse_logprob_response(response, token_to_label)
        return label, prob, cost

    @staticmethod
    def _parse_logprob_response(
        response: Any, token_to_label: dict[str, Any]
    ) -> "tuple[Any, float]":
        """
        Map a logprobs completion response to ``(label, prob)``.

        Pure (no I/O): looks at the alternatives for the first generated token,
        keeps those that match a menu digit, softmax-normalizes their logprobs,
        and returns the argmax label with its conditional probability. Using
        softmax over menu tokens gives P(label | answer is valid) which provides
        meaningful confidence variation even when raw probabilities are extreme
        (common with gpt-4o-mini binary classification where logprobs are 0 or
        -25). When only one menu token appears, falls back to a clipped
        ``exp(logprob)`` so the score is never a degenerate 1.0. Kept separate
        from the network call so it can be unit-tested with synthetic responses.
        """
        try:
            content_lp = response.choices[0].logprobs.content
        except (AttributeError, IndexError, TypeError):
            content_lp = None
        if not content_lp:
            raise ValueError(
                "Model did not return token logprobs; the cascade proxy path "
                "requires a provider/model that supports `logprobs`."
            )

        first = content_lp[0]
        alts = getattr(first, "top_logprobs", None) or [first]

        logp_by_label: dict[Any, float] = {}
        for alt in alts:
            tok = (getattr(alt, "token", "") or "").strip()
            if tok not in token_to_label:
                continue
            lp = getattr(alt, "logprob", None)
            if lp is None:
                continue
            label = token_to_label[tok]
            if label not in logp_by_label or lp > logp_by_label[label]:
                logp_by_label[label] = lp

        if not logp_by_label:
            raise ValueError(
                "Proxy answer did not match any menu option; got token "
                f"{getattr(first, 'token', None)!r}."
            )

        present_labels = list(logp_by_label)
        logps = [logp_by_label[lbl] for lbl in present_labels]
        best = max(range(len(present_labels)), key=lambda i: logps[i])

        if len(present_labels) >= 2:
            # Compress the logprob gap through a sigmoid so the confidence
            # score has meaningful variation even when the model is extremely
            # confident (gpt-4o-mini routinely returns logprobs of 0 vs -25
            # for binary classification, making raw/softmax probs degenerate).
            # Scale=3 maps a ~6-nat gap to the 0.12–0.88 range.
            gap = logps[best] - min(logps)
            prob = 1.0 / (1.0 + math.exp(-gap / 3.0))
        else:
            prob = min(math.exp(logps[0]), 0.99)
        return present_labels[best], prob

    def parse_llm_response(
        self,
        response: Any,
        schema: dict[str, Any] = {},
        tools: list[dict[str, str]] | None = None,
        manually_fix_errors: bool = False,
        use_structured_output: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Parse the response from a language model.
        This function extracts the tool calls from the LLM response and returns the arguments
        """
        try:
            if not response:
                raise InvalidOutputError("No response from LLM", [{}], schema, [], [])

            # Go through each choice
            results = []
            for index in range(len(response.choices)):
                results.extend(
                    self._parse_llm_response_helper(
                        response, schema, tools, index, use_structured_output
                    )
                )
            return results
        except InvalidOutputError as e:
            if manually_fix_errors:
                rprint(
                    f"[bold red]Could not parse LLM output:[/bold red] {e.message}\n"
                    f"\tExpected Schema: {e.expected_schema}\n"
                    f"\tPlease manually set this output."
                )
                rprint(
                    f"\n[bold yellow]LLM-Generated Response:[/bold yellow]\n{response}"
                )
                output = get_user_input_for_schema(schema)

                return [output]
            else:
                raise e

    def _parse_llm_response_helper(
        self,
        response: Any,
        schema: dict[str, Any] = {},
        tools: list[dict[str, str]] | None = None,
        index: int = 0,
        use_structured_output: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Parse the response from a language model.

        This function extracts the tool calls from the LLM response and returns the arguments
        of any 'send_output' function calls as a list of dictionaries.

        Args:
            response (Any): The response object from the language model.
            schema (dict[str, Any] | None): The schema that was passed to the LLM.
            tools (list[dict[str, str]] | None): The tools that were passed to the LLM.
            use_structured_output (bool): Whether structured output mode was used.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the parsed output.

        Raises:
            InvalidOutputError: If the response is not valid.
        """
        # Handle structured output mode
        if use_structured_output:
            # Raw assistant content
            content = response.choices[index].message.content

            # Special-case deepseek-r1 style <think> tags
            if is_deepseek_r1(response.model):
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
                    # Content after </think>
                    main_content = re.split(r"</think>", content, maxsplit=1)[
                        -1
                    ].strip()
                else:
                    think_content = None
                    main_content = content

                # Parse the main JSON content
                try:
                    parsed_output = json.loads(main_content)
                except json.JSONDecodeError:
                    raise InvalidOutputError(
                        "Could not decode structured output JSON response",
                        main_content,
                        schema,
                        response.choices,
                        tools or [],
                    )

                if think_content is not None:
                    parsed_output = {"think": think_content, **parsed_output}

                # Attempt to parse any top-level values that are JSON-encoded strings
                for k, v in parsed_output.items():
                    if isinstance(v, str):
                        try:
                            if v.strip().startswith("{") or v.strip().startswith("["):
                                parsed_output[k] = json.loads(v)
                        except json.JSONDecodeError:
                            # leave value as-is if parsing fails
                            pass

                # Unwrap nested dict that redundantly nests the same key
                if (
                    isinstance(parsed_output[k], dict)
                    and len(parsed_output[k]) == 1
                    and k in parsed_output[k]
                ):
                    parsed_output[k] = parsed_output[k][k]

                return [parsed_output]

            # Default: just load JSON for structured output
            try:
                parsed_output = json.loads(content)
            except json.JSONDecodeError:
                raise InvalidOutputError(
                    "Could not decode structured output JSON response",
                    content,
                    schema,
                    response.choices,
                    tools or [],
                )

            # Attempt to parse any top-level values that are JSON-encoded strings
            for k, v in parsed_output.items():
                if isinstance(v, str):
                    try:
                        if v.strip().startswith("{") or v.strip().startswith("["):
                            parsed_output[k] = json.loads(v)
                    except json.JSONDecodeError:
                        # leave value as-is if parsing fails
                        pass

                # Unwrap nested dict that redundantly nests the same key
                if (
                    isinstance(parsed_output[k], dict)
                    and len(parsed_output[k]) == 1
                    and k in parsed_output[k]
                ):
                    parsed_output[k] = parsed_output[k][k]

            return [parsed_output]
        if is_snowflake(response.model):
            tool_calls = (
                [
                    ChatCompletionMessageToolCall(
                        function=Function(
                            name=content.get("tool_use", {}).get("name"),
                            arguments=content.get("tool_use", {}).get("input"),
                        )
                    )
                    for content in response.choices[index].message.content_list
                    if content.get("type") == "tool_use"
                ]
                if hasattr(response.choices[index].message, "content_list")
                else []
            )
        else:
            tool_calls = (
                response.choices[index].message.tool_calls
                if "tool_calls" in dir(response.choices[index].message)
                else []
            )

        # Check if there are no tools and the schema has a single key-value pair
        if not tools and len(schema) == 1 and not tool_calls:
            key = next(iter(schema))
            content = response.choices[index].message.content

            # Handle deepseek-r1 models' think tags
            if is_deepseek_r1(response.model):
                result = {}
                # Extract think content if present
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    result["think"] = think_match.group(1).strip()
                    # Get the remaining content after </think>
                    main_content = re.split(r"</think>", content, maxsplit=1)[
                        -1
                    ].strip()
                    result[key] = main_content
                else:
                    # If no think tags, just use the content as is
                    result[key] = content
                return [result]

            # For other models, continue with existing behavior
            if content is None:
                self.runner.console.log(
                    f"[bold yellow]Warning:[/bold yellow] LLM returned None content for single-key schema "
                    f"(key={key!r}, model={response.model!r}, "
                    f"finish_reason={response.choices[index].finish_reason!r}, "
                    f"completion_tokens={response.usage.completion_tokens if response.usage else 'unknown'!r}). "
                    f"Full response: {response!r}"
                )
                raise InvalidOutputError(
                    f"LLM returned None content (completion_tokens=0, finish_reason={response.choices[index].finish_reason!r}) — "
                    f"model={response.model!r}. Possible silent content policy refusal.",
                    [{}],
                    schema,
                    response.choices,
                    tools or [],
                )
            return [{key: content}]

        # Parse the response based on the provided tools
        if tools:
            # If custom tools are provided, parse accordingly
            tool_calls = response.choices[index].message.tool_calls
            results = []
            for tool_call in tool_calls:
                for tool in tools:
                    if tool_call.function.name == tool["function"]["name"]:
                        try:
                            function_args = (
                                json.loads(tool_call.function.arguments)
                                if isinstance(tool_call.function.arguments, str)
                                else tool_call.function.arguments
                            )
                        except json.JSONDecodeError:
                            self.runner.console.log(
                                f"[bold yellow]Warning:[/bold yellow] JSONDecodeError parsing tool call arguments "
                                f"(tool={tool_call.function.name!r}, model={response.model!r}). "
                                f"Raw arguments: {tool_call.function.arguments!r}"
                            )
                            return [{}]
                        # Execute the function defined in the tool's code
                        local_scope = {}
                        exec(tool["code"].strip(), globals(), local_scope)
                        function_result = local_scope[tool["function"]["name"]](
                            **function_args
                        )
                        function_args.update(function_result)
                        results.append(function_args)
            return results
        else:
            if not tool_calls:
                raise InvalidOutputError(
                    "No tool calls in LLM response", [{}], schema, response.choices, []
                )

            outputs = []
            for tool_call in tool_calls:
                if response.choices[index].finish_reason == "content_filter":
                    raise InvalidOutputError(
                        "Content filter triggered by LLM provider.",
                        "",
                        schema,
                        response.choices,
                        tools,
                    )
                try:
                    output_dict = (
                        json.loads(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, str)
                        else tool_call.function.arguments
                    )
                    # Augment output_dict with empty values for any keys in the schema that are not in output_dict
                    for key in schema:
                        if key not in output_dict:
                            output_dict[key] = "Not found"

                    if "ollama" in response.model or "sagemaker" in response.model:
                        for key, value in output_dict.items():
                            if not isinstance(value, str):
                                continue
                            try:
                                output_dict[key] = ast.literal_eval(value)
                            except Exception:
                                try:
                                    if value.startswith("["):
                                        output_dict[key] = ast.literal_eval(value + "]")
                                    else:
                                        output_dict[key] = value
                                except Exception:
                                    pass
                    outputs.append(output_dict)
                except json.JSONDecodeError:
                    raise InvalidOutputError(
                        "Could not decode LLM JSON response",
                        [tool_call.function.arguments],
                        schema,
                        response.choices,
                        tools,
                    )
                except Exception as e:
                    raise InvalidOutputError(
                        f"Error parsing LLM response: {e}",
                        [tool_call.function.arguments],
                        schema,
                        response.choices,
                        tools,
                    )

            return outputs

        # message = response.choices[0].message
        # return [json.loads(message.content)]

    def validate_output(self, operation: dict, output: dict, console: Console) -> bool:
        """
        Validate the output against the specified validation rules in the operation.

        Args:
            operation (dict): The operation dictionary containing validation rules.
            output (dict): The output to be validated.
            console (Console): The console object for logging.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        if "validate" not in operation:
            return True
        for validation in operation["validate"]:
            label = (
                getattr(validation, "__name__", None) or repr(validation)
                if callable(validation)
                else validation
            )
            try:
                passed = (
                    bool(validation(output))
                    if callable(validation)
                    else safe_eval(validation, output)
                )
                if not passed:
                    console.log(f"[bold red]Validation failed:[/bold red] {label}")
                    console.log(f"[yellow]Output:[/yellow] {output}")
                    return False
            except Exception as e:
                console.log(f"[bold red]Validation error:[/bold red] {str(e)}")
                console.log(f"[yellow]Output:[/yellow] {output}")
                return False
        return True

    def should_glean(
        self, gleaning_config: dict[str, Any] | None, output: dict[str, Any]
    ) -> bool:
        """Determine whether to execute a gleaning round based on an optional conditional expression.

        If ``gleaning_config`` contains an ``"if"`` key, its value is treated as a Python
        boolean expression that will be evaluated with the current ``output`` bound to the
        name ``output`` using :pyfunc:`safe_eval`. When the expression evaluates to
        ``True`` the gleaning round proceeds. If it evaluates to ``False`` (or raises an
        exception) the gleaning loop should terminate early.

        If no ``"if"`` key is present the method defaults to returning ``True`` so that
        gleaning proceeds normally.
        """
        # No gleaning_config or no conditional -> always glean
        if not gleaning_config or "if" not in gleaning_config:
            return True

        condition = gleaning_config.get("if")
        if not isinstance(condition, str):
            raise ValueError(
                f"Invalid gleaning condition (should be a string): {condition}"
            )

        try:
            return safe_eval(condition, output)
        except Exception as exc:
            # If evaluation fails, default to not glean and log for visibility
            self.runner.console.log(
                f"[bold red]Error evaluating gleaning condition '{condition}': {exc}; executing gleaning round anyway[/bold red]"
            )
            return False
