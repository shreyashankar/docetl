# Python API Reference

The Python API lets you build, run, and optimize DocETL pipelines with chainable operations.

```python
import docetl
```

!!! warning "Deprecated: the typed `Pipeline` class"
    The older object API (`from docetl.api import Pipeline` with `MapOp`,
    `ReduceOp`, etc.) is deprecated in favor of the Frame API documented on
    this page.

---

## Quick start

```python
import docetl

docetl.default_model = "gpt-4o-mini"

results = (
    docetl.read_json("input.json")
    .map(
        prompt="Classify this document: {{ input.text }}",
        output={"schema": {"category": "string"}},
    )
    .filter(prompt="Is this document about technology? {{ input.text }}")
    .reduce(
        reduce_key="category",
        prompt="Summarize these documents: {% for item in inputs %}{{ item.text }}{% endfor %}",
        output={"schema": {"summary": "string"}},
    )
    .collect()
)
```

## Configuration

Set global defaults as module-level attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `docetl.default_model` | `str` | `None` | Default LLM model for all operations |
| `docetl.agent_model` | `str` | `None` | Model for optimizer rewrites |
| `docetl.max_threads` | `int` | `cpu_count * 4` | Concurrent threads |
| `docetl.bypass_cache` | `bool` | `False` | Skip LLM cache |
| `docetl.intermediate_dir` | `str` | `None` | Directory for intermediate results (a relative path resolves against the working directory at run time) |
| `docetl.rate_limits` | `dict` | `None` | Rate limits per model |
| `docetl.fallback_models` | `list[str]` | `None` | Fallback chain on failure |
| `docetl.fallback_embedding_models` | `list[str]` | `None` | Fallback embedding models |
| `docetl.system_prompt` | `dict` | `None` | `{"dataset_description": ..., "persona": ...}` applied to all operations |

**Precedence.** Settings layer from most to least specific: a per-operation
parameter (e.g. `model=` on `.map()`) beats per-pipeline settings carried by a
Frame (set when loading a YAML via `Frame.from_yaml`), which beat the
module-level `docetl.*` globals above, which beat built-in defaults. Loading a
YAML never changes the globals — its settings travel with that Frame only.

---

## Reading Data

| Function | Description |
|----------|-------------|
| `docetl.read_json(path)` | Load from JSON file |
| `docetl.read_csv(path)` | Load from CSV file |
| `docetl.read_parquet(path)` | Load from Parquet file |
| `docetl.read_dir(path)` | One row per file in a directory (recursive): `text` is the file's content, plus `filename` and `path`. PDF/Word/PowerPoint/Excel are converted to text; other files read as UTF-8 |
| `docetl.from_list(data)` | Load from a list of dicts |
| `docetl.Frame.from_yaml(path)` | Load from a YAML pipeline config |

All return a `Frame`.

---

## Frame

A lazy pipeline — operations are recorded but not executed until a terminal action is called. Frames are immutable; every operation returns a new `Frame`.

### LLM Operations

#### `.map()`

Applies an LLM prompt to each document independently.

```python
frame.map(
    prompt="Classify: {{ input.text }}",
    output={"schema": {"category": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Jinja2 template. Access fields via `{{ input.key }}`. |
| `output` | `dict` | — | Output schema, e.g. `{"schema": {"field": "str"}}` |
| `model` | `str` | `None` | Override default model |
| `validate` | `list[str \| callable]` | `None` | Validators: expression strings over `output`, or callables taking the output dict (callables can't be exported to YAML) |
| `num_retries_on_validate_failure` | `int` | `None` | Retries on validation failure |
| `sample` | `int` | `None` | Process only N documents |
| `agent` | `docetl.Agent` | `None` | Tool-equipped agent using the OpenAI Agents SDK |
| `drop_keys` | `list[str]` | `None` | Keys to remove from output |
| `timeout` | `int` | `None` | Timeout per LLM call (seconds) |
| `max_batch_size` | `int` | `None` | Batch size for batch processing |
| `batch_prompt` | `str` | `None` | Jinja2 template for batch mode |
| `execution` | `dict` | `None` | Deferred provider or vLLM batch execution. See [Deferred Batch Execution](../execution/batch.md). |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |
| `optimize` | `bool` | `None` | Mark for optimization |
| `limit` | `int` | `None` | Max documents to process |

#### `.filter()`

Keeps or removes documents based on an LLM prompt. Output schema must have one boolean field.

```python
frame.filter(
    prompt="Is this about technology? {{ input.text }}",
    output={"schema": {"is_tech": "bool"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Jinja2 template |
| `output` | `dict` | — | Schema with one boolean field |
| `model` | `str` | `None` | Override default model |
| `validate` | `list[str \| callable]` | `None` | Validators: expression strings or callables over the output dict |
| `agent` | `docetl.Agent` | `None` | Tool-equipped agent using the OpenAI Agents SDK |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |
| `cascade` | `dict` | `None` | [Model cascade](../optimization/cascades.md): run a cheap proxy (chat or embedding model) on all items and escalate only uncertain ones, with a statistical guarantee. Also available on `resolve` and `equijoin`. |

```python
frame.filter(
    prompt="Is this review about shipping problems? {{ input.text }}",
    output={"schema": {"keep": "bool"}},
    cascade={"proxy_model": "text-embedding-3-small", "guarantee": "recall",
             "target": 0.9, "label_budget": 120},
)
```

#### `.reduce()`

Groups documents by key and reduces each group with an LLM.

```python
frame.reduce(
    reduce_key="category",
    prompt="Summarize: {% for item in inputs %}{{ item.text }}{% endfor %}",
    output={"schema": {"summary": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduce_key` | `str \| list[str]` | — | Key(s) to group by. Use `"_all"` for one group. |
| `prompt` | `str` | — | Jinja2 template. Iterate with `{% for item in inputs %}`. |
| `output` | `dict` | — | Output schema |
| `fold_prompt` | `str` | `None` | Prompt for incremental folding |
| `fold_batch_size` | `int` | `None` | Items per fold iteration |
| `merge_prompt` | `str` | `None` | Prompt for merging fold results |
| `pass_through` | `bool` | `None` | Pass through non-reduced keys |
| `associative` | `bool` | `None` | Enable parallel reduction |
| `agent` | `docetl.Agent` | `None` | Tool-equipped agent using the OpenAI Agents SDK |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |

#### Tool-equipped map/filter/reduce

Use `docetl.Agent` when an operation should call tools over multiple turns
before returning DocETL's structured output. DocETL adapts Python functions into
OpenAI Agents SDK tools and routes the operation's LiteLLM model through the
SDK's LiteLLM integration.

```python
import docetl

@docetl.tool
def lookup_sla(customer_tier: str) -> dict[str, str | int]:
    """Return support entitlements for a customer tier."""
    return {
        "enterprise": {"response_hours": 1, "escalation": "page-on-call"},
        "growth": {"response_hours": 4, "escalation": "queue-lead"},
        "free": {"response_hours": 48, "escalation": "self-serve"},
    }.get(customer_tier.lower(), {"response_hours": 24, "escalation": "standard"})

agent = docetl.Agent(tools=[lookup_sla], max_turns=5, max_tool_calls=3)

rows = (
    docetl.from_list([
        {
            "ticket": "Production API latency is above the SLO.",
            "customer_tier": "enterprise",
        }
    ])
    .map(
        prompt=(
            "Use lookup_sla to classify this ticket and explain the next action: "
            "{{ input.ticket }} / tier={{ input.customer_tier }}"
        ),
        output={"schema": {"priority": "str", "next_action": "str"}},
        model="azure/gpt-4o-mini",
        agent=agent,
    )
    .collect()
)
```

The model is still specified on the operation (`model=`) or through
`docetl.default_model`; use LiteLLM model names such as `azure/gpt-4o-mini`,
`anthropic/...`, or `together_ai/...`. The selected model/provider must work
with the OpenAI Agents SDK LiteLLM integration and the requested tools. Plain
Python function tools execute as trusted Python in your process; OpenAI Agents
SDK sandbox/native tools may provide isolation where that SDK/backend supports
it. Agent configs are Python-only and cannot be exported to YAML. See OpenAI's
[Agents SDK guide](https://developers.openai.com/api/docs/guides/agents) and
[tools guide](https://openai.github.io/openai-agents-python/tools/) for hosted
tool behavior.

Agents can also expose specialist agents as tools. Use this when a manager agent
should keep ownership of the DocETL output while delegating a bounded task:

```python
specialist = docetl.Agent(
    instructions="Extract numeric evidence from the supplied text.",
    max_turns=4,
)

manager = docetl.Agent(
    tools=[
        specialist.as_tool(
            name="extract_numeric_evidence",
            description="Extract numeric evidence for the manager agent.",
        )
    ],
    max_turns=6,
)
```

The specialist uses the same operation-level LiteLLM model as the manager unless
you pass an SDK-native tool object with its own model configuration. This follows
OpenAI's
[agents-as-tools orchestration pattern](https://developers.openai.com/api/docs/guides/agents/orchestration).

For a persistent shell/sandbox filesystem on OpenAI, create a hosted container
once and bind shell tools to it:

```python
sandbox = docetl.tools.Sandbox.create(
    name="docetl-research",
    network="disabled",
    memory_limit="1g",
)
bash = sandbox.bash()

agent = docetl.Agent(
    tools=[bash],
    max_turns=4,
    max_tool_calls=6,
)
```

Reuse `bash` across a manager and its specialist subagents when they should read
and write the same hosted filesystem. If you only need an ephemeral shell tool,
use `docetl.tools.bash(...)`, which uses `container_auto`. Pass durable data
between DocETL operations through output schemas, not only through hidden
sandbox files.

`docetl.tools.Sandbox.create(...)` is OpenAI-specific: it calls the OpenAI
Containers API and returns shell tools using `container_reference`. It is not
portable to Claude, Together, or other LiteLLM providers. For provider-portable
tooling, use `@docetl.tool` Python functions, MCP tools, or provider-native SDK
tools. See OpenAI's
[sandbox agents guide](https://developers.openai.com/api/docs/guides/agents/sandboxes).

For an end-to-end example, see the
[Tool-Equipped Agents tutorial](../examples/tool-equipped-research-agents.md).

#### `.resolve()`

Deduplicates entities by pairwise LLM comparison.

```python
frame.resolve(
    comparison_prompt="Same person? {{ input1.name }} vs {{ input2.name }}",
    resolution_prompt="Canonical name: {% for e in inputs %}{{ e.name }}{% endfor %}",
    output={"schema": {"name": "str"}},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comparison_prompt` | `str` | — | Jinja2 template comparing `{{ input1 }}` and `{{ input2 }}` |
| `resolution_prompt` | `str` | `None` | Prompt for resolving matched groups |
| `output` | `dict` | `None` | Output schema |
| `blocking_keys` | `list[str]` | `None` | Keys for blocking |
| `blocking_threshold` | `float` | `None` | Similarity threshold |
| `embedding_model` | `str` | `None` | Model for blocking embeddings |
| `optimize` | `bool` | `None` | Mark for optimization |

#### `.extract()`

Extracts information from documents with line-level precision.

```python
frame.extract(
    prompt="Extract key findings from this paper.",
    document_keys=["content"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Extraction prompt |
| `document_keys` | `list[str]` | — | Keys containing document text |
| `retriever` | `Retriever` | `None` | Retriever for context augmentation |

#### `.parallel_map()`

Runs multiple prompts on each document in parallel.

```python
frame.parallel_map(
    prompts=[{"prompt": "...", "output_keys": ["field1"]}, ...],
    output={"schema": {"field1": "str", "field2": "str"}},
)
```

#### `.equijoin()`

Joins two datasets by LLM comparison.

```python
right = docetl.read_json("other.json")
frame.equijoin(right, comparison_prompt="Are these related? {{ left.x }} {{ right.y }}")
```

### Structural Operations (no LLM calls)

| Method | Description |
|--------|-------------|
| `.split(split_key, method, method_kwargs)` | Split documents into chunks |
| `.gather(content_key, doc_id_key, order_key)` | Add surrounding context to chunks |
| `.unnest(unnest_key)` | Flatten a list field into separate rows |
| `.cluster(embedding_keys)` | Cluster documents by embedding similarity |
| `.sample(samples, method)` | Sample a subset of documents |

### Code Operations (no LLM calls)

`code` takes any callable, e.g., a lambda, or a string of Python source defining a `transform` function (the YAML form).

| Method | Description |
|--------|-------------|
| `.code_map(code=lambda doc: {...})` | Per-document Python transform |
| `.code_filter(code=lambda doc: bool(...))` | Per-document Python filter (return bool) |
| `.code_reduce(reduce_key, code=lambda items: {...})` | Per-group Python aggregation |

---

## Retriever

Augment LLM operations with retrieved context from a LanceDB index.

```python
retriever = docetl.Retriever(
    dataset="knowledge_base",
    index_dir="./lance_index",
    index_types=["fts", "embedding"],
    fts={"index_phrase": "{{ input.text }}", "query_phrase": "{{ input.question }}"},
    embedding={"model": "text-embedding-3-small", "index_phrase": "...", "query_phrase": "..."},
    query={"mode": "hybrid", "top_k": 5},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| list[dict]` | — | File path or in-memory records to index (alternative to `dataset`) |
| `dataset` | `str` | — | Name of an existing pipeline dataset or step output to index |
| `index_dir` | `str` | — | Directory for the LanceDB index |
| `index_types` | `list[str]` | — | `["fts"]`, `["embedding"]`, or both |
| `fts` | `dict` | `None` | Full-text search config (`index_phrase`, `query_phrase`) |
| `embedding` | `dict` | `None` | Embedding config (`model`, `index_phrase`, `query_phrase`) |
| `query` | `dict` | `None` | Query config (`mode`, `top_k`) |
| `build_index` | `str` | `"if_missing"` | `"if_missing"`, `"always"`, or `"never"` |

Pass to any LLM operation via `retriever=`. Retrieved context is available as `{{ retrieval_context }}` in prompts.

Pass exactly one of `data` (a file path or list of dicts to index) or `dataset` (the name of an existing pipeline dataset — the frame's own input or a previous step's output, `step_<operation_name>`). See the [Retrievers guide](../retrievers.md).

---

## Inspection (no execution)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `frame.schema()` | `dict[str, str]` | Output schema from operation definitions, including structural ops (split/unnest/gather/extract). Best-effort: `code_*` op outputs can't be known statically |
| `frame.count()` | `int` | Input count (no ops) or output count (executes if ops present) |
| `frame.to_yaml()` | `str` | Export pipeline as YAML config |
| `frame.to_yaml(path)` | `str` | Also write YAML to file |
| `frame.to_python()` | `str` | Export as Python source code |

---

## Terminal Actions

| Method | Return Type | Description |
|--------|-------------|-------------|
| `frame.show(max=5)` | `DataFrame` | Run on a sample and print results. Works on bare datasets too. |
| `frame.collect()` | `list[dict]` | Execute the pipeline, return the result rows |
| `frame.to_pandas()` | `DataFrame` | Execute the pipeline, return a pandas DataFrame |
| `frame.write_json(path)` | `None` | Execute and write to JSON |
| `frame.write_csv(path)` | `None` | Execute and write to CSV |
| `frame.write_parquet(path)` | `None` | Execute and write to Parquet |

Terminal actions are memoized on the Frame: repeated calls with an unchanged configuration reuse the previous result instead of re-running the pipeline. Changing ops, in-memory data, or `docetl.*` settings invalidates the memo; edits to input *files* between calls are not detected.

### Cost & Token Tracking

```python
rows = frame.collect()

print(f"Cost: ${frame.total_cost:.4f}")
print(f"Tokens: {frame.token_usage}")

df = frame.to_pandas()
print(f"Cost: ${df.attrs['_total_cost']:.4f}")  # also on the DataFrame
```

---

## Optimization

```python
@docetl.register_eval
def evaluate(results):
    correct = sum(1 for r in results if r.get("correct"))
    return {"score": correct}

optimized = frame.optimize(
    eval_fn=evaluate,
    metric_key="score",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_fn` | `Callable` | **required** | Scores pipeline output. Returns a dict of metrics. |
| `metric_key` | `str` | **required** | Key in eval_fn's return dict to optimize |
| `models` | `list[str]` | Auto-detect | LiteLLM model names to explore |
| `agent_model` | `str` | Auto-select | Model for rewrite agent |
| `max_iterations` | `int` | `20` | Search budget |
| `save_dir` | `str` | Temp dir | Where to save results |
| `exploration_weight` | `float` | `1.414` | UCB exploration constant |
| `dataset_path` | `str` | Pipeline's dataset | Sample dataset for optimization |
| `max_threads` | `int` | `None` | Max concurrent LLM calls per run |
| `max_concurrent_agents` | `int` | `3` | Parallel MCTS search agents |

Returns an optimized `Frame`. Access search results via `optimized.search_results`:

| Method / Property | Return Type | Description |
|-------------------|-------------|-------------|
| `.best()` | `OptimizedPipeline` | Highest-accuracy solution |
| `.cheapest()` | `OptimizedPipeline` | Lowest-cost solution |
| `.frontier` | `list[OptimizedPipeline]` | All Pareto-optimal solutions |
| `.to_df()` | `DataFrame` | All explored plans |
