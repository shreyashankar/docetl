# Map Operation

The Map operation applies a transformation to each item in your input data.

```mermaid
flowchart LR
    d1["doc 1"] --> o1["doc 1 + new fields"]
    d2["doc 2"] --> o2["doc 2 + new fields"]
    d3["doc 3"] --> o3["doc 3 + new fields"]
    dn["..."] --> on["..."]
```

## Example: Analyzing Long-Form News Articles

=== "YAML"

    ```yaml
    - name: analyze_news_article
      type: map
      prompt: |
        Analyze the following news article:
        "{{ input.article }}"

        Provide the following information:
        1. Main topic (1-3 words)
        2. Summary (2-3 sentences)
        3. Key entities mentioned (list up to 5, with brief descriptions)
        4. Sentiment towards the main topic (positive, negative, or neutral)
        5. Potential biases or slants in reporting (if any)
        6. Relevant categories (e.g., politics, technology, environment; list up to 3)
        7. Credibility score (1-10, where 10 is highly credible)

      output:
        schema:
          main_topic: string
          summary: string
          key_entities: list[object]
          sentiment: string
          biases: list[string]
          categories: list[string]
          credibility_score: integer

      model: gpt-4o-mini
      validate:
        - len(output["main_topic"].split()) <= 3
        - len(output["key_entities"]) <= 5
        - output["sentiment"] in ["positive", "negative", "neutral"]
        - len(output["categories"]) <= 3
        - 1 <= output["credibility_score"] <= 10
      num_retries_on_validate_failure: 2
    ```

=== "Python"

    ```python
    import docetl

    docetl.default_model = "gpt-4o-mini"

    frame = docetl.read_json("articles.json")
    frame = frame.map(
        prompt="""Analyze the following news article:
    "{{ input.article }}"

    Provide the following information:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Key entities mentioned (list up to 5, with brief descriptions)
    4. Sentiment towards the main topic (positive, negative, or neutral)
    5. Potential biases or slants in reporting (if any)
    6. Relevant categories (e.g., politics, technology, environment; list up to 3)
    7. Credibility score (1-10, where 10 is highly credible)""",
        output={
            "schema": {
                "main_topic": "string",
                "summary": "string",
                "key_entities": "list[object]",
                "sentiment": "string",
                "biases": "list[string]",
                "categories": "list[string]",
                "credibility_score": "integer",
            }
        },
        model="gpt-4o-mini",
        validate=[
            lambda output: len(output["main_topic"].split()) <= 3,
            lambda output: len(output["key_entities"]) <= 5,
            lambda output: output["sentiment"] in ["positive", "negative", "neutral"],
            lambda output: 1 <= output["credibility_score"] <= 10,
        ],
        num_retries_on_validate_failure=2,
    )
    rows = frame.collect()
    ```

??? example "Sample Input and Output"

    Input:
    ```json
    [
      {
        "article": "In a groundbreaking move, the European Union announced yesterday a comprehensive plan to transition all member states to 100% renewable energy by 2050. The ambitious proposal, dubbed 'Green Europe 2050', aims to completely phase out fossil fuels and nuclear power across the continent.

        European Commission President Ursula von der Leyen stated, 'This is not just about fighting climate change; it's about securing Europe's energy independence and economic future.' The plan includes massive investments in solar, wind, and hydroelectric power, as well as significant funding for research into new energy storage technologies.

        However, the proposal has faced criticism from several quarters. Some Eastern European countries, particularly Poland and Hungary, argue that the timeline is too aggressive and could damage their economies, which are still heavily reliant on coal. Industry groups have also expressed concern about the potential for job losses in the fossil fuel sector.

        Environmental groups have largely praised the initiative, with Greenpeace calling it 'a beacon of hope in the fight against climate change.' However, some activists argue that the 2050 target is not soon enough, given the urgency of the climate crisis.

        The plan also includes provisions for a 'just transition,' with billions of euros allocated to retraining workers and supporting regions that will be most affected by the shift away from fossil fuels. Additionally, it proposes stricter energy efficiency standards for buildings and appliances, and significant investments in public transportation and electric vehicle infrastructure.

        Experts are divided on the feasibility of the plan. Dr. Maria Schmidt, an energy policy researcher at the University of Berlin, says, 'While ambitious, this plan is achievable with the right political will and technological advancements.' However, Dr. John Smith from the London School of Economics warns, 'The costs and logistical challenges of such a rapid transition should not be underestimated.'

        As the proposal moves forward for debate in the European Parliament, it's clear that 'Green Europe 2050' will be a defining issue for the continent in the coming years, with far-reaching implications for Europe's economy, environment, and global leadership in climate action."
      }
    ]
    ```

    Output:
    ```json
    [
      {
        "main_topic": "EU Renewable Energy",
        "summary": "The European Union has announced a plan called 'Green Europe 2050' to transition all member states to 100% renewable energy by 2050. The ambitious proposal aims to phase out fossil fuels and nuclear power, invest in renewable energy sources, and includes provisions for a 'just transition' to support affected workers and regions.",
        "key_entities": [
          {
            "name": "European Union",
            "description": "Political and economic union of 27 member states"
          },
          {
            "name": "Ursula von der Leyen",
            "description": "European Commission President"
          },
          {
            "name": "Poland",
            "description": "Eastern European country critical of the plan"
          },
          {
            "name": "Hungary",
            "description": "Eastern European country critical of the plan"
          },
          {
            "name": "Greenpeace",
            "description": "Environmental organization supporting the initiative"
          }
        ],
        "sentiment": "positive",
        "biases": [
          "Slight bias towards environmental concerns over economic impacts",
          "More emphasis on supportive voices than critical ones"
        ],
        "categories": [
          "Environment",
          "Politics",
          "Economy"
        ],
        "credibility_score": 8
      }
    ]
    ```

## Required Parameters

- `name`: A unique name for the operation.
- `type`: Must be set to "map".

## Optional Parameters

| Parameter                         | Description                                                                                     | Default                       |
| --------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------- |
| `prompt`                          | The prompt template to use for the transformation. Access input variables with `input.keyname`. | None                          |
| `batch_prompt`                    | Template for processing multiple documents in a single prompt. Access batch with `inputs` list. | None                          |
| `max_batch_size`                  | Maximum number of documents to process in a single batch                                        | None                          |
| `output.schema`                   | Schema definition for the output from the LLM.                                                  | None                          |
| `output.n`                        | Number of outputs to generate for each input. (only available for OpenAI models; this is used to generate multiple outputs from a single input and automatically turn into a bigger list)                                                    | 1                             |
| `model`                           | The language model to use                                                                       | Falls back to `default_model` |
| `optimize`                        | Flag to enable operation optimization                                                           | `True`                        |
| `recursively_optimize`            | Flag to enable recursive optimization of operators synthesized as part of rewrite rules         | `false`                       |
| `sample`                     | Number of samples to use for the operation                                                      | Processes all data            |
| `limit`                      | Maximum number of outputs to produce before stopping                                            | Processes all data            |
| `agent`                           | Python-only `docetl.Agent` config for tool-equipped map agents                                  | None                          |
| `validate`                        | List of Python expressions to validate the output                                               | None                          |
| `flush_partial_results`           | Write results of individual batches of map operation to disk for faster inspection              | False  |
| `num_retries_on_validate_failure` | Number of retry attempts on validation failure                                                  | 0                             |
| `gleaning`                        | Configuration for advanced validation and LLM-based refinement                                  | None                          |
| `drop_keys`                       | List of keys to drop from the input before processing                                           | None                          |
| `timeout`                         | Timeout for each LLM call in seconds                                                            | 120                           |
| `max_retries_per_timeout`         | Maximum number of retries per timeout                                                           | 2                             |
| `litellm_completion_kwargs` | Additional parameters to pass to LiteLLM completion calls. | {}                          |
| `skip_on_error` | If true, skip the operation if the LLM returns an error. | False                          |
| `bypass_cache` | If true, bypass the cache for this operation. | False                          |
| `pdf_url_key` | If specified, the key in the input that contains the URL of the PDF to process. | None                          |
| `calibrate` | Improve consistency across documents by using sample data as reference anchors. | False                          |
| `num_calibration_docs` | Number of documents to use sample and generate outputs for, for calibration. | 10                          |
| `retriever` | Name of a retriever to use for RAG. See [Retrievers](../retrievers.md). | None                          |
| `save_retriever_output` | If true, saves the retrieved context to `_<operation_name>_retrieved_context` in the output. | False                          |
| `execution` | Send this operation's LLM calls through a provider Batch API or an offline vLLM job. See [Deferred Batch Execution](../execution/batch.md). | None                          |

Note: If `drop_keys` is specified, `prompt` and `output` become optional parameters.

### Limiting execution

Set `limit` when you only need the first _N_ map results or want to cap LLM spend. The operation slices the processed dataset to the first `limit` entries and also stops scheduling new prompts once that many outputs have been produced, even if a prompt returns multiple records. Filter operations inherit this behavior but redefine the count so the limit only applies to records whose filter predicate evaluates to `true` (see [Filter](./filter.md#optional-parameters)).


!!! info "Validation and Gleaning"

    For more details on validation techniques and implementation, see [operators](../concepts/operators.md#validation).

### Batch Processing

!!! note "Prompt Batching vs. Execution Batching"

    This section describes `batch_prompt`, which combines several rows into
    one prompt. To keep one request per row and submit those requests through
    a provider Batch API or `vllm run-batch` instead, see
    [Deferred Batch Execution](../execution/batch.md).

The `batch_prompt` parameter processes multiple documents in a single prompt. This reduces LLM call counts for simple tasks and short documents, but larger batch sizes (even > 5) can lead to more incorrect results.

??? example "Batch Processing Example"

    === "YAML"

        ```yaml
        - name: classify_documents
          type: map
          max_batch_size: 5  # Process up to 5 documents in a single LLM call
          batch_prompt: |
            Classify each of the following documents into categories (technology, business, or science):
            
            {% for doc in inputs %}
            Document {{loop.index}}:
            {{doc.text}}
            {% endfor %}
            
            Provide a classification for each document.
          prompt: |
            Classify the following document:
            {{input.text}}
          output:
            schema:
              category: string
        ```

    === "Python"

        ```python
        frame = frame.map(
            name="classify_documents",
            max_batch_size=5,  # Process up to 5 documents in a single LLM call
            batch_prompt="""Classify each of the following documents into categories (technology, business, or science):

        {% for doc in inputs %}
        Document {{loop.index}}:
        {{doc.text}}
        {% endfor %}

        Provide a classification for each document.""",
            prompt="""Classify the following document:
        {{input.text}}""",
            output={"schema": {"category": "string"}},
        )
        ```

When using batch processing:

1. The `batch_prompt` template receives an `inputs` list containing the batch of documents
2. Use `max_batch_size` to control how many documents are processed in each batch
3. You must also provide a `prompt` parameter that will be used in case the batch prompt's response cannot be parsed into the output schema
4. Gleaning and validation are applied to each document in the batch individually, after the batch has been processed by the LLM

!!! tip "Batch Size Considerations"

    Choose your `max_batch_size` carefully:
    
    - Larger batches may be more efficient but risk hitting token limits
    - Start with smaller batches (3-5 documents) and adjust based on your needs
    - Consider document length when setting batch size

### Calibration for Consistency

With `calibrate: true`, the operation samples a subset of documents, processes them with the original prompt, and uses those results to generate reference anchors that are appended to the prompt for all documents. Use it for:

- Classification tasks where documents are evaluated relative to each other
- Rating/scoring operations that need consistent scales
- Subjective judgments that benefit from concrete examples
- Datasets that vary widely, when consistency matters more than individual accuracy (works best with 20+ documents)

??? example "Document Priority Classification with Calibration"

    Imagine you're processing a large collection of customer support tickets and want to classify them by priority. Without calibration, the LLM might be inconsistent - a "medium" priority ticket early in processing might be classified as "high" later when the LLM sees more severe issues.

    === "YAML"

        ```yaml
        - name: classify_ticket_priority
          type: map
          calibrate: true  # Enable calibration
          num_calibration_docs: 15  # Use 15 tickets for calibration
          prompt: |
            Classify the following customer support ticket by priority level:
            
            Subject: {{ input.subject }}
            Description: {{ input.description }}
            Customer Tier: {{ input.customer_tier }}
            
            Classify as: low, medium, high, or critical
          output:
            schema:
              priority: string
              reasoning: string
          model: gpt-4o-mini
        ```

    === "Python"

        ```python
        frame = frame.map(
            name="classify_ticket_priority",
            calibrate=True,  # Enable calibration
            num_calibration_docs=15,  # Use 15 tickets for calibration
            prompt="""Classify the following customer support ticket by priority level:

        Subject: {{ input.subject }}
        Description: {{ input.description }}
        Customer Tier: {{ input.customer_tier }}

        Classify as: low, medium, high, or critical""",
            output={"schema": {"priority": "string", "reasoning": "string"}},
            model="gpt-4o-mini",
        )
        ```

    **How calibration works:**

    1. **Sample**: Randomly selects 15 tickets from your dataset (using seed=42 for reproducibility)
    2. **Process**: Runs the original prompt on these 15 tickets
    3. **Analyze**: An LLM analyzes the sample results and generates reference anchors
    4. **Augment**: Appends these reference anchors to your original prompt
    5. **Execute**: Processes all tickets with the augmented prompt

    **Example calibration output:**
    ```yaml
    # Original prompt gets augmented with something like:
    # 
    # For reference, consider 'Server completely down for 500+ users' → critical as your baseline for critical issues.
    # Documents similar to 'Login button not working for one user' → low priority.
    # For reference, consider 'Payment processing delays affecting checkout' → high as your standard for high priority issues.
    ```

!!! note "Calibration Considerations"

    - Adds a small overhead cost (processes calibration samples + calibration analysis)
    - Uses a fixed seed (42) for reproducible sampling
    - The calibration LLM call uses temperature=0.0 for consistent results

## Advanced Features

### PDF Processing

The Map operation can directly process PDFs using Claude or Gemini models. To use this feature:

1. Your input dataset must contain a key representing the URL of the PDF to process
2. Specify this field name using the `pdf_url_key` parameter in your map operation
3. The URLs must be publicly accessible or accessible to your environment

??? example "PDF Processing Example"

    Here's an example of processing a dataset of papers, where each paper is represented by a URL.

    === "YAML"

        ```yaml
        datasets:
          papers:
            type: file
            path: "papers/urls.json"  # Contains documents with PDF URLs

        default_model: gemini/gemini-2.0-flash  # or claude models
        operations:
          - name: extract_paper_info
            type: map
            pdf_url_key: url  # Tells DocETL which field contains the PDF URL
            prompt: |
              Summarize the paper.
            output:
              schema:
                paper_info: string

        pipeline:
          steps:
            - name: extract_paper_info
              input: papers
              operations:
                - extract_paper_info
        ```

    === "Python"

        ```python
        import docetl

        docetl.default_model = "gemini/gemini-2.0-flash"  # or claude models

        frame = docetl.read_json("papers/urls.json")  # Contains documents with PDF URLs
        frame = frame.map(
            name="extract_paper_info",
            pdf_url_key="url",  # Tells DocETL which field contains the PDF URL
            prompt="Summarize the paper.",
            output={"schema": {"paper_info": "string"}},
        )
        rows = frame.collect()
        ```

    Your input data (`papers/urls.json`) should contain documents with PDF URLs:
    ```json
    [
      {"url": "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card.pdf"},
      ...
    ]
    ```

    DocETL will:
    1. Download each PDF
    2. Extract the text content
    3. Pass the content to the LLM with your prompt
    4. Return the processed results:

    ```json
    [
      {
        "url": "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card.pdf",
        "paper_info": "This paper introduces Claude 3.5 Haiku and the upgraded Claude 3.5 Sonnet..."
      },
      ...
    ]
    ```

### Tool-equipped map agents

For Python pipelines, pass `agent=docetl.Agent(...)` when a map operation should
call tools before returning its structured output. The operation's `model=`
still controls model selection. Function tools wrapped with `@docetl.tool` work
through the OpenAI Agents SDK tool loop; OpenAI-hosted tools such as web search
or hosted shell require an OpenAI-compatible hosted-tool path.

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

frame = frame.map(
    prompt=(
        "Use lookup_sla to classify this ticket and explain the next action: "
        "{{ input.ticket }} / tier={{ input.customer_tier }}"
    ),
    output={"schema": {"priority": "str", "next_action": "str"}},
    model="azure/gpt-4o-mini",
    agent=agent,
)
```

Plain Python tools execute as trusted Python in your process. OpenAI Agents SDK
tools, including sandbox/native tools where supported by the SDK backend, can be
passed through in Python agent configs. Agent configs are Python-only and cannot
be exported to YAML.

See the [Python API reference](../api-reference/python.md#tool-equipped-mapfilterreduce)
for the full API and the
[Tool-Equipped Agents tutorial](../examples/tool-equipped-research-agents.md) for
a map/reduce example with web search, hosted shell, and specialist subagents.

### Input Truncation

If the input doesn't fit within the token limit, DocETL automatically truncates tokens from the middle of the input data, preserving the beginning and end which often contain more important context. A warning is displayed when truncation occurs.

### Batching

To limit how many documents are processed concurrently, set `max_batch_size`.

=== "YAML"

    ```yaml
    - name: extract_summaries
      type: map
      max_batch_size: 5
      clustering_method: random
      prompt: |
        Summarize this text: "{{ input.text }}"
      output:
        schema:
          summary: string
    ```

=== "Python"

    ```python
    frame = frame.map(
        name="extract_summaries",
        max_batch_size=5,
        clustering_method="random",
        prompt='Summarize this text: "{{ input.text }}"',
        output={"schema": {"summary": "string"}},
    )
    ```

In the above config, there will be no more than 5 API calls to the LLM at a time (i.e., 5 documents processed at a time, one per API call).

### Dropping Keys

A map operation with only `drop_keys` acts as a no-op that removes the listed key-value pairs (no LLM calls).

=== "YAML"

    ```yaml
    - name: drop_keys_example
      type: map
      drop_keys:
        - "keyname1"
        - "keyname2"
    ```

=== "Python"

    ```python
    frame = frame.map(
        name="drop_keys_example",
        drop_keys=["keyname1", "keyname2"],
    )
    ```

## Best Practices

1. **Optimize for Scale**: For large datasets, use `sample` to test your operation before running on the full dataset.
2. **Use Tools**: Tools can run any Python code, including calling other APIs, for calculations the LLM might get wrong.

### Synthetic Data Generation

Set `output.n` greater than 1 to generate multiple outputs per input, returned as separate items. This multiplies your dataset size by `n`.

??? example "Synthetic Email Generation Example"

    Imagine you have a dataset of prospects and want to generate multiple email templates for each person. Here's how to generate 10 different email templates per prospect:

    === "YAML"

        ```yaml
        datasets:
          prospects:
            type: file
            path: "prospects.json"  # Contains names, companies, roles, etc.

        operations:
          - name: generate_email_templates
            type: map
            bypass_cache: true
            optimize: true
            output:
              n: 10  # Generate 10 unique emails per prospect
              schema:
                subject: "str"
                body: "str"
                call_to_action: "str"
            prompt: |
              Create a personalized cold outreach email for the following prospect:
              
              Name: {{ input.name }}
              Company: {{ input.company }}
              Role: {{ input.role }}
              Industry: {{ input.industry }}
              
              The email should:
              - Have a compelling subject line
              - Be brief (3-5 sentences)
              - Mention a specific pain point for their industry
              - Include a clear call to action
              - Sound natural and conversational, not sales-y
              
              Your response should be formatted as:
              Subject: [Your subject line]
              
              [Your email body]
              
              Call to action: [Your specific call to action]

        pipeline:
          steps:
            - name: email_generation
              input: prospects
              operations:
                - generate_email_templates
          
          output:
            type: file
            path: "email_templates.json"
        ```

    === "Python"

        ```python
        import docetl

        frame = docetl.read_json("prospects.json")  # Contains names, companies, roles, etc.
        frame = frame.map(
            name="generate_email_templates",
            bypass_cache=True,
            optimize=True,
            output={
                "n": 10,  # Generate 10 unique emails per prospect
                "schema": {
                    "subject": "str",
                    "body": "str",
                    "call_to_action": "str",
                },
            },
            prompt="""Create a personalized cold outreach email for the following prospect:

        Name: {{ input.name }}
        Company: {{ input.company }}
        Role: {{ input.role }}
        Industry: {{ input.industry }}

        The email should:
        - Have a compelling subject line
        - Be brief (3-5 sentences)
        - Mention a specific pain point for their industry
        - Include a clear call to action
        - Sound natural and conversational, not sales-y

        Your response should be formatted as:
        Subject: [Your subject line]

        [Your email body]

        Call to action: [Your specific call to action]""",
        )
        frame.write_json("email_templates.json")
        ```

    With this configuration, if your prospects.json file has 50 prospects, the output will contain 500 email templates (50 prospects × 10 emails each).

!!! note "Important Considerations"

    - The `output.n` parameter is only available for OpenAI models
    - Higher values of `n` will increase the cost of your operation proportionally
    - For optimal results, keep your prompt focused and clear about generating diverse outputs
    - When using pandas, the `n` parameter can be passed directly to the `map` method
