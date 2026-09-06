import { NextResponse } from "next/server";
import { generateText } from "ai";
import { createAzure } from "@ai-sdk/azure";
import yaml from "js-yaml";
import os from "os";
import path from "path";
import fs from "fs/promises";

const LLMS_INSTRUCTIONS_URL =
  process.env.DWLITE_LLMS_URL || "https://www.docetl.org/llms-full.txt";

const FASTAPI_URL = `${
  process.env.NEXT_PUBLIC_BACKEND_HTTPS ? "https" : "http"
}://${process.env.NEXT_PUBLIC_BACKEND_HOST}:${
  process.env.NEXT_PUBLIC_BACKEND_PORT
}`;

type DesiredColumn = {
  name: string;
  type: string;
  description?: string;
};

type PipelineOperation = {
  name?: string;
  type?: string;
  prompt?: string;
  [key: string]: unknown;
};

type PipelineStep = {
  name?: string;
  input?: string;
  operations?: string[];
};

type PipelineConfig = {
  name?: string;
  default_model?: string;
  datasets?: Record<string, { type?: string; path?: string; [key: string]: unknown }>;
  operations?: PipelineOperation[];
  pipeline?: {
    steps?: PipelineStep[];
    output?: { type?: string; path?: string; intermediate_dir?: string; [key: string]: unknown };
  };
  system_prompt?: Record<string, unknown>;
  [key: string]: unknown;
};

type DwliteRequest = {
  namespace: string;
  dataset: { path: string; name: string };
  specification: string;
  desiredColumns: DesiredColumn[];
  runSample: boolean;
  sampleSize?: number;
  feedback?: string[];
  previousPipelineYaml?: string;
  previousSummary?: string;
  sessionId?: string;
};

const MAX_SAMPLE_ROWS = 5;
const OUTPUT_PREVIEW_LIMIT = 50;

function extractTagContent(text: string, tag: string): string | null {
  const pattern = `<${tag}>([\\s\\S]*?)</${tag}>`;
  const regex = new RegExp(pattern, "i");
  const match = text.match(regex);

  if (!match || !match[1]) {
    return null;
  }

  return match[1].trim();
}

function sanitizePipelineName(rawName: string | undefined, fallback: string): string {
  const base = (rawName || fallback || "dwlite-pipeline").toLowerCase();
  const sanitized = base
    .replace(/[^a-z0-9-_]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);

  return sanitized || "dwlite-pipeline";
}

async function readDatasetSample(datasetPath: string): Promise<unknown[]> {
  try {
    const fileContent = await fs.readFile(datasetPath, "utf-8");
    const parsed = JSON.parse(fileContent);

    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed.slice(0, MAX_SAMPLE_ROWS);
    }

    return Array.isArray(parsed) ? parsed : [parsed];
  } catch (error) {
    console.error("Failed to read dataset sample:", error);
    return [];
  }
}

let cachedInstructions: string | null = null;

async function fetchPlannerInstructions(): Promise<string> {
  if (cachedInstructions) {
    return cachedInstructions;
  }

  try {
    const response = await fetch(LLMS_INSTRUCTIONS_URL, {
      headers: { "User-Agent": "DocWrangler-Lite/1.0" },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const text = await response.text();
    cachedInstructions = text;
    return text;
  } catch (error) {
    console.warn(
      "DocWrangler Lite: failed to load llms-full instructions – using fallback message.",
      error
    );
    return "DocETL instructions unavailable. Use best judgment based on provided context.";
  }
}

function formatSampleForPrompt(sample: unknown[]): string {
  if (!sample.length) {
    return "No sample data available.";
  }

  return sample
    .map((row, index) => {
      const serialized = JSON.stringify(row, null, 2);
      return `Row ${index + 1}:\n${serialized.slice(0, 1500)}`;
    })
    .join("\n\n");
}

function formatDesiredColumns(columns: DesiredColumn[]): string {
  if (!columns.length) {
    return "User did not specify desired output columns.";
  }

  return columns
    .map(
      (column) =>
        `- ${column.name} (${column.type})${
          column.description ? `: ${column.description}` : ""
        }`
    )
    .join("\n");
}

function formatFeedback(feedback: string[] | undefined): string {
  if (!feedback || feedback.length === 0) {
    return "No additional feedback provided.";
  }

  return feedback.map((item, idx) => `${idx + 1}. ${item}`).join("\n");
}

function hasValidJinja(prompt: string): boolean {
  if (!prompt.includes("{{") || !prompt.includes("}}")) {
    return false;
  }

  const jinjaBlocks = prompt.match(/{{[\s\S]*?}}/g) || [];
  return jinjaBlocks.some((block) => /input[s]?\./.test(block));
}

function validateJinjaTemplates(operations: PipelineOperation[]) {
  const failing = operations.filter(
    (op) =>
      typeof op.prompt === "string" &&
      op.type &&
      ["map", "reduce", "rank", "extract", "resolve", "gather", "split"].includes(op.type) &&
      !hasValidJinja(op.prompt)
  );

  if (failing.length > 0) {
    const names = failing
      .map((op) => op.name || op.type || "unknown operation")
      .join(", ");
    throw new Error(
      `One or more operations are missing Jinja references to the input document: ${names}`
    );
  }
}

function generateUniqueOperationName(
  baseName: string,
  existingOperations: PipelineOperation[]
): string {
  const existingNames = new Set(
    existingOperations.map((op) => (op.name || "").toLowerCase())
  );

  if (!existingNames.has(baseName.toLowerCase())) {
    return baseName;
  }

  let counter = 1;
  let candidate = `${baseName}-${counter}`;

  while (existingNames.has(candidate.toLowerCase())) {
    counter += 1;
    candidate = `${baseName}-${counter}`;
  }

  return candidate;
}

function ensureSampleOperation(
  config: PipelineConfig,
  sampleSize: number,
  datasetKey: string
): string {
  if (!Array.isArray(config.operations)) {
    config.operations = [];
  }

  const existingSample = config.operations.find(
    (op) => (op.type || "").toLowerCase() === "sample"
  );

  if (existingSample) {
    existingSample.name =
      existingSample.name ||
      generateUniqueOperationName("dwlite_sample", config.operations);
    existingSample.method = existingSample.method || "first";
    existingSample.samples = sampleSize;
    return existingSample.name as string;
  }

  const sampleName = generateUniqueOperationName("dwlite_sample", config.operations);
  const sampleOperation: PipelineOperation = {
    name: sampleName,
    type: "sample",
    method: "first",
    samples: sampleSize,
  };

  config.operations.unshift(sampleOperation);

  if (config.pipeline?.steps && config.pipeline.steps.length > 0) {
    const firstStep = config.pipeline.steps[0];
    const existingOps = firstStep.operations || [];
    const filteredOps = existingOps.filter((op) => op !== sampleName);
    firstStep.operations = [sampleName, ...filteredOps];
    firstStep.input = firstStep.input || datasetKey;
  } else if (config.pipeline) {
    config.pipeline.steps = [
      {
        name: "dwlite_step",
        operations: [sampleName],
        input: datasetKey,
      },
    ];
  } else {
    config.pipeline = {
      steps: [
        {
          name: "dwlite_step",
          operations: [sampleName],
          input: datasetKey,
        },
      ],
      output: {},
    };
  }

  return sampleName;
}

function ensureDatasetPath(config: PipelineConfig, datasetPath: string): string {
  if (!config.datasets || Object.keys(config.datasets).length === 0) {
    config.datasets = {
      input: {
        type: "file",
        path: datasetPath,
      },
    };
    return "input";
  }

  const datasetKey = Object.keys(config.datasets)[0];
  config.datasets[datasetKey] = {
    ...config.datasets[datasetKey],
    type: "file",
    path: datasetPath,
  };

  return datasetKey;
}

async function ensureOutputPaths(
  config: PipelineConfig,
  namespace: string,
  pipelineSlug: string
) {
  const homeDir = process.env.DOCETL_HOME_DIR || os.homedir();
  const pipelinesDir = path.join(homeDir, ".docetl", namespace, "pipelines");
  const outputPath = path.join(pipelinesDir, "outputs", `${pipelineSlug}.json`);
  const intermediateDir = path.join(
    pipelinesDir,
    pipelineSlug,
    "intermediates"
  );

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.mkdir(intermediateDir, { recursive: true });

  if (!config.pipeline) {
    config.pipeline = {};
  }

  config.pipeline.output = {
    ...(config.pipeline.output || {}),
    type: "file",
    path: outputPath,
    intermediate_dir: intermediateDir,
  };

  return { outputPath, intermediateDir };
}

async function buildPrompt(
  request: DwliteRequest,
  sampleRows: unknown[]
): Promise<string> {
  const llmInstructions = await fetchPlannerInstructions();
  const sampleText = formatSampleForPrompt(sampleRows);
  const desiredColumns = formatDesiredColumns(request.desiredColumns);
  const feedbackText = formatFeedback(request.feedback);
  const previousSummary = request.previousSummary
    ? `Previous summary of results:\n${request.previousSummary}`
    : "No previous summary available.";
  const previousPipeline = request.previousPipelineYaml
    ? `Previous pipeline YAML (for reference, improve upon shortcomings):\n${request.previousPipelineYaml}`
    : "No previous pipeline provided.";

  return `
You are designing a DocETL pipeline for DocWrangler Lite. Produce a concise, reliable plan that can be executed immediately.

Follow these requirements exactly:
- All LLM prompts must be Jinja templates that reference the input document using {{ input.key }} or iterate over {{ inputs }}. Never produce plain text prompts without templating.
- Favor transparent schemas in operation outputs. Ensure the final pipeline yields the requested output columns.
- Name datasets and operations using concise, descriptive snake_case.
- If the user wants to run on a sample, include a sample operation that limits execution to the specified number of documents.
- Never require the end user to edit code; everything must be configured declaratively.
- Keep the number of operations minimal but sufficient to achieve the task.
- Always include provenance fields or notes when applicable.

Context for this run:
- Dataset path: ${request.dataset.path}
- Dataset sample (first ${Math.min(
    sampleRows.length,
    MAX_SAMPLE_ROWS
  )} rows):\n${sampleText}
- Desired output columns:\n${desiredColumns}
- Should run on sample: ${request.runSample ? "yes" : "no"}${
    request.runSample ? ` (target size: ${request.sampleSize || 5})` : ""
  }
- User specification:\n${request.specification}
- Additional feedback to address:\n${feedbackText}
- ${previousSummary}
- ${previousPipeline}
- DocETL system description and detailed instructions:\n${llmInstructions}

Respond in the following XML-inspired format without Markdown code fences:
<name>Concise pipeline name</name>
<summary>Brief summary explaining how the pipeline achieves the goal</summary>
<pipeline>
[Valid DocETL YAML configuration with datasets, operations, and pipeline definition. Use "DATASET_PATH_PLACEHOLDER" for dataset paths and "OUTPUT_PATH_PLACEHOLDER"/"INTERMEDIATE_DIR_PLACEHOLDER" for pipeline output placeholders. Set default_model to azure/gpt-5 or a compatible family.]
</pipeline>
`.trim();
}

async function callPlanner(prompt: string) {
  const azure = createAzure({
    apiKey: process.env.AZURE_API_KEY!,
    apiVersion: process.env.AZURE_API_VERSION,
    resourceName: process.env.AZURE_RESOURCE_NAME,
  });

  const deployment =
    process.env.AZURE_DWLITE_DEPLOYMENT ||
    process.env.AZURE_DEPLOYMENT_NAME ||
    "gpt-4o-mini";

  const result = await generateText({
    model: azure(deployment),
    prompt,
  });

  return { text: result.text, usage: result.usage };
}

async function runPipeline(yamlConfig: string) {
  const response = await fetch(`${FASTAPI_URL}/run_pipeline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml_config: yamlConfig }),
  });

  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail || detail;
    } catch {
      // keep raw detail
    }

    throw new Error(
      `Pipeline execution failed with status ${response.status}: ${detail}`
    );
  }

  return response.json();
}

async function readOutputPreview(outputPath: string) {
  try {
    const content = await fs.readFile(outputPath, "utf-8");
    const parsed = JSON.parse(content);

    if (Array.isArray(parsed)) {
      return {
        totalRows: parsed.length,
        preview: parsed.slice(0, OUTPUT_PREVIEW_LIMIT),
      };
    }

    return {
      totalRows: 1,
      preview: [parsed],
    };
  } catch (error) {
    console.error("Failed to read pipeline output:", error);
    return {
      totalRows: 0,
      preview: [],
    };
  }
}

export async function POST(request: Request) {
  let body: DwliteRequest;

  try {
    body = await request.json();
  } catch (error) {
    return NextResponse.json(
      { error: "Invalid JSON payload", detail: error instanceof Error ? error.message : String(error) },
      { status: 400 }
    );
  }

  const {
    namespace,
    dataset,
    specification,
    desiredColumns,
    runSample,
    sampleSize = 5,
  } = body;

  if (!namespace || !dataset?.path || !specification) {
    return NextResponse.json(
      {
        error:
          "Missing required fields. Please provide namespace, dataset.path, and specification.",
      },
      { status: 400 }
    );
  }

  try {
    const datasetSample = await readDatasetSample(dataset.path);
    const prompt = await buildPrompt(body, datasetSample);
    const plannerResult = await callPlanner(prompt);

    const pipelineYaml = extractTagContent(plannerResult.text, "pipeline");
    if (!pipelineYaml) {
      throw new Error(
        "Failed to parse pipeline from planner response. Expected <pipeline> ... </pipeline> section."
      );
    }

    const pipelineName =
      extractTagContent(plannerResult.text, "name") || "dwlite-pipeline";
    const summary = extractTagContent(plannerResult.text, "summary") || "";

    const parsedConfig = yaml.load(
      pipelineYaml.replace(/OUTPUT_PATH_PLACEHOLDER/g, "output.json").replace(
        /INTERMEDIATE_DIR_PLACEHOLDER/g,
        "intermediate"
      )
    ) as PipelineConfig;

    if (!parsedConfig || typeof parsedConfig !== "object") {
      throw new Error("Planner returned an invalid pipeline configuration.");
    }

    if (!Array.isArray(parsedConfig.operations) || parsedConfig.operations.length === 0) {
      throw new Error("Pipeline must include at least one operation.");
    }

    validateJinjaTemplates(parsedConfig.operations);

    const datasetKey = ensureDatasetPath(parsedConfig, dataset.path);
    if (parsedConfig.pipeline?.steps && parsedConfig.pipeline.steps.length > 0) {
      parsedConfig.pipeline.steps[0].input =
        parsedConfig.pipeline.steps[0].input || datasetKey;
    }

    if (!parsedConfig.default_model) {
      parsedConfig.default_model =
        process.env.DWLITE_DEFAULT_MODEL || "azure/gpt-5-nano";
    }

    const appliedSampleName = runSample
      ? ensureSampleOperation(parsedConfig, sampleSize, datasetKey)
      : null;

    const sanitizedName = sanitizePipelineName(
      pipelineName,
      body.sessionId || `dwlite-${Date.now()}`
    );

    const { outputPath } = await ensureOutputPaths(
      parsedConfig,
      namespace,
      sanitizedName
    );

    const finalConfigYaml = yaml.dump(parsedConfig, {
      forceQuotes: false,
      noRefs: true,
    });

    const runResult = await runPipeline(finalConfigYaml);
    const outputPreview = await readOutputPreview(outputPath);

    return NextResponse.json({
      status: "ok",
      pipelineName: sanitizedName,
      summary,
      pipelineYaml: finalConfigYaml,
      outputPath,
      appliedSampleOperation: appliedSampleName,
      runResult,
      outputPreview,
      plannerUsage: plannerResult.usage,
    });
  } catch (error) {
    console.error("DocWrangler Lite pipeline generation failed:", error);
    return NextResponse.json(
      {
        error: "Failed to generate or execute pipeline",
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
