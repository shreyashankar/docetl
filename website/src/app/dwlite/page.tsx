"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Sparkles,
  Upload,
  Loader2,
  Wand2,
  RefreshCw,
  FileText,
  BarChart3,
  Settings,
    History,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { BookmarkProvider } from "@/contexts/BookmarkContext";
import BookmarksPanel from "@/components/BookmarksPanel";
import ResizableDataTable, {
  ColumnType,
} from "@/components/ResizableDataTable";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import type { File as DocFile } from "@/app/types";

type DesiredColumn = {
  name: string;
  type: string;
  description?: string;
};

type PlannerUsage = {
  promptTokens?: number;
  completionTokens?: number;
};

type OutputPreview = {
  totalRows: number;
  preview: Record<string, unknown>[];
};

type PipelineRunResult = {
  cost?: number;
  message?: string;
  [key: string]: unknown;
};

type DwliteResponse = {
  status: "ok";
  pipelineName: string;
  summary: string;
  pipelineYaml: string;
  outputPath: string;
  appliedSampleOperation: string | null;
  runResult: PipelineRunResult;
  outputPreview: OutputPreview;
  plannerUsage?: PlannerUsage;
};

type ColumnTypeOption =
  | "string"
  | "int"
  | "float"
  | "boolean"
  | "list"
  | "dict"
  | "enum";

const COLUMN_TYPE_OPTIONS: { label: string; value: ColumnTypeOption }[] = [
  { label: "Text", value: "string" },
  { label: "Integer", value: "int" },
  { label: "Float", value: "float" },
  { label: "Boolean", value: "boolean" },
  { label: "List", value: "list" },
  { label: "Dictionary", value: "dict" },
  { label: "Enum", value: "enum" },
];

const DEFAULT_SAMPLE_SIZE = 5;

const sanitizeColumns = (columns: DesiredColumn[]): DesiredColumn[] =>
  columns
    .map((column) => ({
      name: column.name.trim(),
      type: column.type.trim() as ColumnTypeOption,
      description: column.description?.trim(),
    }))
    .filter((column) => column.name && column.type);

const createNamespace = () =>
  `dwlite-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const INITIAL_COLUMN: DesiredColumn = {
  name: "",
  type: "string",
  description: "",
};

const buildColumns = (
  data: Record<string, unknown>[]
): ColumnType<Record<string, unknown>>[] => {
  if (!data || data.length === 0) return [];

  const keys = Array.from(
    data.reduce((acc, row) => {
      Object.keys(row).forEach((key) => acc.add(key));
      return acc;
    }, new Set<string>())
  );

  return keys.map((key) => ({
    accessorKey: key,
    header: key,
    cell: ({ getValue }) => {
      const value = getValue();
      if (value === null || value === undefined) {
        return <span className="text-muted-foreground">—</span>;
      }
      if (typeof value === "object") {
        return (
          <pre className="whitespace-pre-wrap text-xs leading-5 font-mono">
            {JSON.stringify(value, null, 2)}
          </pre>
        );
      }
      return (
        <span className="text-sm leading-5 whitespace-pre-wrap">
          {String(value)}
        </span>
      );
    },
    initialWidth: 220,
  }));
};

const formatCost = (value: number | undefined) => {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  if (value < 0.01) return "< $0.01";
  return `$${value.toFixed(2)}`;
};

const formatTokens = (usage?: PlannerUsage) => {
  if (!usage) return "n/a";
  const { promptTokens = 0, completionTokens = 0 } = usage;
  if (!promptTokens && !completionTokens) return "n/a";
  return `${promptTokens + completionTokens} tokens`;
};

const SectionLabel: React.FC<{ title: string; icon?: React.ReactNode }> = ({
  title,
  icon,
}) => (
  <div className="flex items-center gap-2 text-xs font-semibold tracking-wide uppercase text-slate-400">
    {icon}
    <span>{title}</span>
  </div>
);

function DocWranglerLitePage() {
  const [namespace] = useState<string>(createNamespace);
  const [datasetFile, setDatasetFile] = useState<DocFile | null>(null);
  const [, setCurrentFile] = useState<DocFile | null>(null);
  const [specification, setSpecification] = useState("");
  const [desiredColumns, setDesiredColumns] = useState<DesiredColumn[]>([
    INITIAL_COLUMN,
  ]);
  const [runOnSample, setRunOnSample] = useState(true);
  const [sampleSize, setSampleSize] = useState<number>(DEFAULT_SAMPLE_SIZE);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DwliteResponse | null>(null);
  const [runHistory, setRunHistory] = useState<DwliteResponse[]>([]);
  const [feedbackHistory, setFeedbackHistory] = useState<string[]>([]);
  const [showPipelineDialog, setShowPipelineDialog] = useState(false);
  const [showRerunDialog, setShowRerunDialog] = useState(false);
  const [rerunFeedback, setRerunFeedback] = useState("");
  const [isCheckingNamespace, setIsCheckingNamespace] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const { toast } = useToast();

  const { uploadLocalDataset, uploadingFiles } = useDatasetUpload({
    namespace,
    onFileUpload: (file) => {
      setDatasetFile(file);
      setCurrentFile(file);
      toast({
        title: "Dataset uploaded",
        description: `${file.name} is ready to use.`,
      });
    },
    setCurrentFile,
  });

  useEffect(() => {
    const ensureNamespace = async () => {
      try {
        const response = await fetch("/api/checkNamespace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ namespace }),
        });
        if (!response.ok) {
          throw new Error("Failed to initialize workspace");
        }
      } catch (err) {
        console.error(err);
        toast({
          title: "Namespace initialization failed",
          description:
            "We could not prepare storage for this session. Please refresh and try again.",
          variant: "destructive",
        });
      } finally {
        setIsCheckingNamespace(false);
      }
    };

    ensureNamespace();
  }, [namespace, toast]);

  const handleAddColumn = () => {
    setDesiredColumns((prev) => [...prev, { ...INITIAL_COLUMN }]);
  };

  const handleRemoveColumn = (index: number) => {
    setDesiredColumns((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleUpdateColumn = (
    index: number,
    field: keyof DesiredColumn,
    value: string
  ) => {
    setDesiredColumns((prev) =>
      prev.map((column, idx) =>
        idx === index ? { ...column, [field]: value } : column
      )
    );
  };

  const handleDatasetInputChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const docFile: DocFile = {
        name: file.name,
        path: file.name,
        type: "json",
        blob: file,
      };

      await uploadLocalDataset(docFile);
      event.target.value = "";
    } catch (err) {
      console.error(err);
      toast({
        title: "Upload failed",
        description:
          err instanceof Error ? err.message : "Could not upload dataset.",
        variant: "destructive",
      });
    }
  };

  const columnsForTable = useMemo(() => {
    if (!result?.outputPreview?.preview) return [];
    return buildColumns(result.outputPreview.preview);
  }, [result]);

  const boldedColumns = useMemo(
    () => sanitizeColumns(desiredColumns).map((column) => column.name),
    [desiredColumns]
  );

  const handleSubmit = useCallback(
    async (additionalFeedback?: string): Promise<boolean> => {
      if (!datasetFile) {
        toast({
          title: "Dataset required",
          description: "Please upload a dataset before running DocWrangler Lite.",
          variant: "destructive",
        });
        setError("Dataset is required.");
        return false;
      }

      if (!specification.trim()) {
        toast({
          title: "Describe your task",
          description:
            "Tell DocWrangler Lite what you want to build in natural language.",
          variant: "destructive",
        });
        setError("Task description cannot be empty.");
        return false;
      }

      const cleanedColumns = sanitizeColumns(desiredColumns);
      if (!cleanedColumns.length) {
        toast({
          title: "Output columns required",
          description:
            "Specify at least one output column so we can design the pipeline.",
          variant: "destructive",
        });
        setError("At least one output column is required.");
        return false;
      }

      if (runOnSample && (!sampleSize || sampleSize < 1)) {
        toast({
          title: "Sample size invalid",
          description: "Sample size must be at least 1.",
          variant: "destructive",
        });
        setError("Sample size must be a positive integer.");
        return false;
      }

      const pendingFeedback = additionalFeedback?.trim();
      const nextFeedbackHistory = pendingFeedback
        ? [...feedbackHistory, pendingFeedback]
        : [...feedbackHistory];

      setIsSubmitting(true);
      setError(null);

      try {
        const response = await fetch("/api/dwlite", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            namespace,
            dataset: {
              path: datasetFile.path,
              name: datasetFile.name,
            },
            specification: specification.trim(),
            desiredColumns: cleanedColumns,
            runSample: runOnSample,
            sampleSize,
            feedback: nextFeedbackHistory,
            previousPipelineYaml: result?.pipelineYaml,
            previousSummary: result?.summary,
            sessionId: namespace,
          }),
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || payload.error || "Request failed");
        }

        const data = (await response.json()) as DwliteResponse;
        setResult(data);
        setRunHistory((prev) => [...prev, data]);
        setFeedbackHistory(nextFeedbackHistory);
        toast({
          title: "Pipeline executed",
          description: `DocWrangler Lite produced ${data.outputPreview.totalRows} rows.`,
        });
        return true;
      } catch (err) {
        console.error(err);
        const message =
          err instanceof Error ? err.message : "Failed to run pipeline.";
        setError(message);
        toast({
          title: "Execution failed",
          description: message,
          variant: "destructive",
        });
        return false;
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      datasetFile,
      desiredColumns,
      feedbackHistory,
      namespace,
      result?.pipelineYaml,
      result?.summary,
      runOnSample,
      sampleSize,
      specification,
      toast,
    ]
  );

  const handleInitialSubmit = async () => {
    const success = await handleSubmit();
    if (success) {
      setShowSettings(false);
    }
  };

  const handleRerun = async () => {
    const success = await handleSubmit(rerunFeedback);
    if (success) {
      setRerunFeedback("");
      setShowRerunDialog(false);
    }
  };

  const resetSession = () => {
    setResult(null);
    setRunHistory([]);
    setFeedbackHistory([]);
    setShowSettings(false);
    toast({
      title: "Session reset",
      description: "You can start a brand new pipeline exploration.",
    });
  };

  const renderDesiredColumnsForm = () => (
    <div className="space-y-4">
      {desiredColumns.map((column, index) => (
        <Card key={index} className="bg-slate-900/50 border-slate-800">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-200">
                Column {index + 1}
              </div>
              {desiredColumns.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-slate-400 hover:text-red-400"
                  onClick={() => handleRemoveColumn(index)}
                >
                  Remove
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="grid gap-3">
            <div className="grid md:grid-cols-2 gap-3">
              <div>
                <Label htmlFor={`column-name-${index}`} className="text-xs">
                  Column name
                </Label>
                <Input
                  id={`column-name-${index}`}
                  value={column.name}
                  onChange={(event) =>
                    handleUpdateColumn(index, "name", event.target.value)
                  }
                  placeholder="e.g. summary, sentiment"
                  className="bg-slate-950 border-slate-800 text-slate-100"
                />
              </div>
              <div>
                <Label htmlFor={`column-type-${index}`} className="text-xs">
                  Type
                </Label>
                <select
                  id={`column-type-${index}`}
                  value={column.type}
                  onChange={(event) =>
                    handleUpdateColumn(index, "type", event.target.value)
                  }
                  className="w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                >
                  {COLUMN_TYPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div>
              <Label htmlFor={`column-description-${index}`} className="text-xs">
                Description
              </Label>
              <Textarea
                id={`column-description-${index}`}
                value={column.description}
                onChange={(event) =>
                  handleUpdateColumn(index, "description", event.target.value)
                }
                placeholder="What should this field contain? Any hints for the model?"
                className="bg-slate-950 border-slate-800 text-slate-100 min-h-[80px]"
              />
            </div>
          </CardContent>
        </Card>
      ))}
      <Button
        type="button"
        variant="outline"
        className="border-dashed border-slate-700 text-slate-200 hover:text-slate-100"
        onClick={handleAddColumn}
      >
        Add another column
      </Button>
    </div>
  );

  const renderInitialView = () => (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex items-center justify-center px-6 py-16">
      <div className="max-w-6xl w-full grid lg:grid-cols-[1.15fr,0.85fr] gap-12">
        <div className="flex flex-col justify-center space-y-8">
          <Badge className="bg-slate-800 text-slate-200 w-fit px-4 py-1">
            DocWrangler Lite
          </Badge>
          <h1 className="text-4xl md:text-5xl font-semibold leading-tight text-slate-50">
            Build DocETL pipelines from natural language.
          </h1>
          <p className="text-lg text-slate-300 leading-relaxed">
            Upload your dataset, describe your goal, and DocWrangler Lite will
            draft, optimize, and run a DocETL pipeline for you—no coding or
            pipeline editing required.
          </p>
          <div className="flex items-center gap-3 text-sm text-slate-400">
            <Sparkles className="h-4 w-4 text-amber-400" />
            Powered by GPT-5 planning and the DocETL runtime
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-400">
            <BarChart3 className="h-4 w-4 text-emerald-400" />
            Bring your data to life: generate structured tables, insights, and
            summaries in minutes.
          </div>
        </div>
        <Card className="bg-slate-900/60 backdrop-blur border border-slate-800 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-xl text-slate-100">
              Your pipeline brief
            </CardTitle>
            <CardDescription className="text-slate-400">
              Upload your dataset, describe your task, and define the outputs
              you want.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label className="text-xs text-slate-300">Upload dataset</Label>
              <div className="border border-dashed border-slate-700 rounded-lg p-4 bg-slate-950/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-slate-900 flex items-center justify-center">
                      <Upload className="h-5 w-5 text-slate-200" />
                    </div>
                    <div className="text-sm text-slate-300">
                      {datasetFile ? (
                        <>
                          <div className="font-medium text-slate-100">
                            {datasetFile.name}
                          </div>
                          <div className="text-xs text-slate-400">
                            Stored in your DocWrangler Lite workspace
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="font-medium text-slate-100">
                            Drag a JSON/CSV file here
                          </div>
                          <div className="text-xs text-slate-400">
                            We&apos;ll convert it into DocETL-ready JSON.
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      type="file"
                      accept=".json,.csv"
                      className="hidden"
                      id="dataset-upload-input"
                      onChange={handleDatasetInputChange}
                      disabled={uploadingFiles.size > 0 || isSubmitting}
                    />
                    <Label
                      htmlFor="dataset-upload-input"
                      className={cn(
                        "px-4 py-2 rounded-md text-sm font-medium cursor-pointer bg-slate-800 text-slate-100 hover:bg-slate-700",
                        (uploadingFiles.size > 0 || isSubmitting) &&
                          "opacity-60 cursor-not-allowed"
                      )}
                    >
                      {uploadingFiles.size > 0 ? (
                        <div className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Uploading...
                        </div>
                      ) : datasetFile ? (
                        "Replace"
                      ) : (
                        "Browse"
                      )}
                    </Label>
                    {datasetFile && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-slate-300 hover:text-red-400"
                        onClick={() => {
                          setDatasetFile(null);
                          setCurrentFile(null);
                        }}
                      >
                        Remove
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="specification" className="text-xs text-slate-300">
                Describe the workflow you want
              </Label>
              <Textarea
                id="specification"
                value={specification}
                onChange={(event) => setSpecification(event.target.value)}
                placeholder="Example: Extract key issues from customer support tickets, tag each ticket with sentiment and priority, and produce a summary with action items."
                className="min-h-[160px] resize-none bg-slate-950/60 border-slate-800 text-slate-100"
              />
            </div>

            <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-950/50 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs text-slate-300">
                    Run on sample first?
                  </Label>
                  <p className="text-xs text-slate-500">
                    We&apos;ll add a sample operator so you can vet the prompts
                    faster.
                  </p>
                </div>
                <Switch
                  checked={runOnSample}
                  onCheckedChange={setRunOnSample}
                  disabled={isSubmitting}
                />
              </div>
              {runOnSample && (
                <div>
                  <Label htmlFor="sample-size" className="text-xs text-slate-300">
                    Sample size
                  </Label>
                  <Input
                    id="sample-size"
                    type="number"
                    min={1}
                    value={sampleSize}
                    onChange={(event) =>
                      setSampleSize(Number.parseInt(event.target.value, 10))
                    }
                    className="bg-slate-950 border-slate-800 text-slate-100 w-32"
                  />
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label className="text-xs text-slate-300">
                Desired output columns
              </Label>
              <Accordion type="single" collapsible defaultValue="columns">
                <AccordionItem value="columns">
                  <AccordionTrigger className="text-slate-200 hover:text-slate-100">
                    Define outputs
                  </AccordionTrigger>
                  <AccordionContent className="pt-4">
                    {renderDesiredColumnsForm()}
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>

            {error && (
              <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            )}
          </CardContent>
          <CardHeader className="pt-0">
            <Button
              onClick={handleInitialSubmit}
              disabled={isSubmitting || uploadingFiles.size > 0 || isCheckingNamespace}
              className="w-full bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-semibold text-sm py-6"
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  DocWrangler Lite is thinking...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Build my pipeline
                </span>
              )}
            </Button>
          </CardHeader>
        </Card>
      </div>
    </div>
  );

  const renderRunStats = () => {
    if (!result) return null;
    return (
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="py-4">
            <SectionLabel title="Rows produced" icon={<BarChart3 className="h-3 w-3" />} />
            <div className="text-2xl font-semibold text-slate-50">
              {result.outputPreview.totalRows.toLocaleString()}
            </div>
            <p className="text-xs text-slate-400">
              Sampled rows shown below (first {result.outputPreview.preview.length})
            </p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="py-4">
            <SectionLabel title="LLM spend" icon={<Sparkles className="h-3 w-3" />} />
            <div className="text-2xl font-semibold text-slate-50">
              {formatCost(result.runResult.cost)}
            </div>
            <p className="text-xs text-slate-400">
              {formatTokens(result.plannerUsage)}
            </p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="py-4">
            <SectionLabel title="Sample operator" icon={<Settings className="h-3 w-3" />} />
            <div className="text-sm text-slate-100">
              {result.appliedSampleOperation
                ? `Inserted ${result.appliedSampleOperation}`
                : "Disabled (full dataset)"}
            </div>
            <p className="text-xs text-slate-400">
              Toggle sampling in the rerun dialog.
            </p>
          </CardContent>
        </Card>
        <Card className="bg-slate-900/60 border-slate-800">
          <CardContent className="py-4">
            <SectionLabel title="Workspace" icon={<History className="h-3 w-3" />} />
            <div className="text-sm text-slate-100 break-all">
              {namespace}
            </div>
            <p className="text-xs text-slate-400">
              Files stored under ~/.docetl/{namespace}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderRunHistory = () => {
    if (runHistory.length <= 1) return null;

    return (
      <Card className="bg-slate-900/40 border-slate-800">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <SectionLabel title="Iteration history" icon={<History className="h-3 w-3" />} />
            <Badge variant="outline" className="border-slate-700 text-slate-300">
              {runHistory.length} runs
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {[...runHistory]
            .reverse()
            .slice(0, 4)
            .map((run, index) => (
              <div
                key={`${run.pipelineName}-${index}`}
                className="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
              >
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-slate-200">
                    {run.pipelineName}
                  </div>
                  <Badge className="bg-slate-800 text-slate-300">
                    {formatCost(run.runResult.cost)}
                  </Badge>
                </div>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                  {run.summary || "No summary provided."}
                </p>
              </div>
            ))}
        </CardContent>
      </Card>
    );
  };

  const renderResultsView = () => {
    if (!result) return null;

    return (
      <div className="min-h-screen bg-slate-950 text-slate-50">
        <div className="grid lg:grid-cols-[3fr,1fr] min-h-screen">
          <div className="flex flex-col">
            <header className="border-b border-slate-900/70 bg-slate-950/80 backdrop-blur sticky top-0 z-30">
              <div className="px-6 py-4 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-slate-400 text-xs uppercase tracking-wide">
                    <Sparkles className="h-3 w-3" />
                    DocWrangler Lite
                  </div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-semibold text-slate-100">
                      {result.pipelineName}
                    </h2>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs text-slate-300"
                      onClick={() => setShowPipelineDialog(true)}
                    >
                      <FileText className="h-3 w-3 mr-1" />
                      View YAML
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs text-slate-300"
                      onClick={() => setShowSettings((prev) => !prev)}
                    >
                      <Settings className="h-3 w-3 mr-1" />
                      {showSettings ? "Hide brief" : "Edit brief"}
                    </Button>
                  </div>
                  <p className="text-sm text-slate-400 max-w-2xl">
                    {result.summary || "Pipeline summary coming soon."}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-slate-300 hover:text-red-400"
                    onClick={resetSession}
                  >
                    Reset
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-slate-700 text-slate-200"
                    onClick={() => setShowSettings(true)}
                  >
                    Update brief
                  </Button>
                  <Button
                    size="sm"
                    className="bg-emerald-500 hover:bg-emerald-400 text-emerald-950"
                    onClick={() => setShowRerunDialog(true)}
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <RefreshCw className="h-3 w-3 mr-1" />
                        Rerun with feedback
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </header>
            <main className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
              {showSettings && (
                <Card className="bg-slate-900/40 border-slate-800">
                  <CardHeader className="pb-3">
                    <SectionLabel title="Current brief" icon={<Wand2 className="h-3 w-3" />} />
                    <CardDescription className="text-slate-300">
                      Update your instructions or target schema before rerunning.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-xs text-slate-300">
                        Task description
                      </Label>
                      <Textarea
                        value={specification}
                        onChange={(event) => setSpecification(event.target.value)}
                        className="min-h-[120px] bg-slate-950 border-slate-800 text-slate-100"
                      />
                    </div>
                    <Accordion type="single" collapsible defaultValue="schema">
                      <AccordionItem value="schema">
                        <AccordionTrigger className="text-sm text-slate-200">
                          Output schema ({boldedColumns.length} fields)
                        </AccordionTrigger>
                        <AccordionContent className="pt-3">
                          {renderDesiredColumnsForm()}
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                    <div className="flex items-center justify-between border-t border-slate-800 pt-3">
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <Switch
                          checked={runOnSample}
                          onCheckedChange={setRunOnSample}
                        />
                        <span>
                          {runOnSample
                            ? `Sampling first ${sampleSize} rows`
                            : "Full dataset"}
                        </span>
                      </div>
                      <Button
                        size="sm"
                        className="bg-slate-800 text-slate-200"
                        onClick={handleInitialSubmit}
                        disabled={isSubmitting}
                      >
                        {isSubmitting ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          "Rebuild pipeline"
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {renderRunStats()}
              {renderRunHistory()}

              <Card className="bg-slate-900/40 border-slate-800">
                <CardHeader className="pb-2">
                  <SectionLabel title="Output table" icon={<TableIcon />} />
                  <CardDescription className="text-slate-300">
                    Inspect results, leave notes, and iterate with feedback.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg border border-slate-800 bg-slate-950/40">
                    {columnsForTable.length ? (
                      <ResizableDataTable
                        data={result.outputPreview.preview}
                        columns={columnsForTable}
                        boldedColumns={boldedColumns}
                        startingRowHeight={200}
                        currentOperation="dwlite_output"
                      />
                    ) : (
                      <div className="p-8 text-center text-slate-400">
                        No output rows captured yet.
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </main>
          </div>
          <aside className="border-l border-slate-900 bg-slate-950/60 backdrop-blur hidden lg:flex">
            <BookmarksPanel />
          </aside>
        </div>
      </div>
    );
  };

  return (
    <BookmarkProvider>
      {result ? renderResultsView() : renderInitialView()}

      <Dialog open={showPipelineDialog} onOpenChange={setShowPipelineDialog}>
        <DialogContent className="max-w-3xl bg-slate-950 border-slate-800 text-slate-200">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-100">
              <FileText className="h-4 w-4" />
              Generated DocETL pipeline
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              Review the YAML configuration DocWrangler Lite generated. You can
              run it via the CLI or DocETL API if desired.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] overflow-y-auto rounded-lg border border-slate-800 bg-slate-900/80 p-4">
            <pre className="text-xs text-slate-100 leading-relaxed whitespace-pre-wrap">
              {result?.pipelineYaml}
            </pre>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              className="border-slate-700 text-slate-200"
              onClick={() => {
                if (!result) return;
                navigator.clipboard.writeText(result.pipelineYaml);
                toast({
                  title: "Pipeline copied",
                  description: "The YAML configuration is copied to clipboard.",
                });
              }}
            >
              Copy YAML
            </Button>
            <Button onClick={() => setShowPipelineDialog(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showRerunDialog} onOpenChange={setShowRerunDialog}>
        <DialogContent className="bg-slate-950 border-slate-800 text-slate-200">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-50">
              <Wand2 className="h-4 w-4 text-emerald-400" />
              Help DocWrangler Lite improve
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              Leave feedback describing what to change. We&apos;ll keep your
              existing brief, output schema, and previous feedback.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label className="text-xs text-slate-300">
                What should change next?
              </Label>
              <Textarea
                value={rerunFeedback}
                onChange={(event) => setRerunFeedback(event.target.value)}
                placeholder="Example: Ask for more context in the summary, always include ticket IDs, and filter out entries without sentiment."
                className="min-h-[140px] bg-slate-950 border-slate-800 text-slate-100"
              />
              <p className="text-xs text-slate-500 mt-2">
                We use GPT-5 to revise the pipeline, preserving previous runs and
                incorporating your aggregated feedback.
              </p>
            </div>
            <div className="flex items-center justify-between border border-slate-800 rounded-lg px-4 py-3 bg-slate-900/40">
              <div>
                <div className="text-sm font-medium text-slate-200">
                  Sampling is currently {runOnSample ? "enabled" : "disabled"}
                </div>
                <div className="text-xs text-slate-400">
                  Toggle sampling in the brief panel if you need different scale.
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(true)}
                className="border-slate-700 text-slate-200"
              >
                Adjust settings
              </Button>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="ghost"
              onClick={() => setShowRerunDialog(false)}
              className="text-slate-300"
            >
              Cancel
            </Button>
            <Button
              onClick={handleRerun}
              disabled={isSubmitting || rerunFeedback.trim().length === 0}
              className="bg-emerald-500 hover:bg-emerald-400 text-emerald-950"
            >
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Rerun with feedback
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </BookmarkProvider>
  );
}

const TableIcon = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="text-slate-400"
  >
    <rect
      x="3"
      y="5"
      width="18"
      height="14"
      rx="2"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <path
      d="M9 5V19"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <path
      d="M3 11H21"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

export default DocWranglerLitePage;
