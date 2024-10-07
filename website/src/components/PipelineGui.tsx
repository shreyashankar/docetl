import React, { useState } from 'react';
import { DropResult } from 'react-beautiful-dnd';
import { Operation, File } from '@/app/types';
import { Droppable, DragDropContext } from 'react-beautiful-dnd';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { OperationCard } from '@/components/OperationCard';
import { Button } from '@/components/ui/button';
import { Plus, ChevronDown, Play, Settings, PieChart } from 'lucide-react';
import { usePipelineContext } from '@/contexts/PipelineContext';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useToast } from "@/hooks/use-toast"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const PipelineGUI: React.FC<{ 
  onDragEnd: (result: DropResult) => void;
}> = ({ onDragEnd }) => {
  const { operations, setOperations, pipelineName, setPipelineName, sampleSize, setSampleSize, numOpRun, setNumOpRun, currentFile, setCurrentFile, output, setOutput, isLoadingOutputs, setIsLoadingOutputs, files, setCost, defaultModel, setDefaultModel } = usePipelineContext();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [tempPipelineName, setTempPipelineName] = useState(pipelineName);
  const [tempSampleSize, setTempSampleSize] = useState(sampleSize?.toString() || '');
  const [tempCurrentFile, setTempCurrentFile] = useState<File | null>(currentFile);
  const [tempDefaultModel, setTempDefaultModel] = useState(defaultModel);
  const { toast } = useToast();

  const onRunAll = async () => {
    const lastOpIndex = operations.length - 1;
    if (lastOpIndex < 0) return;

    const lastOperation = operations[lastOpIndex];
    setIsLoadingOutputs(true);

    try {
      const updatedOperations = operations.map((op, index) => ({
        ...op,
        runIndex: numOpRun + index + 1
      }));

      const response = await fetch('/api/runPipeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          default_model: defaultModel,
          data: currentFile,
          operations: updatedOperations,
          operation_id: lastOperation.id,
          name: `${pipelineName}.yaml`,
          sample_size: sampleSize
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Failed to run pipeline: ${errorData.error || response.statusText}`);
      }

      const { apiResponse, outputPath, inputPath } = await response.json();
      const runCost = apiResponse.cost || 0;
      setCost(prevCost => prevCost + runCost);
      toast({
        title: "Pipeline Run Complete",
        description: `The pipeline run cost $${runCost.toFixed(4)}`,
        duration: 3000,
      });

      setOutput({
        path: outputPath,
        operationId: lastOperation.id,
        inputPath: inputPath
      });

      setNumOpRun(prevNum => prevNum + operations.length);
      setOperations(updatedOperations);
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : String(error),
        variant: "destructive",
      });
    } finally {
      setIsLoadingOutputs(false);
    }
  }

  const handleAddOperation = (llmType: 'LLM' | 'non-LLM', type: string, name: string) => {
    const newOperation: Operation = {
      id: String(Date.now()),
      llmType,
      type: type as Operation['type'],
      name: `${name} ${numOpRun}`,
    };
    setOperations([...operations, newOperation]);
  };

  const handleSettingsSave = () => {
    setPipelineName(tempPipelineName);
    setSampleSize(tempSampleSize === '' ? null : tempSampleSize === null ? null : parseInt(tempSampleSize, 10));
    setCurrentFile(tempCurrentFile);
    setDefaultModel(tempDefaultModel);
    setIsSettingsOpen(false);
  };

  return (
    <div className="h-full overflow-auto">
      <div className="sticky top-0 z-10 p-2 bg-white">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <h2 className="text-sm font-bold uppercase">{pipelineName.toUpperCase()}.yaml</h2>
            {sampleSize && (
              <TooltipProvider delayDuration={0}>
                <Tooltip>
                  <TooltipTrigger>
                    <div className="flex items-center">
                      <PieChart size={16} className="text-primary mr-1" />
                      <span className="text-xs text-primary">{sampleSize} samples</span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-[200px]">
                    <p>
                      Pipeline will run on a sample of {sampleSize} random documents.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
          <div className="flex space-x-2">
            <Button size="sm" variant="ghost" onClick={() => setIsSettingsOpen(true)}>
              <Settings size={16} />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" className="rounded-sm">
                  <Plus size={16} className="mr-2" /> Add Operation <ChevronDown size={16} className="ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>LLM Operations</DropdownMenuLabel>
                <DropdownMenuItem onClick={() => handleAddOperation('LLM', 'map', 'Untitled Map')}>Map</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('LLM', 'reduce', 'Untitled Reduce')}>Reduce</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('LLM', 'resolve', 'Untitled Resolve')}>Resolve</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('LLM', 'filter', 'Untitled Filter')}>Filter</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('LLM', 'parallel_map', 'Untitled Parallel Map')}>Parallel Map</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Non-LLM Operations</DropdownMenuLabel>
                <DropdownMenuItem onClick={() => handleAddOperation('non-LLM', 'unnest', 'Untitled Unnest')}>Unnest</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('non-LLM', 'split', 'Untitled Split')}>Split</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleAddOperation('non-LLM', 'gather', 'Untitled Gather')}>Gather</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button onClick={onRunAll} size="sm" className="rounded-sm" disabled={isLoadingOutputs}>
              <Play size={16} className="mr-2" /> Run All
            </Button>
          </div>
        </div>
      </div>
      <div className="p-2">
      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="operations">
          {(provided) => (
            <div {...provided.droppableProps} ref={provided.innerRef}>
              {operations.map((op, index) => (
                <OperationCard 
                  key={op.id} 
                  index={index} 
                />
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
      </div>
      <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Pipeline Settings</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={tempPipelineName}
                onChange={(e) => setTempPipelineName(e.target.value)}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="sampling" className="text-right">
                Sample Size
              </Label>
              <Input
                id="sampling"
                type="number"
                value={tempSampleSize}
                onChange={(e) => setTempSampleSize(e.target.value)}
                placeholder="None"
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="currentFile" className="text-right">
                Dataset JSON
              </Label>
              <Select
                value={tempCurrentFile?.path || ''}
                onValueChange={(value) => setTempCurrentFile(files.find(file => file.path === value) || null)}
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select a file" />
                </SelectTrigger>
                <SelectContent>
                  {files.map((file) => (
                    <SelectItem key={file.path} value={file.path}>
                      {file.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="defaultModel" className="text-right">
                Default Model
              </Label>
              <Input
                id="defaultModel"
                value={tempDefaultModel}
                onChange={(e) => setTempDefaultModel(e.target.value)}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button onClick={handleSettingsSave}>Save changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PipelineGUI;