import React, { useState, useEffect, useMemo } from 'react';
import { ColumnType } from '@/components/ResizableDataTable';
import ResizableDataTable from '@/components/ResizableDataTable';
import { usePipelineContext } from '@/contexts/PipelineContext';
import { Loader2, Maximize2, Download, Columns } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "@/components/ui/dialog";
import BookmarkableText from '@/components/BookmarkableText';
import { Operation, OutputRow } from '@/app/types';
import { Parser } from 'json2csv';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export const Output: React.FC = () => {
  const { output, isLoadingOutputs, operations } = usePipelineContext();
  const [outputs, setOutputs] = useState<OutputRow[]>([]);
  const [inputCount, setInputCount] = useState<number>(0);
  const [outputCount, setOutputCount] = useState<number>(0);

  const [operation, setOperation] = useState<Operation | undefined>(undefined);
  const [opName, setOpName] = useState<string | undefined>(undefined);
  const [isResolveOrReduce, setIsResolveOrReduce] = useState<boolean>(false);

  useEffect(() => {
    const foundOperation = operations.find((op: Operation) => op.id === output?.operationId);
    setOperation(foundOperation);
    setOpName(foundOperation?.name);
    setIsResolveOrReduce(foundOperation?.type === 'resolve' || foundOperation?.type === 'reduce');
  }, [operations, output]);

  useEffect(() => {
    const fetchData = async () => {
      if (output) {
        const importantColumns = operation?.otherKwargs?.prompts?.[0]?.output_keys;
        try {
          // Fetch output data
          const outputResponse = await fetch(`/api/readFile?path=${output.path}`);
          if (!outputResponse.ok) {
            throw new Error('Failed to fetch output file');
          }
          const outputContent = await outputResponse.text();
          let parsedOutputs = JSON.parse(outputContent) as OutputRow[];
          setOutputCount(parsedOutputs.length);

          // Sort and reorder columns (existing logic)
          if (parsedOutputs.length > 0) {
            if ('date' in parsedOutputs[0]) {
              parsedOutputs.sort((a, b) => {
                const dateA = (a as any).date;
                const dateB = (b as any).date;
                if (dateA && dateB) {
                  return new Date(dateB).getTime() - new Date(dateA).getTime();
                }
                return 0;
              });
            }

            if (importantColumns && importantColumns.length > 0) {
              parsedOutputs = parsedOutputs.map(row => {
                const orderedRow: OutputRow = {};
                importantColumns.forEach((col: string) => {
                  if (col in row) {
                    orderedRow[col] = row[col];
                  }
                });
                Object.keys(row).forEach(key => {
                  if (!importantColumns.includes(key)) {
                    orderedRow[key] = row[key];
                  }
                });
                return orderedRow;
              });
            }
          }
          
          setOutputs(parsedOutputs);

          // Fetch input data if inputPath exists
          if (output.inputPath) {
            const inputResponse = await fetch(`/api/readFile?path=${output.inputPath}`);
            if (!inputResponse.ok) {
              throw new Error('Failed to fetch input file');
            }
            const inputContent = await inputResponse.text();
            const parsedInputs = JSON.parse(inputContent);
            setInputCount(Array.isArray(parsedInputs) ? parsedInputs.length : 1);
          }
          else {
            setInputCount(0);
          }
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      }
    };

    fetchData();
  }, [output, operation]);

  const columns: ColumnType<any>[] = React.useMemo(() => {
    const importantColumns = operation?.output?.schema ? Object.keys(operation.output.schema) : [];

    return outputs.length > 0
      ? Object.keys(outputs[0]).map((key) => ({
          accessorKey: key,
          header: key,
          cell: ({ getValue }: { getValue: () => any }) => {
            const value = getValue();
            const stringValue = typeof value === 'object' && value !== null ? JSON.stringify(value, null, 2) : String(value);
            return (
              <pre className="whitespace-pre-wrap font-mono text-sm">{stringValue}</pre>
            );
          },
          initialWidth: importantColumns?.includes(key) ? 300 : 150,
        }))
      : [];
  }, [outputs, operation?.output?.schema]);
  
  const TableContent = () => (
    <div className="flex-grow overflow-y-auto">
      {isLoadingOutputs ? (
        <div className="flex items-center justify-center h-full">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          <span className="ml-2 text-muted-foreground">Loading outputs...</span>
        </div>
      ) : outputs.length > 0 ? (
        <BookmarkableText source="output">
        <ResizableDataTable
          data={outputs}
          columns={columns}
          startingRowHeight={180}
        />
        </BookmarkableText>
      ) : (
        <p className="text-center text-muted-foreground">No outputs available.</p>
      )}
    </div>
  );

  const VisualizeContent = () => {

    const visualizationColumn = useMemo(() => {
      if (!opName) return null;
      const reduceColumnName = `_counts_prereduce_${opName}`;
      const resolveColumnName = `_kv_pairs_preresolve_${opName}`;
      return outputs.length > 0 && reduceColumnName in outputs[0] 
        ? { name: reduceColumnName, type: 'reduce' }
        : outputs.length > 0 && resolveColumnName in outputs[0]
        ? { name: resolveColumnName, type: 'resolve' }
        : null;
    }, [outputs, opName, operation]);

    if (!visualizationColumn || !operation) {
      return <p className="text-center text-muted-foreground">No visualization data available.</p>;
    }

    if (operation.type === 'reduce') {
      const reduceKeys = operation.otherKwargs?.reduce_key || [];
      return (
        <div className="flex-grow overflow-y-auto">
          {outputs
            .sort((a, b) => Number(b[visualizationColumn.name]) - Number(a[visualizationColumn.name]))
            .map((row, index) => (
            <div key={index} className="mb-2">
              <h3 className="text-sm font-semibold mb-2">
                {reduceKeys.map((key: string) => `${key}: ${row[key]}`).join(', ')} ({row[visualizationColumn.name]})
              </h3>
              <div className="flex">
                {Array.from({ length: Number(row[visualizationColumn.name]) }).map((_, i) => (
                  <div
                    key={i}
                    className="w-4 h-10 bg-primary mr-1"
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      );
    } else if (operation.type === 'resolve') {
      const groupedData = useMemo(() => {
        const intersectionKeys = new Set(
          outputs.flatMap(row => {
            const kvPairs = row[visualizationColumn.name];
            return Object.keys(kvPairs).filter(key => key in row);
          })
        );

        let groupedByIntersection: { [key: string]: any[] } = {};
        outputs.forEach(row => {
          const kvPairs = row[visualizationColumn.name];
          const key = Array.from(intersectionKeys)
            .map(k => row[k])
            .join('|');
          if (!groupedByIntersection[key]) {
            groupedByIntersection[key] = [];
          }
          groupedByIntersection[key].push({ row, oldValues: kvPairs });
        });

        return groupedByIntersection;
      }, [outputs, visualizationColumn.name]);

      return (
        <div className="flex-grow overflow-y-auto">
          {Object.entries(groupedData)
            .sort(([, groupA], [, groupB]) => groupB.length - groupA.length)
            .map(([key, group]: [string, any[]]) => (
            <div key={key} className="mb-2">
              <h3 className="text-sm font-semibold mb-2">{key} ({group.length})</h3>
              <div className="flex">
                {group.map((item, index) => (
                  <TooltipProvider key={index}>
                    <Tooltip delayDuration={0}>
                      <TooltipTrigger>
                        <Button
                          className="w-4 h-10 p-0 mr-1"
                          variant="default"
                        />
                      </TooltipTrigger>
                      <TooltipContent side="bottom" align="center" className="max-w-[300px]">
                        <p className="text-xs">Document values before resolution:</p>
                        <pre>{JSON.stringify(item.oldValues, null, 2)}</pre>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ))}
              </div>
            </div>
          ))}
        </div>
      );
    } else {
      return <p className="text-center text-muted-foreground">Visualization not supported for this operation type.</p>;
    }
  };

  const ConsoleContent = () => (
    <div className="flex-grow overflow-y-auto bg-black text-white p-4 font-mono rounded-lg">
      <p className="mb-2">Console output will be displayed here.</p>
      {/* Add actual console output logic here */}
    </div>
  );

  const downloadCSV = () => {
    if (outputs.length === 0) return;

    try {
      const parser = new Parser();
      const csvContent = parser.parse(outputs);

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'output.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (err) {
      console.error('Error converting to CSV:', err);
    }
  };

  const selectivityFactor = inputCount > 0 ? (outputCount / inputCount).toFixed(2) : 'N/A';

  return (
    <div className="h-full p-4 bg-white flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-sm font-bold mb-2 flex items-center uppercase">
          Output {opName && <span className="text-gray-500 ml-1">- {opName}</span>}
        </h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">
            {inputCount} inputs | {outputCount} outputs | Selectivity: {selectivityFactor}x
          </span>
          <Button variant="ghost" size="sm" className="p-1" onClick={downloadCSV} disabled={outputs.length === 0}>
            <Download size={16} />
          </Button>
          {/* <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="sm" className="p-1">
                <Maximize2 size={16} />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-[90vw] w-[90vw] max-h-[90vh] h-[90vh] flex flex-col pointer-events-auto bg-white">
            <div className="spotlight-overlay-target">
              <h2 className="text-lg font-bold mb-2">Output</h2>
              <OutputContent />
            </div>
            </DialogContent>
          </Dialog> */}
        </div>
      </div>
      <Tabs defaultValue="table" className="w-full">
        <TabsList>
          <TabsTrigger value="table">Table</TabsTrigger>
          <TabsTrigger value="console">Console</TabsTrigger>
          <TabsTrigger value="visualize" disabled={!isResolveOrReduce}>Visualize Input Distribution</TabsTrigger>
        </TabsList>
        <TabsContent value="table">
          <TableContent />
        </TabsContent>
        <TabsContent value="console">
          <ConsoleContent />
        </TabsContent>
        <TabsContent value="visualize">
          <div>
            <VisualizeContent />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
