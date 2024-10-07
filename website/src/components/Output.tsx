import React, { useState, useEffect } from 'react';
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

export const Output: React.FC = () => {
  const { output, isLoadingOutputs, operations } = usePipelineContext();
  const [outputs, setOutputs] = useState<OutputRow[]>([]);
  const [inputCount, setInputCount] = useState<number>(0);
  const [outputCount, setOutputCount] = useState<number>(0);

  const operation = operations.find((op: Operation) => op.id === output?.operationId);
  const opName = operation?.name;

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
  
  const OutputContent = () => (
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
      <OutputContent />
    </div>
  );
};
