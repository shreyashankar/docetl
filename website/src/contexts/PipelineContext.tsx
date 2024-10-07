import React, { createContext, useContext, useState } from 'react';
import { Operation, File, OutputType } from '@/app/types';
import { mockFiles, initialOperations, mockSampleSize, mockPipelineName } from '@/mocks/mockData';

interface PipelineContextType {
  operations: Operation[];
  currentFile: File | null;
  setOperations: React.Dispatch<React.SetStateAction<Operation[]>>;
  setCurrentFile: React.Dispatch<React.SetStateAction<File | null>>;
  output: OutputType | null;
  setOutput: React.Dispatch<React.SetStateAction<OutputType | null>>;
  isLoadingOutputs: boolean;
  setIsLoadingOutputs: React.Dispatch<React.SetStateAction<boolean>>;
  numOpRun: number;
  setNumOpRun: React.Dispatch<React.SetStateAction<number>>;
  pipelineName: string;
  setPipelineName: React.Dispatch<React.SetStateAction<string>>;
  sampleSize: number | null;
  setSampleSize: React.Dispatch<React.SetStateAction<number | null>>;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  cost: number;
  setCost: React.Dispatch<React.SetStateAction<number>>;
  defaultModel: string;
  setDefaultModel: React.Dispatch<React.SetStateAction<string>>;
}

const PipelineContext = createContext<PipelineContextType | undefined>(undefined);

export const PipelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [operations, setOperations] = useState<Operation[]>(initialOperations);
  const [currentFile, setCurrentFile] = useState<File | null>(mockFiles[0]);
  const [output, setOutput] = useState<OutputType | null>(null);
  const [isLoadingOutputs, setIsLoadingOutputs] = useState<boolean>(false);
  const [numOpRun, setNumOpRun] = useState<number>(0);
  const [pipelineName, setPipelineName] = useState<string>(mockPipelineName);
  const [sampleSize, setSampleSize] = useState<number | null>(mockSampleSize);
  const [files, setFiles] = useState<File[]>(mockFiles);
  const [cost, setCost] = useState<number>(0);
  const [defaultModel, setDefaultModel] = useState<string>("gpt-4o-mini");
  return (
    <PipelineContext.Provider value={{
      operations,
      currentFile,
      setOperations,
      setCurrentFile,
      output,
      setOutput,
      isLoadingOutputs,
      setIsLoadingOutputs,
      numOpRun,
      setNumOpRun,
      pipelineName,
      setPipelineName,
      sampleSize,
      setSampleSize,
      files,
      setFiles,
      cost,
      setCost,
      defaultModel,
      setDefaultModel,
    }}>
      {children}
    </PipelineContext.Provider>
  );
};

export const usePipelineContext = () => {
  const context = useContext(PipelineContext);
  if (context === undefined) {
    throw new Error('usePipelineContext must be used within a PipelineProvider');
  }
  return context;
};