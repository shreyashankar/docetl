'use client'

import React, { useState } from 'react';
import { FileText, Maximize2, Minimize2, Plus, Play, GripVertical, Trash2, ChevronDown, Zap, Upload, Scroll } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { DropResult } from 'react-beautiful-dnd';
import { OperationCard } from '@/components/OperationCard';
import SpotlightOverlay from '@/components/SpotlightOverlay';
import { Output } from '@/components/Output';
import { File, Operation } from '@/app/types';
import { FileExplorer } from '@/components/FileExplorer';
import { PipelineProvider, usePipelineContext } from '@/contexts/PipelineContext';
import DatasetView from '@/components/DatasetView';
import PipelineGUI from '@/components/PipelineGui';
import { useFileExplorer } from '@/hooks/useFileExplorer';
import { BookmarkProvider } from '@/contexts/BookmarkContext';
import BookmarksPanel from '@/components/BookmarksPanel';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"


const LeftPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <path 
      fill={isActive ? "currentColor" : "none"} 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M2 2.5h3.5v11H2z"
    />
    <path 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M5.5 2.5h8.5v11H5.5z"
    />
  </svg>
);

const BottomPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <path 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M2 2.5h12v7.5H2z"
    />
    <path 
      fill={isActive ? "currentColor" : "none"} 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M2 10h12v3.5H2z"
    />
  </svg>
);

const RightPanelIcon: React.FC<{ isActive: boolean }> = ({ isActive }) => (
  <svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <path 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M2 2.5h8.5v11H2z"
    />
    <path 
      fill={isActive ? "currentColor" : "none"} 
      stroke="currentColor" 
      strokeWidth="1.2" 
      d="M10.5 2.5h3.5v11h-3.5z"
    />
  </svg>
);


const CodeEditorPipelineApp: React.FC = () => {
  const [showFileExplorer, setShowFileExplorer] = useState(true);
  const [showOutput, setShowOutput] = useState(true);
  const [showDatasetView, setShowDatasetView] = useState(false);

  const { operations, currentFile, setOperations, setCurrentFile, cost } = usePipelineContext();
  const { files, handleFileClick, handleFileUpload, handleFileDelete } = useFileExplorer();

  const handleAddOperation = (llmType: string, type: string, name: string) => {
    const newOperation: Operation = {
      id: String(Date.now()),
      llmType: llmType as 'LLM' | 'non-LLM',
      type: type as 'map' | 'reduce' | 'filter' | 'resolve' | 'parallel_map' | 'unnest' | 'split' | 'gather',
      name: name,
    };
    setOperations([...operations, newOperation]);
  };

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;

    const items = Array.from(operations);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setOperations(items);
  };

  return (
    <BookmarkProvider>
    {/* <SpotlightOverlay> */}
    <div className="h-screen flex flex-col bg-gray-50">
      <div className="p-1 flex justify-between items-center border-b">
        <div className="flex-1"></div>
        <div className="flex items-center">
          <Scroll className="mr-2 text-primary" size={20} />
          <h1 className="text-lg font-bold text-primary">DocETL</h1>
        </div>
        <div className="flex-1 flex justify-end items-center space-x-2">
          <span className="text-sm font-medium text-gray-600">Cost: ${cost.toFixed(2)}</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowFileExplorer(!showFileExplorer)}
                  className="w-10 h-10"
                >
                  <LeftPanelIcon isActive={showFileExplorer} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle File Explorer</p>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowOutput(!showOutput)}
                  className="w-10 h-10"
                >
                  <BottomPanelIcon isActive={showOutput} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle Output Panel</p>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowDatasetView(!showDatasetView)}
                  className="w-10 h-10"
                >
                  <RightPanelIcon isActive={showDatasetView} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Toggle Dataset View</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
      {/* Main content */}
      <ResizablePanelGroup direction="horizontal" className="flex-grow">
          {/* File Explorer and Bookmarks */}
          {showFileExplorer && (
            <ResizablePanel defaultSize={10} minSize={6}>
              <ResizablePanelGroup direction="vertical">
                <ResizablePanel defaultSize={40} minSize={20}>
                  <FileExplorer 
                    files={files} 
                    onFileClick={(file) => {
                      handleFileClick(file);
                      setCurrentFile(file);
                    }} 
                    onFileUpload={handleFileUpload}
                    onFileDelete={handleFileDelete}
                    setCurrentFile={setCurrentFile}
                    setShowDatasetView={setShowDatasetView}
                    currentFile={currentFile}
                  />
                </ResizablePanel>
                <ResizableHandle withHandle className="h-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200" />
                <ResizablePanel defaultSize={60} minSize={20}>
                  <BookmarksPanel />
                </ResizablePanel>
              </ResizablePanelGroup>
            </ResizablePanel>
          )}
          {showFileExplorer && <ResizableHandle withHandle className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200" />}

          {/* Pipeline GUI and Output */}
          <ResizablePanel defaultSize={60} minSize={30}>
            <ResizablePanelGroup direction="vertical">
              <ResizablePanel defaultSize={70} minSize={30}>
                <PipelineGUI 
                  onDragEnd={handleDragEnd}
                />
              </ResizablePanel>
              {showOutput && <ResizableHandle withHandle className="h-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200" />}
              {showOutput && (
                <ResizablePanel defaultSize={105} minSize={20}>
                  <Output />
                </ResizablePanel>
              )}
            </ResizablePanelGroup>
          </ResizablePanel>

          {/* Dataset View */}
          {showDatasetView && <ResizableHandle withHandle className="w-2 bg-gray-200 hover:bg-gray-300 transition-colors duration-200" />}
          {showDatasetView && currentFile && (
            <ResizablePanel defaultSize={20} minSize={10}>
              <DatasetView file={currentFile} />
            </ResizablePanel>
          )}
        </ResizablePanelGroup>
      </div>
    {/* </SpotlightOverlay> */}
    </BookmarkProvider>
  );
};

const WrappedCodeEditorPipelineApp: React.FC = () => (
  <PipelineProvider>
    <CodeEditorPipelineApp />
  </PipelineProvider>
);

export default WrappedCodeEditorPipelineApp;