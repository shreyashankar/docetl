import { useState, useEffect } from 'react';
import { File } from '@/app/types';
import { mockFiles } from '@/mocks/mockData';

export const useFileExplorer = () => {
  const [files, setFiles] = useState<File[]>(mockFiles);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileClick = (file: File) => {
    setSelectedFile(file);
  };

  const handleFileUpload = (file: File) => {
    setFiles(prevFiles => [...prevFiles, file]);
  };

  const handleFilesUpdate = (updatedFiles: File[]) => {
    setFiles(updatedFiles);
  };

  const handleFileDelete = (file: File) => {
    setFiles(prevFiles => prevFiles.filter(f => f.name !== file.name));
  };

  return {
    files,
    selectedFile,
    handleFileClick,
    handleFileUpload,
    handleFilesUpdate,
    handleFileDelete,
  };
};