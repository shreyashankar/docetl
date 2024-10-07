import { File } from '@/app/types';
import React, { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';

const DatasetView: React.FC<{ file: File | null }> = ({ file }) => {
  const [content, setContent] = useState<string | null>(null);
  const [keys, setKeys] = useState<string[]>([]);

  useEffect(() => {
    const loadFileContent = async () => {
      if (file?.path) {
        try {
          // Fetch file content from the server
          const response = await fetch(`/api/readFile?path=${encodeURIComponent(file.path)}`);
          if (!response.ok) {
            throw new Error('Failed to fetch file content');
          }
          const fileContent = await response.text();
          setContent(fileContent);
          const jsonContent = JSON.parse(fileContent);
          if (Array.isArray(jsonContent) && jsonContent.length > 0) {
            setKeys(Object.keys(jsonContent[0]));
          }
        } catch (error) {
          console.error('Error reading or parsing file:', error);
          setContent(null);
          setKeys([]);
        }
      }
    };

    loadFileContent();
  }, [file]);

  return (
    <div className="h-full p-4 bg-white overflow-x-auto">
      <h2 className="text-lg font-bold mb-2">{file?.name}</h2>
      <div className="mb-4">
        <p className="mb-2 font-semibold">Available keys:</p>
        {keys.map((key) => (
          <Badge key={key} className="mr-2 mb-2">{key}</Badge>
        ))}
      </div>
      <pre className="whitespace-pre-wrap">
        {content?.split('\n').map((line, index) => (
          <div key={index}>
            <span className="inline-block w-8 text-gray-500 select-none">{index + 1}</span>
            {line}
          </div>
        ))}
      </pre>
    </div>
  );
};

export default DatasetView;