import { NextResponse } from 'next/server';
import yaml from 'js-yaml';
import fs from 'fs/promises';
import path from 'path';
import axios from 'axios';
import os from 'os';
import { Operation, SchemaItem } from '@/app/types';

export async function POST(request: Request) {
  try {
    const { default_model, data, operations, operation_id, name, sample_size } = await request.json();


    if (!name) {
      return NextResponse.json({ error: 'Pipeline name is required' }, { status: 400 });
    }

    if (!data) {
      return NextResponse.json({ error: 'Data is required. Please select a file in the sidebar.' }, { status: 400 });
    }

    // Create pipeline configuration based on tutorial.yaml example
    const homeDir = os.homedir();

    const datasets = {
        input: {
            type: 'file',
            path: data.path,
            source: 'local'
        }
    };

    // Augment the first operation with sample if sampleSize is not null
    if (operations.length > 0 && sample_size !== null) {
      operations[0] = {
        ...operations[0],
        sample: sample_size
      };
    }

    // Fix the output schema of all operations to ensure correct typing
    const updatedOperations: Record<string, string> = operations.map((op: Operation) => {

        // Let new op be a dictionary representation of the operation
        let newOp: Record<string, any> = {
            ...op,
            ...op.otherKwargs
        }


      if (!op.output || !op.output.schema) return newOp;
      
      const processSchemaItem = (item: SchemaItem): string => {
        if (item.type === 'list') {
          if (!item.subType) {
            throw new Error(`List type must specify its elements for field: ${item.key}`);
          }
          const subType = typeof item.subType === 'string' ? item.subType : processSchemaItem(item.subType as SchemaItem);
          return `list[${subType}]`;
        } else if (item.type === 'dict' ) {
          if (!item.subType) {
            throw new Error(`Dict/Object type must specify its structure for field: ${item.key}`);
          }
          const subSchema = Object.entries(item.subType).reduce((acc, [key, value]) => {
            acc[key] = processSchemaItem(value as SchemaItem);
            return acc;
          }, {} as Record<string, string>);
          return JSON.stringify(subSchema);
        } else {
          return item.type;
        }
      };

      return {
        ...newOp,
        output: {
          schema: op.output.schema.reduce((acc: Record<string, string>, item: SchemaItem) => {
            acc[item.key] = processSchemaItem(item);
            return acc;
          }, {})
        }
      };
    });
            

    // Fetch all operations up until and including the operation_id
    const operationsToRun = operations.slice(0, operations.findIndex((op: Operation) => op.id === operation_id) + 1);

    const pipelineConfig = {
      datasets,
      default_model,
      operations: updatedOperations,
      pipeline: {
        steps: [
          {
            name: 'data_processing',
            input: Object.keys(datasets)[0], // Assuming the first dataset is the input
            operations: operationsToRun.map((op: any) => op.name)
          }
        ],
        output: {
          type: 'file',
          path: path.join(homeDir, '.docetl', 'pipelines', 'outputs', `${name}.json`),
          intermediate_dir: path.join(homeDir, '.docetl', 'pipelines', name,'intermediates')
        }
      }
    };

    // Get the inputPath from the intermediate_dir
    let inputPath;
    const prevOpIndex = operationsToRun.length - 2;

    if (prevOpIndex >= 0) {
      const inputBase = pipelineConfig.pipeline.output.intermediate_dir;
      const opName = operationsToRun[prevOpIndex].name;
      inputPath = path.join(inputBase, "data_processing", opName + '.json');
    } else {
      // If there are no previous operations, use the dataset path
      inputPath = data.path;
    }
    const yamlString = yaml.dump(pipelineConfig);

    console.log(yamlString);

    // Save the YAML file in the user's home directory
    const pipelineDir = path.join(homeDir, '.docetl', 'pipelines', 'configs');
    await fs.mkdir(pipelineDir, { recursive: true });
    const filePath = path.join(pipelineDir, `${name}.yaml`);
    await fs.writeFile(filePath, yamlString, 'utf8');

    // Submit the YAML config to the FastAPI endpoint
    const response = await axios.post('http://localhost:8000/run_pipeline', {
      yaml_config: filePath
    });

    return NextResponse.json({ 
      message: 'Pipeline YAML created and submitted successfully', 
      filePath,
      apiResponse: response.data,
      outputPath: pipelineConfig.pipeline.output.path,
      inputPath: inputPath
    });
  } catch (error) {
    let errorMessage;
    if (error instanceof axios.AxiosError && error.response && error.response.data) {
      errorMessage = error.response.data.detail || String(error);
    } else if (error instanceof Error) {
      errorMessage = error.message;
    } else {
      errorMessage = String(error);
    }
    return NextResponse.json({ error: `Failed to create or submit pipeline YAML: ${errorMessage}` }, { status: 500 });
  }
}
