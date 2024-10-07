import { File, Operation, SchemaItem } from '@/app/types';

export const mockFiles: File[] = [
  { name: 'debate_transcripts.json', path: '/Users/shreyashankar/Documents/hacking/motion-v3/website/public/debate_transcripts.json' },
];

export const initialOperations: Operation[] = [
  {
    id: '1',
    llmType: 'LLM',
    type: 'map',
    name: 'extract_themes',
    prompt: 'summarize the themes discussed in the debate. here is the transcript: \n {{ input.content }}',
    output: {
      schema: [
        { key: 'summary', type: 'string' }
      ]
    }
  },
  {
    id: '2',
    llmType: 'LLM',
    type: 'map',
    name: 'extract_candidates',
    prompt: 'Extract the candidates participating in the debate and their respective parties. Here is the transcript: \n {{ input.content }}',
    output: {
      schema: [
        { key: 'candidates', type: 'list', subType: { key: 'candidate', type: 'string' } }
      ]
    }
  },
];

export const mockSampleSize = 5;
export const mockPipelineName = 'Debate_Analysis';