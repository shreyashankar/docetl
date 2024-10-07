export type File = {
    name: string;
    path: string;
};

export type Operation = {
    id: string;
    llmType: 'LLM' | 'non-LLM';
    type: 'map' | 'reduce' | 'filter' | 'resolve' | 'parallel_map' | 'unnest' | 'split' | 'gather';
    name: string;
    prompt?: string;
    output?: {schema: SchemaItem[]};
    validate?: string[];
    otherKwargs?: Record<string, any>;
    runIndex?: number;
};

export type OutputRow = Record<string, string>;

export type SchemaType = 'string' | 'float' | 'int' | 'boolean' | 'list' | 'dict';

export interface SchemaItem {
  key: string;
  type: SchemaType;
  subType?: SchemaItem | SchemaItem[];
}

export interface UserNote {
    id: string;
    note: string;
}

export interface Bookmark {
    id: string;
    text: string;
    source: string;
    color: string;
    notes: UserNote[];
}
  
export interface BookmarkContextType {
    bookmarks: Bookmark[];
    addBookmark: (text: string, source: string, color: string, notes: UserNote[]) => void;
    removeBookmark: (id: string) => void;
}

export interface OutputType {
    path: string;
    operationId: string;
    inputPath?: string;
}
  