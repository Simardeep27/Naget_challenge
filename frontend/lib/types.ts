export type ResearchMethod = "standard" | "deep" | "lightning";

export interface SourceCitation {
  source_id: string;
  source_title: string;
  source_url: string;
  quote: string;
}

export interface TableCell {
  value: string | null;
  citations: SourceCitation[];
}

export interface TableColumn {
  key: string;
  label: string;
  description: string;
}

export interface EntityRow {
  entity_id: string;
  cells: Record<string, TableCell>;
}

export interface SourceRecord {
  source_id: string;
  title: string;
  url: string;
  snippet: string | null;
}

export interface StructuredEntityTable {
  query: string;
  title: string;
  columns: TableColumn[];
  rows: EntityRow[];
  sources: SourceRecord[];
}

export interface QueryRun {
  query: string;
  results: number;
  error?: string;
}

export interface RecursiveResearchMeta {
  enabled?: boolean;
  skipped_for_mode?: boolean;
  rows_considered?: number;
  cells_filled?: number;
  rounds_attempted?: number;
}

export interface InformationAgentMeta {
  mode?: string;
  deep_research?: boolean;
  recursive_research?: RecursiveResearchMeta;
  queries_run?: QueryRun[];
  total_results?: number;
  fetch_calls?: number;
  fetch_limit?: number;
  execution_time_ms?: number;
  model?: string;
}

export interface InformationAgentResponse {
  status: string;
  json_file_path: string;
  markdown_file_path: string;
  result: StructuredEntityTable;
  meta: InformationAgentMeta;
}

export interface ResearchProgressEntry {
  id: string;
  message: string;
}

export type ResearchStreamEvent =
  | {
      type: "progress";
      message: string;
    }
  | {
      type: "result";
      data: InformationAgentResponse;
    }
  | {
      type: "error";
      message: string;
    }
  | {
      type: "heartbeat";
    }
  | {
      type: "done";
    };
