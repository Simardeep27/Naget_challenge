import type {
  InformationAgentResponse,
  ResearchProgressEntry,
  ResearchMethod,
  SourceCitation,
  StructuredEntityTable,
  TableCell,
} from "@/lib/types";

interface ResultsPanelProps {
  activeMethod: ResearchMethod;
  isLoading: boolean;
  loadingLabel: string;
  progressEntries: ResearchProgressEntry[];
  recursiveResearchSelected?: boolean;
  response: InformationAgentResponse | null;
}

function formatDuration(milliseconds?: number): string {
  if (!milliseconds || milliseconds < 1000) {
    return milliseconds ? `${milliseconds} ms` : "Pending";
  }

  const seconds = milliseconds / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function resolveMethodLabel(
  activeMethod: ResearchMethod,
  response: InformationAgentResponse | null,
): string {
  if (!response) {
    if (activeMethod === "deep") {
      return "deep research";
    }

    if (activeMethod === "standard") {
      return "standard research";
    }

    return "lightning";
  }

  if (response.meta.mode === "lightning") {
    return "lightning";
  }

  return response.meta.deep_research ? "deep research" : "standard research";
}

function renderCitationSummary(citations: SourceCitation[]): string {
  if (!citations.length) {
    return "";
  }

  const ids = citations.slice(0, 2).map((citation) => citation.source_id).join(", ");
  const suffix = citations.length > 2 ? ` +${citations.length - 2}` : "";
  return `${citations.length} source${citations.length === 1 ? "" : "s"} · ${ids}${suffix}`;
}

function renderCell(cell: TableCell | undefined) {
  if (!cell?.value) {
    return <span className="tableCellEmpty">Not available</span>;
  }

  return (
    <div className="tableCellValue">
      <span>{cell.value}</span>
      {cell.citations.length > 0 ? (
        <span className="tableCellMeta">{renderCitationSummary(cell.citations)}</span>
      ) : null}
    </div>
  );
}

function renderTable(result: StructuredEntityTable) {
  if (result.rows.length === 0) {
    return (
      <div className="emptyCard">
        <h3>No entities returned yet</h3>
        <p>Run a search to see the entity table, sources, and execution trail here.</p>
      </div>
    );
  }

  return (
    <div className="tableShell">
      <table className="entityTable">
        <thead>
          <tr>
            {result.columns.map((column) => (
              <th key={column.key}>
                <div>{column.label}</div>
                <span>{column.description}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.rows.map((row) => (
            <tr key={row.entity_id}>
              {result.columns.map((column) => (
                <td key={`${row.entity_id}-${column.key}`}>
                  {renderCell(row.cells[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderLoadingProgress(
  progressEntries: ResearchProgressEntry[],
  loadingLabel: string,
) {
  const visibleEntries =
    progressEntries.length > 0
      ? progressEntries.slice(-8)
      : [
          {
            id: "progress-pending",
            message: loadingLabel,
          },
        ];
  const currentEntry = visibleEntries[visibleEntries.length - 1];

  return (
    <div aria-live="polite" className="loadingShell" role="status">
      <div className="loadingHeader">
        <div className="loadingHeaderCopy">
          <div className="loadingIndicator">
            <span aria-hidden="true" className="loadingSpinner" />
            <span>Search in progress</span>
          </div>
          <p className="eyebrow">Live Progress</p>
          <h3 className="loadingTitle">{currentEntry.message}</h3>
        </div>
        <span className="loadingCount">
          {progressEntries.length || 1} step{(progressEntries.length || 1) === 1 ? "" : "s"}
        </span>
      </div>

      <div className="progressTimeline">
        {visibleEntries.map((entry, index) => {
          const isActive = index === visibleEntries.length - 1;

          return (
            <article className={`progressEntry ${isActive ? "active" : ""}`} key={entry.id}>
              <div className="progressDot" />
              <p>{entry.message}</p>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function ResultsPanel({
  activeMethod,
  isLoading,
  loadingLabel,
  progressEntries,
  recursiveResearchSelected = false,
  response,
}: ResultsPanelProps) {
  const result = response?.result ?? null;
  const meta = response?.meta;
  const queryTrail = meta?.queries_run ?? [];
  const recursiveEnabled =
    Boolean(meta?.recursive_research?.enabled) || (isLoading && recursiveResearchSelected);

  return (
    <section className="resultsSurface">
      <div className="resultsHeader">
        <div>
          <p className="eyebrow">Results</p>
          <h2>
            {result?.title ?? "Your research run will land here as a structured table."}
          </h2>
          <p className="supportingText">
            {isLoading
              ? loadingLabel
              : result?.query ??
                "Pick a method, submit a search, and the response will render with sources and metadata."}
          </p>
        </div>
        <div className="resultsBadges">
          <span className="pillBadge">{resolveMethodLabel(activeMethod, response)}</span>
          {recursiveEnabled ? <span className="pillBadge">recursive research</span> : null}
          {response ? (
            <span className={`pillBadge ${response.status === "success" ? "success" : "error"}`}>
              {response.status}
            </span>
          ) : null}
        </div>
      </div>

      {isLoading ? renderLoadingProgress(progressEntries, loadingLabel) : null}

      <div className="statsGrid">
        <article className="statCard">
          <span>Execution time</span>
          <strong>{formatDuration(meta?.execution_time_ms)}</strong>
        </article>
        <article className="statCard">
          <span>Search queries</span>
          <strong>{queryTrail.length}</strong>
        </article>
        <article className="statCard">
          <span>Fetch calls</span>
          <strong>
            {meta?.fetch_calls ?? 0}
            {meta?.fetch_limit ? ` / ${meta.fetch_limit}` : ""}
          </strong>
        </article>
        <article className="statCard">
          <span>Model</span>
          <strong>{meta?.model ?? "Pending"}</strong>
        </article>
      </div>

      <div className="resultsGrid">
        <div className="resultsColumn wide">
          {result ? renderTable(result) : null}
        </div>

        <aside className="resultsColumn">
          <div className="stackCard">
            <div className="stackCardHeader">
              <h3>Search Path</h3>
              <span>{queryTrail.length}</span>
            </div>
            {queryTrail.length ? (
              <div className="traceList">
                {queryTrail.map((item, index) => (
                  <article className="traceItem" key={`${item.query}-${index}`}>
                    <div className="traceIndex">{index + 1}</div>
                    <div>
                      <h4>{item.query}</h4>
                      <p>
                        {item.results} result{item.results === 1 ? "" : "s"}
                      </p>
                      {item.error ? <span className="inlineError">{item.error}</span> : null}
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="stackEmpty">The executed search trail will appear here.</p>
            )}
          </div>

          <div className="stackCard">
            <div className="stackCardHeader">
              <h3>Sources</h3>
              <span>{result?.sources.length ?? 0}</span>
            </div>
            {result?.sources.length ? (
              <div className="sourceList">
                {result.sources.map((source) => (
                  <a
                    className="sourceCard"
                    href={source.url}
                    key={source.source_id}
                    rel="noreferrer"
                    target="_blank"
                  >
                    <div className="sourceMeta">
                      <span>{source.source_id}</span>
                      <strong>{source.title}</strong>
                    </div>
                    {source.snippet ? <p>{source.snippet}</p> : null}
                  </a>
                ))}
              </div>
            ) : (
              <p className="stackEmpty">Fetched sources and supporting snippets will appear here.</p>
            )}
          </div>

          <div className="stackCard">
            <div className="stackCardHeader">
              <h3>Artifacts</h3>
              <span>files</span>
            </div>
            {response ? (
              <div className="artifactList">
                <code>{response.json_file_path}</code>
                <code>{response.markdown_file_path}</code>
              </div>
            ) : (
              <p className="stackEmpty">JSON and markdown output paths will appear after a run.</p>
            )}
          </div>
        </aside>
      </div>
    </section>
  );
}
