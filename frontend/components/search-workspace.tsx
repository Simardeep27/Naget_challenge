"use client";

import { startTransition, useEffect, useRef, useState } from "react";

import { ResultsPanel } from "@/components/results-panel";
import type {
  InformationAgentResponse,
  ResearchMethod,
  ResearchProgressEntry,
  ResearchStreamEvent,
} from "@/lib/types";

type SearchModeId = "deep" | "recursive";

const SEARCH_MODES: Array<{
  id: SearchModeId;
  label: string;
  description: string;
  method: ResearchMethod;
  recursiveResearch: boolean;
}> = [
  {
    id: "deep",
    label: "Deep research",
    description: "Broader sweep with more search and fetch passes.",
    method: "deep",
    recursiveResearch: false,
  },
  {
    id: "recursive",
    label: "Recursive research",
    description: "Deep research plus targeted follow-up to fill missing gaps.",
    method: "deep",
    recursiveResearch: true,
  },
];

const DEFAULT_MODE_ID: SearchModeId = "deep";

const API_BASE = process.env.NEXT_PUBLIC_AGENT_URL || "/api";

function getApiError(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  return "The search failed before a result could be returned.";
}

function SubmitIcon() {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      <path
        d="M7.75 12H16.25"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M12.75 8.5L16.25 12L12.75 15.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function LoaderIcon({ className }: { className: string }) {
  return <span aria-hidden="true" className={className} />;
}

function ChevronIcon() {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      <path
        d="M7.5 10.25L12 14.75L16.5 10.25"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      <path
        d="M6.75 12.5L10.25 16L17.25 9"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function parseResearchStreamEvent(rawLine: string): ResearchStreamEvent | null {
  const trimmedLine = rawLine.trim();
  if (!trimmedLine) {
    return null;
  }

  try {
    return JSON.parse(trimmedLine) as ResearchStreamEvent;
  } catch {
    return null;
  }
}

export function SearchWorkspace() {
  const [query, setQuery] = useState("");
  const [selectedModeId, setSelectedModeId] = useState<SearchModeId>(DEFAULT_MODE_ID);
  const [response, setResponse] = useState<InformationAgentResponse | null>(null);
  const [progressEntries, setProgressEntries] = useState<ResearchProgressEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModeMenuOpen, setIsModeMenuOpen] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const progressCountRef = useRef(0);
  const modeControlRef = useRef<HTMLDivElement | null>(null);

  const selectedMode =
    SEARCH_MODES.find((option) => option.id === selectedModeId) ?? SEARCH_MODES[0];
  const initialProgressMessage = selectedMode.recursiveResearch
    ? "Starting recursive research"
    : "Starting deep research";
  const currentProgress =
    progressEntries.at(-1)?.message ?? (isLoading ? initialProgressMessage : "Ready");

  useEffect(() => {
    if (!isModeMenuOpen) {
      return undefined;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!(event.target instanceof Node)) {
        return;
      }

      if (!modeControlRef.current?.contains(event.target)) {
        setIsModeMenuOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsModeMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isModeMenuOpen]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  function handleModeSelect(modeId: SearchModeId) {
    setSelectedModeId(modeId);
    setIsModeMenuOpen(false);
  }

  function appendProgressMessage(message: string) {
    const trimmedMessage = message.trim();
    if (!trimmedMessage) {
      return;
    }

    setProgressEntries((currentEntries) => {
      if (currentEntries.at(-1)?.message === trimmedMessage) {
        return currentEntries;
      }

      progressCountRef.current += 1;

      return [
        ...currentEntries,
        {
          id: `progress-${progressCountRef.current}`,
          message: trimmedMessage,
        },
      ];
    });
  }

  async function readResearchStream(streamResponse: Response) {
    const reader = streamResponse.body?.getReader();
    if (!reader) {
      const payload = (await streamResponse.json()) as InformationAgentResponse;
      startTransition(() => {
        setResponse(payload);
      });
      return payload;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let streamedResult: InformationAgentResponse | null = null;

    const processEvent = (event: ResearchStreamEvent) => {
      if (event.type === "heartbeat" || event.type === "done") {
        return;
      }

      if (event.type === "progress") {
        appendProgressMessage(event.message);
        return;
      }

      if (event.type === "error") {
        throw new Error(event.message);
      }

      streamedResult = event.data;
      startTransition(() => {
        setResponse(event.data);
      });
    };

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const event = parseResearchStreamEvent(line);
        if (event) {
          processEvent(event);
        }
      }

      if (done) {
        const trailingEvent = parseResearchStreamEvent(buffer);
        if (trailingEvent) {
          processEvent(trailingEvent);
        }
        break;
      }
    }

    if (!streamedResult) {
      throw new Error("The research stream ended before returning a result.");
    }

    return streamedResult;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError("Enter a search before you submit.");
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    progressCountRef.current = 1;
    setProgressEntries([
      {
        id: "progress-1",
        message: initialProgressMessage,
      },
    ]);
    setError(null);
    setIsLoading(true);
    setIsModeMenuOpen(false);
    setResponse(null);

    try {
      const request = await fetch(`${API_BASE}/research/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmedQuery,
          method: selectedMode.method,
          recursive_research: selectedMode.recursiveResearch,
        }),
        signal: controller.signal,
      });

      if (!request.ok) {
        const payload = (await request.json()) as { detail?: unknown; message?: unknown };
        throw new Error(getApiError(payload.detail ?? payload.message));
      }

      await readResearchStream(request);
    } catch (submitError) {
      if (submitError instanceof DOMException && submitError.name === "AbortError") {
        return;
      }

      setError(
        submitError instanceof Error ? submitError.message : "The search failed unexpectedly.",
      );
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsLoading(false);
    }
  }

  return (
    <main className="sooglePage">
      <header className="brandBar">
        <a
          aria-label="Soogle"
          className="brandWordmark"
          href="/"
          title="Simardeep's google"
        >
          <span className="brandText brandCollapsed">Soogle</span>
          <span className="brandText brandExpanded">Simardeep&apos;s google</span>
        </a>
      </header>

      <section className="heroStack">
        <p className="greeting">Good to see you Chris</p>

        <form className="searchPanel" onSubmit={handleSubmit}>
          <label className="srOnly" htmlFor="query">
            Search anything
          </label>
          <input
            autoComplete="off"
            className="searchInput"
            enterKeyHint="search"
            id="query"
            name="query"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search anything"
            spellCheck={false}
            type="search"
            value={query}
          />

          <div className="searchTools">
            <div className="searchMeta">
              <div className="modeControl" ref={modeControlRef}>
                <button
                  aria-expanded={isModeMenuOpen}
                  aria-haspopup="dialog"
                  className={`modeTrigger ${isModeMenuOpen ? "open" : ""}`}
                  disabled={isLoading}
                  onClick={() => setIsModeMenuOpen((open) => !open)}
                  type="button"
                >
                  <span className="modeTriggerLabel">{selectedMode.label}</span>
                  <span className="modeTriggerIcon">
                    <ChevronIcon />
                  </span>
                </button>

                {isModeMenuOpen ? (
                  <div className="modeMenu">
                    {SEARCH_MODES.map((option) => {
                      const isActive = option.id === selectedModeId;

                      return (
                        <button
                          aria-pressed={isActive}
                          className={`modeOption ${isActive ? "active" : ""}`}
                          key={option.id}
                          onClick={() => handleModeSelect(option.id)}
                          type="button"
                        >
                          <span className="modeOptionHeader">
                            <strong>{option.label}</strong>
                            {isActive ? (
                              <span className="modeCheck">
                                <CheckIcon />
                              </span>
                            ) : null}
                          </span>
                          <span>{option.description}</span>
                        </button>
                      );
                    })}
                  </div>
                ) : null}
              </div>

              {isLoading ? (
                <div aria-live="polite" className="activityBadge" role="status">
                  <LoaderIcon className="activitySpinner" />
                  <span>Search in progress</span>
                </div>
              ) : null}
            </div>

            <button
              aria-label={isLoading ? "Searching" : "Submit search"}
              className={`submitOrb ${isLoading ? "loading" : ""}`}
              disabled={isLoading || !query.trim()}
              type="submit"
            >
              {isLoading ? <LoaderIcon className="submitSpinner" /> : <SubmitIcon />}
            </button>
          </div>
        </form>

        {error ? <p className="errorInline">{error}</p> : null}
      </section>

      {isLoading || response ? (
        <section className="resultsDock">
          <ResultsPanel
            activeMethod={selectedMode.method}
            isLoading={isLoading}
            loadingLabel={currentProgress}
            progressEntries={progressEntries}
            recursiveResearchSelected={selectedMode.recursiveResearch}
            response={response}
          />
        </section>
      ) : null}
    </main>
  );
}
