const LOCAL_AGENT_API_URL = "http://127.0.0.1:8000";

function resolveAgentApiBase(): string {
  const configuredBase =
    process.env.AGENT_API_URL ??
    (process.env.NEXT_PUBLIC_AGENT_URL?.startsWith("http")
      ? process.env.NEXT_PUBLIC_AGENT_URL
      : undefined);

  return (configuredBase || LOCAL_AGENT_API_URL).replace(/\/$/, "");
}

function buildForwardHeaders(request: Request): Headers {
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  const accept = request.headers.get("accept");

  if (contentType) {
    headers.set("content-type", contentType);
  }

  if (accept) {
    headers.set("accept", accept);
  }

  return headers;
}

function buildResponseHeaders(upstreamHeaders: Headers): Headers {
  const headers = new Headers();

  for (const key of ["content-type", "cache-control", "connection", "x-accel-buffering"]) {
    const value = upstreamHeaders.get(key);
    if (value) {
      headers.set(key, value);
    }
  }

  return headers;
}

export async function proxyAgentRequest(request: Request, path: string): Promise<Response> {
  const body =
    request.method === "GET" || request.method === "HEAD" ? undefined : await request.text();

  try {
    const upstream = await fetch(`${resolveAgentApiBase()}${path}`, {
      method: request.method,
      headers: buildForwardHeaders(request),
      body,
      cache: "no-store",
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: buildResponseHeaders(upstream.headers),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown backend error";

    return Response.json(
      {
        detail: `Agent backend is unavailable: ${message}`,
      },
      { status: 502 },
    );
  }
}
