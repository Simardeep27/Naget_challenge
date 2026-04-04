const LOCAL_AGENT_API_URL = "http://127.0.0.1:8000";

function resolveAgentApiBase(): string {
  const configuredBase =
    process.env.AGENT_API_URL ??
    // Vercel Services injects a server-to-server URL for the `agent` service.
    process.env.AGENT_URL ??
    (process.env.NEXT_PUBLIC_AGENT_URL?.startsWith("http")
      ? process.env.NEXT_PUBLIC_AGENT_URL
      : undefined);

  return (configuredBase || LOCAL_AGENT_API_URL).replace(/\/$/, "");
}

function buildForwardHeaders(request: Request): Headers {
  const headers = new Headers();
  for (const key of [
    "accept",
    "authorization",
    "content-type",
    "cookie",
    "x-vercel-protection-bypass",
  ]) {
    const value = request.headers.get(key);
    if (value) {
      headers.set(key, value);
    }
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
  const agentApiBase = resolveAgentApiBase();

  try {
    const upstream = await fetch(`${agentApiBase}${path}`, {
      method: request.method,
      headers: buildForwardHeaders(request),
      body,
      cache: "no-store",
    });

    if (!upstream.ok) {
      console.error("Agent proxy upstream returned a non-OK response", {
        path,
        agentApiBase,
        status: upstream.status,
        contentType: upstream.headers.get("content-type"),
      });
    }

    return new Response(upstream.body, {
      status: upstream.status,
      headers: buildResponseHeaders(upstream.headers),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown backend error";
    console.error("Agent proxy request failed", {
      path,
      agentApiBase,
      message,
    });

    return Response.json(
      {
        detail: `Agent backend is unavailable: ${message}`,
      },
      { status: 502 },
    );
  }
}
