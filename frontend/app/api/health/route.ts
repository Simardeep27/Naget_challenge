import { proxyAgentRequest } from "@/app/api/_agent-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return proxyAgentRequest(request, "/health");
}
