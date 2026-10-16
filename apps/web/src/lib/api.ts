export const API = process.env.NEXT_PUBLIC_PGYM_API ?? "http://localhost:8000";
export const RESEARCHER_TOKEN = process.env.NEXT_PUBLIC_PGYM_RESEARCHER_TOKEN ?? "local-researcher";

export type Dist = Record<string, number>;

export interface Message { sender: string; text: string; time?: string | null; is_you?: boolean }
export interface Option { id: string; label: string; description: string; literal?: boolean }
export interface ActionOption { id: string; label: string; description: string; example: string; style: string[] }
export interface Observable {
  scenario_id: string; domain: string; speaker: string; relationship_label: string; channel: string;
  participants: string[]; context: string[]; history: Message[]; question: string;
  interpretations: Option[]; actions: ActionOption[];
}
export interface Provenance {
  backend: string; model: string; is_real_inference: boolean; label: string; prompt_hash?: string;
  timestamp?: string; latency_ms?: number | null; recorded_at?: string | null; settings?: Record<string, unknown>;
}
export interface Score {
  decisive: boolean; ambiguous: boolean; top1_match: boolean | null; tvd: number; cross_entropy: number; kl: number;
  model_confidence: number; reference_confidence: number; confidence_gap: number; entropy_gap: number;
  ref_entropy: number; model_entropy: number; literal_shift: number | null; model_top: string; reference_top: string;
  action?: string; action_eu?: number; action_regret?: number; action_acceptable?: boolean; action_harmful?: boolean;
  best_action?: string; acceptable_actions?: string[]; failures: string[]; primary_failure: string;
}
export interface ScenarioPayload {
  id: string; family_id: string; title: string; capability: string; capabilities: string[]; domain: string;
  source: string; split: string; synthetic: boolean; counterfactual_of: string | null; counterfactual_change: string | null;
  observable: Observable;
  latent?: {
    reference: { distribution: Dist; source: string; version: string; rationale: string; human_distribution?: Dist | null };
    hidden: Record<string, unknown>; why_hidden_matters: string; factors: Record<string, string> | null;
    reference_actions: { best: string; acceptable: string[]; harmful: string[]; expected_utility: Dist };
  };
}

export async function api<T>(path: string, opts: { method?: string; body?: unknown; researcher?: boolean } = {}): Promise<T> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (opts.researcher) headers["x-researcher-token"] = RESEARCHER_TOKEN;
  const res = await fetch(`${API}${path}`, {
    method: opts.method ?? (opts.body ? "POST" : "GET"),
    headers,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new Error(`${res.status} · ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const pct = (x: number | null | undefined, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
export const fmt = (x: number | null | undefined, d = 2) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d));
export const signed = (x: number | null | undefined, d = 2) => (x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
