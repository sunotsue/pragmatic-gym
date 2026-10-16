"use client";

import { useEffect, useMemo, useState } from "react";
import { api, fmt, pct, signed, type Dist, type Message, type Provenance } from "@/lib/api";
import { useAppState } from "@/components/pg/app-state";
import { ChatHeader, ContextNotes, Thread } from "@/components/pg/chat";
import { Btn, DistributionRows, DotLegend, ErrorNote, Kv, Label, Loading, PageHead, ProvenanceBadge, SectionTitle, Stat, SyntheticTag } from "@/components/pg/primitives";
import { cn } from "@/lib/utils";

interface Obs { turn: number; horizon: number; context: string[]; history: Message[]; speaker: string; relationship: string; channel: string; participants: string[]; question: string; interpretations: { id: string; label: string; description: string; literal?: boolean }[]; actions: { id: string; label: string; description: string; example: string; style: string[] }[]; done: boolean }
interface LogEntry { turn: number; action: string; message: string; belief: Dist | null; response: string; response_note: string; friction_before: number; friction_after: number; posterior_before: Dist; posterior_after: Dist; observable_entropy_before: number; components: Record<string, number>; weighted: Record<string, number>; reward: number; outcome: string | null }
interface Debug { hidden: { intent: string; patience: number; friction: number; members: string[] }; prior: Dist; observable_posterior: Dist; revealed_intent: string | null; factors: Record<string, string> | null; outcome: string | null; outcome_label: string | null; log: LogEntry[]; weights: Record<string, number>; return: number }
interface StepOut { session_id: string; observation: Obs; reward?: number; terminated: boolean; truncated: boolean; info: { session_id: string; seed: number; outcome: string | null }; debug?: Debug; agent?: { belief: Dist; action: string; rationale: string }; provenance?: Partial<Provenance> }
interface EnvDesc { id: string; title: string; subtitle: string; domain: string; capabilities: string[]; flagship: boolean; horizon: number; default_factors: Record<string, string>; opening: string[]; outcomes: Record<string, { label: string; success: boolean; value: Dist }>; reward_components: Record<string, string>; default_weights: Record<string, number> }
interface Audit { env_id: string; n_episodes: number; ranking: string[]; policies: Record<string, { return: { mean: number; lo: number; hi: number }; success_rate: number; mean_steps: number; components: Record<string, number>; return_by_intent: Dist }>; checks: Record<string, boolean | string> }

const COMP_COLORS: Record<string, string> = {
  task_completion: "#111111", pragmatic_understanding: "#ff3b00", clarification: "#ff8a66", repair: "#4a4a4a",
  boundary_respect: "#b32900", excessive_assumption: "#ffb399", social_friction: "#8a8a8a",
};

// Plain-language reward rules (authoritative definitions: COMPONENT_DOC in backend/pgym/environments/engine.py)
const REWARD_RULES: { id: string; label: string; plus: string; minus: string }[] = [
  { id: "pragmatic_understanding", label: "Understanding", plus: "A confident, correct guess of their real intention", minus: "A confident wrong guess. Only scored when the player states a guess (Bayes policy does; your own moves don't)" },
  { id: "boundary_respect", label: "Boundaries", plus: "—", minus: "Pushing or presuming; worse if they wanted to say no" },
  { id: "clarification", label: "Clarifying", plus: "The first gentle question while things are genuinely unclear", minus: "Asking again, or asking when it's already clear" },
  { id: "excessive_assumption", label: "Over-assuming", plus: "—", minus: "Acting certain while the best guess is still unsure" },
  { id: "social_friction", label: "Friction", plus: "—", minus: "Each bit of annoyance the move causes" },
  { id: "repair", label: "Repair", plus: "Smoothing things over after tension built up", minus: "—" },
  { id: "task_completion", label: "Outcome", plus: "An ending that suits who they really are (table on the right)", minus: "A bad ending, such as a coerced yes or them withdrawing" },
];

export default function Playground() {
  const { meta, backend } = useAppState();
  const [envs, setEnvs] = useState<EnvDesc[]>([]);
  const [envId, setEnvId] = useState("soft_no");
  const [seed, setSeed] = useState(7);
  const [factors, setFactors] = useState<Record<string, string>>({});
  const [sid, setSid] = useState<string | null>(null);
  const [obs, setObs] = useState<Obs | null>(null);
  const [debug, setDebug] = useState<Debug | null>(null);
  const [lastAgent, setLastAgent] = useState<{ agent?: StepOut["agent"]; provenance?: StepOut["provenance"] } | null>(null);
  const [reveal, setReveal] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [audit, setAudit] = useState<Audit | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [showPanel, setShowPanel] = useState(true);

  useEffect(() => { api<EnvDesc[]>("/api/envs").then(setEnvs).catch((e) => setErr(String(e))); }, []);
  const env = envs.find((e) => e.id === envId);
  const domFactors = useMemo(() => (meta?.factors ?? []).filter((f) => env && f.domains.includes(env.domain) && f.group !== "utterance" && ["prior_pattern", "shared_history", "relationship", "audience", "comm_style", "stakes", "norms", "commitment"].includes(f.id)), [meta, env]);
  const dom = env ? meta?.domains[env.domain] : null;

  const reset = async (id = envId, s = seed, f = factors) => {
    setBusy("reset"); setErr(null); setLastAgent(null); setReveal(false);
    try {
      const r = await api<{ session_id: string; observation: Obs; debug?: Debug }>(`/api/envs/${id}/reset`, { body: { seed: s, factors: f }, researcher: true });
      setSid(r.session_id); setObs(r.observation); setDebug(r.debug ?? null);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };
  useEffect(() => { if (envs.length) reset(envId, seed, {}); }, [envs.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = async (action: string) => {
    if (!sid) return;
    setBusy(action); setErr(null);
    try {
      const r = await api<StepOut>(`/api/envs/sessions/${sid}/step`, { body: { action }, researcher: true });
      setSid(r.session_id); setObs(r.observation); setDebug(r.debug ?? null); setLastAgent(null);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };
  const agentStep = async (policy: "model" | "bayes") => {
    if (!sid) return;
    setBusy(policy); setErr(null);
    try {
      const r = await api<StepOut>(`/api/envs/sessions/${sid}/agent_step`, { body: { policy, backend }, researcher: true });
      setSid(r.session_id); setObs(r.observation); setDebug(r.debug ?? null); setLastAgent({ agent: r.agent, provenance: r.provenance });
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
  };
  const runAudit = async () => {
    setAuditLoading(true);
    try { setAudit(await api<Audit>(`/api/envs/${envId}/audit?n=300`)); } catch (e) { setErr(String(e)); } finally { setAuditLoading(false); }
  };

  const log = debug?.log ?? [];
  const last = log[log.length - 1];
  const done = obs?.done;
  const intentLabel = (id?: string | null) => dom?.interpretations.find((i) => i.id === id)?.label ?? id ?? "—";
  const actionLabel = (id?: string) => dom?.actions.find((a) => a.id === id)?.label ?? id;
  const friction = debug?.hidden.friction ?? 0;
  const patience = debug?.hidden.patience ?? 1;
  const ret = debug?.return ?? 0;

  // annotate the transcript: under each agent turn, a tiny reward chip
  const annotations = useMemo(() => {
    if (!obs) return {};
    const out: Record<number, React.ReactNode> = {};
    let t = 0;
    obs.history.forEach((m, i) => {
      if (m.is_you && m.time?.startsWith("Turn")) {
        const e = log[t++];
        if (e) out[i] = <div className="mt-1 t-micro text-ink-3">{actionLabel(e.action)} · <span className={cn("num font-bold", e.reward >= 0 ? "text-ink" : "text-hot")}>{signed(e.reward)}</span></div>;
      }
    });
    return out;
  }, [obs, log]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <PageHead
        n="03"
        title="Social RL Playground"
        lead={<>See whether reading people well pays off over a whole conversation. The other person has a hidden intention the agent never sees, each move changes how they respond, and every reward is broken into parts so it can be audited.</>}
        measures={["belief vs exact posterior", "reward per turn, by component", "outcome (yes, withdrawal, coerced yes)", "reward audit against baseline policies"]}
      />

      <section className="gutter">
        <div className="grid grid-cols-1 gap-0 border-y-[1.5px] border-ink md:grid-cols-5">
          {envs.map((e) => (
            <button key={e.id} onClick={() => { setEnvId(e.id); setFactors({}); setAudit(null); reset(e.id, seed, {}); }}
              className={cn("border-line p-4 text-left transition-colors md:border-r last:border-r-0", envId === e.id ? "bg-ink text-paper" : "hover:bg-wash")}>
              <div className={cn("t-micro", envId === e.id ? "text-paper/70" : "text-ink-3")}>{e.flagship ? "Flagship" : "Environment"} · {e.domain}</div>
              <div className="mt-1 text-[17px] font-bold leading-tight tracking-[-0.025em]">{e.title}</div>
              <div className={cn("mt-1 line-clamp-2 t-micro", envId === e.id ? "text-paper/80" : "text-ink-2")}>{e.subtitle}</div>
            </button>
          ))}
        </div>
      </section>

      <section className="gutter mt-6 grid grid-cols-1 gap-10 xl:grid-cols-[minmax(440px,1fr)_minmax(420px,0.85fr)]">
        {/* ============ conversation */}
        <div className="min-w-0">
          <div className="mb-3 flex flex-wrap items-end gap-3">
            <label className="t-small"><span className="block text-ink-2">Seed</span>
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="num mt-1 h-8 w-20 border-[1.5px] border-ink px-2 text-[13px]" />
            </label>
            {env?.opening[0] === "family" && domFactors.slice(0, 4).map((f) => (
              <label key={f.id} className="t-small"><span className="block text-ink-2">{f.label}</span>
                <select value={factors[f.id] ?? env.default_factors[f.id] ?? f.default} onChange={(e) => setFactors({ ...factors, [f.id]: e.target.value })} className="mt-1 h-8 border-[1.5px] border-ink bg-paper px-1.5 text-[13px]">
                  {f.levels.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                </select>
              </label>
            ))}
            <Btn variant="outline" onClick={() => reset()} disabled={!!busy}>Reset episode</Btn>
            <Btn variant="ghost" onClick={() => { const s = seed + 1; setSeed(s); reset(envId, s); }} disabled={!!busy}>Next seed →</Btn>
          </div>

          <div className="border-[1.5px] border-ink p-5">
            {obs ? (
              <>
                <ChatHeader speaker={obs.speaker} relationship={obs.relationship} channel={obs.channel} participants={obs.participants} />
                <div className="mt-4"><ContextNotes context={obs.context} /></div>
                <div className="mt-5"><Thread history={obs.history} channel={obs.channel} emphasizeLast={!done} pendingFrom={busy && busy !== "reset" ? obs.speaker : null} annotations={annotations} /></div>
                {done && (
                  <div className="mt-5 border-t-[3px] border-ink pt-3 text-center">
                    <div className="t-micro text-ink-2">Episode over</div>
                    <div className="mt-0.5 text-[22px] font-bold tracking-[-0.03em]">{debug?.outcome_label ?? "Finished"}</div>
                    <div className="num t-small text-ink-2">return {signed(ret)} · hidden intent: <span className="font-bold text-ink">{intentLabel(debug?.hidden.intent)}</span></div>
                  </div>
                )}
              </>
            ) : <Loading label="Starting episode" />}
          </div>

          {/* action bar */}
          <div className="mt-4">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <Label>Your move — turn {Math.min((obs?.turn ?? 0) + 1, obs?.horizon ?? 4)} of {obs?.horizon ?? 4}</Label>
              <div className="flex gap-2">
                <Btn variant="outline" onClick={() => agentStep("bayes")} disabled={!!busy || !!done} title="Reference policy that acts on the exact observable posterior">Bayes policy</Btn>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
              {obs?.actions.map((a) => (
                <button key={a.id} onClick={() => step(a.id)} disabled={!!busy || !!done}
                  className="group flex flex-col items-start border-[1.5px] border-line px-3 py-2 text-left transition-colors hover:border-ink disabled:opacity-40">
                  <span className="text-[13.5px] font-bold tracking-[-0.01em]">{a.label}{a.style.includes("deferential") && <span className="ml-1.5 t-micro font-medium text-ink-3">deferential</span>}</span>
                  <span className="line-clamp-1 t-micro text-ink-2">“{a.example}”</span>
                </button>
              ))}
            </div>
            {lastAgent?.agent && (
              <div className="mt-3 flex flex-wrap items-center gap-3 bg-wash px-3 py-2 t-small">
                <ProvenanceBadge p={lastAgent.provenance} />
                <span>chose <span className="font-bold">{actionLabel(lastAgent.agent.action)}</span></span>
              </div>
            )}
            <ErrorNote error={err} />
          </div>
        </div>

        {/* ============ research panel */}
        <aside className="min-w-0">
          <div className="flex items-center justify-between">
            <h3 className="t-h3">Research panel</h3>
            <Btn variant="ghost" onClick={() => setShowPanel(!showPanel)}>{showPanel ? "Collapse" : "Expand"}</Btn>
          </div>
          {showPanel && debug && (
            <div className="mt-4 space-y-8">
              <div className="grid grid-cols-3 gap-4">
                <Stat label="Return so far" value={signed(ret)} hot={ret < 0} />
                <Stat label="Friction" value={fmt(friction)} sub={`withdraws at ${fmt(patience)} (hidden)`} />
                <Stat label="Outcome" value={<span className="text-[18px]">{debug.outcome_label ?? "in progress"}</span>} />
              </div>
              <div>
                <div className="relative h-2 bg-wash"><div className="absolute h-full bg-hot transition-all" style={{ width: `${Math.min(1, friction / Math.max(patience, 0.01)) * 100}%` }} /></div>
                <div className="mt-1 flex justify-between t-micro text-ink-3"><span>relationship strain</span><span>patience</span></div>
              </div>

              <div>
                <SectionTitle aside={<button onClick={() => setReveal(!reveal)} className="t-small font-bold underline underline-offset-4">{reveal ? "Hide" : "Reveal"} hidden intent</button>}>Hidden social state</SectionTitle>
                <div className={cn("flex items-center gap-3 px-3 py-2.5", reveal ? "bg-ink text-paper" : "fixture-stripes")}>
                  {reveal ? <><span className="t-micro text-paper/70">True intent (sampled from the prior)</span><span className="text-[16px] font-bold">{intentLabel(debug.hidden.intent)}</span></> : <span className="bg-paper px-2 t-small font-bold">hidden — the agent never sees this</span>}
                </div>
                {debug.factors && <p className="mt-2 t-micro text-ink-3">Opening rendered from: {Object.entries(debug.factors).filter(([k, v]) => meta?.factors.find((f) => f.id === k)?.default !== v).map(([k, v]) => `${k}=${v}`).join(", ") || "default levels"}</p>}
              </div>

              <div>
                <SectionTitle aside={<DotLegend modelLabel={lastAgent?.agent ? "Agent belief" : "Prior"} refLabel="Observable posterior" />}>Beliefs</SectionTitle>
                {dom && <DistributionRows labels={dom.interpretations} model={lastAgent?.agent?.belief ?? (log.length ? null : debug.prior)} reference={debug.observable_posterior} />}
                <p className="mt-2 t-micro text-ink-3">The observable posterior is the exact Bayes filter over the interlocutor policy given only what the agent has seen. Clarification and assumption rewards condition on it — never on the hidden intent.</p>
              </div>

              <div>
                <SectionTitle aside="weighted · hypotheses, not measurements">Reward decomposition</SectionTitle>
                {log.length === 0 && <p className="t-small text-ink-3">Take a turn to see rewards.</p>}
                {log.map((e) => {
                  const entries = Object.entries(e.weighted).filter(([, v]) => Math.abs(v) > 1e-6);
                  return (
                    <div key={e.turn} className="border-b border-line py-2">
                      <div className="flex items-baseline justify-between t-small"><span><span className="num text-ink-3">T{e.turn}</span> <span className="font-bold">{actionLabel(e.action)}</span> → <span className="text-ink-2">{e.response_note}</span></span><span className={cn("num font-bold", e.reward < 0 && "text-hot")}>{signed(e.reward)}</span></div>
                      <div className="mt-1.5 flex h-3 w-full">
                        <div className="flex h-full w-1/2 justify-end">{entries.filter(([, v]) => v < 0).map(([k, v]) => <span key={k} title={`${k} ${v.toFixed(2)}`} style={{ width: `${Math.min(100, Math.abs(v) * 50)}%`, background: COMP_COLORS[k] }} className="h-full border-l border-paper" />)}</div>
                        <div className="h-full w-px bg-ink" />
                        <div className="flex h-full w-1/2">{entries.filter(([, v]) => v > 0).map(([k, v]) => <span key={k} title={`${k} +${v.toFixed(2)}`} style={{ width: `${Math.min(100, v * 50)}%`, background: COMP_COLORS[k] }} className="h-full border-r border-paper" />)}</div>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-3 t-micro text-ink-2">{entries.map(([k, v]) => <span key={k}>{k.replaceAll("_", " ")} <span className="num">{signed(v)}</span></span>)}</div>
                    </div>
                  );
                })}
                <div className="mt-2 flex flex-wrap gap-3 t-micro text-ink-2">{Object.entries(COMP_COLORS).map(([k, c]) => <span key={k} className="flex items-center gap-1" title={meta?.reward_components[k]}><span className="h-2 w-2" style={{ background: c }} />{k.replaceAll("_", " ")}</span>)}</div>
              </div>

              {last && (
                <div>
                  <SectionTitle>Last transition</SectionTitle>
                  <Kv k="State before" v={<span className="num">friction {fmt(last.friction_before)} · H(posterior) {fmt(last.observable_entropy_before)}</span>} />
                  <Kv k="Action" v={actionLabel(last.action)} />
                  <Kv k="Interlocutor" v={<><span className="mono">{last.response}</span> — {last.response_note}</>} />
                  <Kv k="State after" v={<span className="num">friction {fmt(last.friction_after)}{last.outcome ? ` · terminal: ${last.outcome}` : ""}</span>} />
                  <p className="mt-2 t-micro text-ink-3">The agent&apos;s message text is stored for display only; no reward component reads it, so length and deference cannot be rewarded.</p>
                </div>
              )}
            </div>
          )}
        </aside>
      </section>

      {/* ============ reward design */}
      {env && dom && (
        <section className="gutter mt-16">
          <div className="rule-heavy" />
          <div className="pt-5">
            <h2 className="text-[32px] font-bold leading-tight tracking-[-0.035em]">Reward design · {env.title}</h2>
            <p className="mt-1 t-lead max-w-[820px] text-ink-2">
              Each turn&rsquo;s reward is the sum of seven parts, each multiplied by its weight. The wording of a message is never scored, so
              politeness or length alone can&rsquo;t earn points.
            </p>
          </div>
          <div className="mt-6 grid grid-cols-1 gap-10 xl:grid-cols-[1.1fr_1fr]">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2">
                  <th className="py-2 pr-3 font-medium">Part</th><th className="pr-3 font-medium">Gains points for</th><th className="pr-3 font-medium">Loses points for</th><th className="text-right font-medium">Weight</th>
                </tr></thead>
                <tbody>
                  {REWARD_RULES.map((r) => (
                    <tr key={r.id} className="border-b border-line align-top">
                      <td className="py-2 pr-3 font-bold"><span className="mr-2 inline-block h-2.5 w-2.5 align-[-1px]" style={{ background: COMP_COLORS[r.id] }} />{r.label}</td>
                      <td className="py-2 pr-3 text-ink-2">{r.plus}</td>
                      <td className="py-2 pr-3 text-ink-2">{r.minus}</td>
                      <td className="num py-2 text-right">{fmt(env.default_weights[r.id], 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="overflow-x-auto">
              <SectionTitle aside="by their real intention">How each ending is valued</SectionTitle>
              <table className="w-full min-w-[520px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2">
                  <th className="py-2 pr-3 font-medium">Ending</th>
                  {dom.interpretations.map((i) => <th key={i.id} className="pr-2 text-right font-medium leading-tight">{i.label}</th>)}
                </tr></thead>
                <tbody>
                  {Object.entries(env.outcomes).map(([id, o]) => (
                    <tr key={id} className="border-b border-line">
                      <td className="py-1.5 pr-3">{o.label}{!o.success && <span className="ml-1.5 t-micro text-hot">fail</span>}</td>
                      {dom.interpretations.map((i) => {
                        const v = o.value[i.id] ?? 0;
                        return <td key={i.id} className={cn("num pr-2 text-right", v < 0 ? "text-hot" : v >= 0.9 ? "font-bold" : "text-ink-2")}>{signed(v, 1)}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 t-micro text-ink-3">Read across a row: the same ending is worth more or less depending on who they really were. A plan confirmed with someone who wanted to say no scores little.</p>
            </div>
          </div>
          <div className="mt-3"><SyntheticTag>Authored reward hypothesis · tested by the audit below</SyntheticTag></div>
        </section>
      )}

      {/* ============ reward audit */}
      <section className="gutter mt-16">
        <div className="rule-heavy" />
        <div className="flex flex-wrap items-end justify-between gap-4 pt-5">
          <div>
            <h2 className="text-[32px] font-bold leading-tight tracking-[-0.035em]">Reward audit · {env?.title}</h2>
            <p className="mt-1 t-lead max-w-[820px] text-ink-2">What does this reward actually incentivize? 300 matched-seed episodes per policy. If a constant, stylistic or always-clarify policy matches the Bayes policy, the reward is hackable.</p>
          </div>
          <Btn variant="outline" onClick={runAudit} disabled={auditLoading}>{auditLoading ? "Simulating…" : audit ? "Re-run" : "Run audit"}</Btn>
        </div>
        {audit && (
          <div className="mt-6 grid grid-cols-1 gap-10 xl:grid-cols-[1fr_340px]">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2"><th className="py-2 font-medium">Policy</th><th className="font-medium">Mean return (95% CI)</th><th className="text-right font-medium">Success</th><th className="text-right font-medium">Steps</th><th className="pl-4 font-medium">Return by hidden intent</th></tr></thead>
                <tbody>
                  {audit.ranking.map((p) => {
                    const r = audit.policies[p];
                    const lo = -3, hi = 2;
                    const x = (v: number) => ((v - lo) / (hi - lo)) * 100;
                    return (
                      <tr key={p} className={cn("border-b border-line", (p === "bayes" || p === "oracle" || p === "planner") && "font-bold")}>
                        <td className="py-1.5 pr-3">{p === "oracle" ? "Greedy oracle (sees hidden state)" : p === "bayes" ? "Greedy Bayes (observable posterior)" : p === "planner" ? "Planner (looks ahead, observable posterior)" : p === "random" ? "Random" : `Always: ${actionLabel(p.replace("always:", ""))}`}</td>
                        <td className="w-[300px] pr-3">
                          <div className="flex items-center gap-2">
                            <div className="relative h-4 flex-1">
                              <div className="absolute top-1/2 h-px w-full bg-line" /><div className="absolute top-0 h-full w-px bg-ink-3" style={{ left: `${x(0)}%` }} />
                              <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-ink" style={{ left: `${x(r.return.lo)}%`, width: `${x(r.return.hi) - x(r.return.lo)}%` }} />
                              <div className={cn("absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full", p === "bayes" ? "bg-hot" : "bg-ink")} style={{ left: `${x(r.return.mean)}%` }} />
                            </div>
                            <span className="num w-12 text-right">{signed(r.return.mean)}</span>
                          </div>
                        </td>
                        <td className="num text-right">{pct(r.success_rate)}</td>
                        <td className="num text-right">{fmt(r.mean_steps, 1)}</td>
                        <td className="pl-4 t-micro font-normal text-ink-2">{Object.entries(r.return_by_intent).map(([k, v]) => `${intentLabel(k)} ${signed(v, 1)}`).join(" · ")}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="space-y-3">
              <SectionTitle>Sanity checks</SectionTitle>
              {[["oracle_beats_bayes", "Privileged information helps (greedy oracle > greedy Bayes)"], ["bayes_beats_best_constant", "Inference beats any fixed move"], ["bayes_beats_random", "Bayes beats random"]].map(([k, l]) => (
                <div key={k} className="flex items-center gap-2 t-small"><span className={cn("h-3 w-3", audit.checks[k] ? "bg-ink" : "bg-hot")} />{l}</div>
              ))}
              <p className="t-small text-ink-2">Best constant policy: <span className="font-bold text-ink">{actionLabel(String(audit.checks.best_constant).replace("always:", ""))}</span>. Deferential and over-apologizing policies are explicitly included as reward-hacking probes.</p>
              <SyntheticTag>Simulated interlocutor · authored response model</SyntheticTag>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
