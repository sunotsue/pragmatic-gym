"use client";

import { useEffect, useState } from "react";
import { api, fmt, pct, type Dist, type Provenance, type ScenarioPayload, type Score } from "@/lib/api";
import { useAppState } from "@/components/pg/app-state";
import { ChatHeader, ContextNotes, Thread } from "@/components/pg/chat";
import {
  Btn, DistributionRows, DotLegend, ErrorNote, FailureTag, Kv, Label, Loading, PageHead, ProvenanceBadge, SectionTitle, Stat, SyntheticTag,
} from "@/components/pg/primitives";
import { cn } from "@/lib/utils";

interface ListItem { id: string; title: string; capability: string; domain: string; relationship: string; speaker: string; channel: string; counterfactual_of: string | null; counterfactual_change: string | null; preview: string; ambiguity?: number }
interface InferOut { scenario_id: string; output: { interpretation_probs: Dist; most_likely: string; rationale: string; action: string; reply: string; provenance: Provenance }; score?: Score; wall_ms: number }
interface Diagnosis { provenance: { backend: string; model: string; is_real_inference: boolean }; overall: Record<string, { mean: number | null; lo: number | null; hi: number | null; n: number }> & { failure_counts: Record<string, number>; n: number }; by_capability: Record<string, { failure_counts: Record<string, number>; n: number; tvd: { mean: number }; top1_match: { mean: number | null }; action_regret: { mean: number } }>; rows: { scenario_id: string; title: string; capability: string; score: Score }[]; created: string }

export default function Explorer() {
  const { meta, backend } = useAppState();
  const [list, setList] = useState<ListItem[]>([]);
  const [cap, setCap] = useState<string>("indirect_speech");
  const [sid, setSid] = useState<string>("soft_no_birthday");
  const [scen, setScen] = useState<ScenarioPayload | null>(null);
  const [out, setOut] = useState<InferOut | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [diag, setDiag] = useState<Diagnosis | null>(null);
  const [diagLoading, setDiagLoading] = useState(false);

  useEffect(() => { api<ListItem[]>("/api/scenarios", { researcher: true }).then(setList).catch((e) => setErr(String(e))); }, []);
  useEffect(() => {
    setOut(null);
    api<ScenarioPayload>(`/api/scenarios/${sid}`, { researcher: true }).then(setScen).catch((e) => setErr(String(e)));
  }, [sid]);

  const caps = meta?.capabilities ?? {};
  const bases = list.filter((s) => !s.counterfactual_of && s.capability === cap);
  const family = scen ? list.filter((s) => s.id === scen.family_id || s.counterfactual_of === scen.family_id) : [];

  const run = async () => {
    setRunning(true); setErr(null);
    try { setOut(await api<InferOut>("/api/infer", { body: { scenario_id: sid, backend }, researcher: true })); }
    catch (e) { setErr(String(e)); }
    finally { setRunning(false); }
  };

  const loadDiag = async () => {
    setDiagLoading(true); setErr(null);
    try { setDiag(await api<Diagnosis>(`/api/diagnosis`, { researcher: true })); }
    catch (e) { setErr(String(e)); }
    finally { setDiagLoading(false); }
  };

  const obs = scen?.observable;
  const ref = scen?.latent?.reference.distribution;
  const actionLabel = (id?: string) => obs?.actions.find((a) => a.id === id)?.label ?? id;
  const sc = out?.score;

  return (
    <div>
      <PageHead
        n="01"
        title="Capability Explorer"
        lead={<>Find where a model misreads a message. Each conversation&rsquo;s meaning depends on the relationship, what happened before, and who is listening. Run a model and compare its reading with the authored reference.</>}
        measures={["reading vs reference", "confidence (calibration)", "chosen move vs best move", "failure type"]}
      />

      <section className="gutter grid grid-cols-1 gap-8 xl:grid-cols-[250px_minmax(420px,1fr)_430px]">
        {/* ---------------- scenario browser */}
        <aside className="xl:max-h-[calc(100vh-120px)] xl:overflow-auto quiet-scroll">
          <Label className="mb-2">Capability</Label>
          <div className="flex flex-col">
            {Object.entries(caps).map(([id, c]) => (
              <button key={id} onClick={() => { setCap(id); const first = list.find((s) => s.capability === id && !s.counterfactual_of); if (first) setSid(first.id); }}
                className={cn("border-l-[3px] py-1.5 pl-3 text-left text-[14px] leading-tight tracking-[-0.01em] transition-colors", cap === id ? "border-hot font-bold text-ink" : "border-transparent text-ink-2 hover:text-ink")}>
                {c.label}
              </button>
            ))}
          </div>
          <div className="rule my-4" />
          <Label className="mb-2">Scenarios</Label>
          <div className="flex flex-col gap-1">
            {bases.map((s) => (
              <button key={s.id} onClick={() => setSid(s.id)} className={cn("px-3 py-2 text-left transition-colors", (scen?.family_id === s.id) ? "bg-ink text-paper" : "hover:bg-wash")}>
                <div className="text-[14px] font-bold leading-tight tracking-[-0.015em]">{s.title}</div>
                <div className={cn("mt-0.5 line-clamp-1 t-micro", scen?.family_id === s.id ? "text-paper/70" : "text-ink-3")}>{s.relationship} · “{s.preview}”</div>
              </button>
            ))}
          </div>
        </aside>

        {/* ---------------- conversation */}
        <div className="min-w-0">
          {obs && scen && (
            <>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <SyntheticTag />
                <span className="t-micro text-ink-3">{caps[scen.capability]?.label}</span>
              </div>
              <div className="border-[1.5px] border-ink bg-paper p-5">
                <ChatHeader speaker={obs.speaker} relationship={obs.relationship_label} channel={obs.channel} participants={obs.participants} />
                <div className="mt-4"><ContextNotes context={obs.context} /></div>
                <div className="mt-5"><Thread history={obs.history} channel={obs.channel} /></div>
                {out && (
                  <div className="mt-5 border-t border-dashed border-ink-3 pt-4">
                    <div className="t-micro mb-2 text-ink-2">Model&apos;s suggested reply · <span className="font-bold text-ink">{actionLabel(out.output.action)}</span></div>
                    <div className="flex justify-end"><div className="max-w-[78%] border-[1.5px] border-dashed border-ink bg-paper px-3.5 py-2 text-[15px] leading-snug">{out.output.reply}</div></div>
                  </div>
                )}
              </div>
              <div className="mt-4 bg-paper">
                <Label>Question to the model</Label>
                <p className="mt-1 text-[16px] font-medium tracking-[-0.015em]">{obs.question}</p>
              </div>
              {family.length > 1 && (
                <div className="mt-6">
                  <Label className="mb-2">Counterfactual variants — same family, one change</Label>
                  <div className="flex flex-col gap-1">
                    {family.map((f) => (
                      <button key={f.id} onClick={() => setSid(f.id)} className={cn("flex items-baseline gap-3 border-b border-line py-1.5 text-left t-small transition-colors", f.id === sid ? "text-ink" : "text-ink-2 hover:text-ink")}>
                        <span className={cn("h-2 w-2 shrink-0 rounded-full", f.id === sid ? "bg-hot" : "bg-line")} />
                        <span className="font-bold">{f.counterfactual_of ? "Variant" : "Original"}</span>
                        <span>{f.counterfactual_change ?? "As written"}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ---------------- research panel */}
        <aside className="min-w-0 xl:border-l xl:border-line xl:pl-8">
          <div className="flex items-center justify-between gap-3">
            <h3 className="t-h3">Research panel</h3>
            <Btn variant="hot" onClick={run} disabled={running || !backend}>{running ? "Running…" : "Run model"}</Btn>
          </div>
          <div className="mt-2 min-h-5">{running ? <Loading label={`Inference · ${backend}`} /> : out && <ProvenanceBadge p={out.output.provenance} />}</div>
          <ErrorNote error={err} />

          <div className="mt-6">
            <SectionTitle aside={<DotLegend />}>What it might mean</SectionTitle>
            {obs && <DistributionRows labels={obs.interpretations} model={out?.output.interpretation_probs} reference={ref} highlight={out?.output.most_likely} />}
            {!out && <p className="mt-3 t-small text-ink-3">Run the model to see its interpretation distribution against the authored reference.</p>}
          </div>

          {out && (
            <>
              <div className="mt-6 grid grid-cols-3 gap-4 border-t border-line pt-4">
                <Stat label="Model confidence" value={pct(out.output.interpretation_probs[out.output.most_likely])} sub={sc ? `reference ${pct(sc.reference_confidence)}` : undefined} hot />
                <Stat label="Uncertainty" value={sc ? fmt(sc.model_entropy) : "—"} sub={sc ? `reference ${fmt(sc.ref_entropy)} (norm. entropy)` : undefined} />
                <Stat label="TVD to ref." value={sc ? fmt(sc.tvd) : "—"} sub={sc ? `CE ${fmt(sc.cross_entropy)} nats` : undefined} />
              </div>
              <div className="mt-6">
                <SectionTitle>What it does</SectionTitle>
                <Kv k="Chosen move" v={<span className="font-bold">{actionLabel(out.output.action)}</span>} />
                {sc && <Kv k="Best under ref." v={actionLabel(sc.best_action)} />}
                {sc && <Kv k="Acceptable" v={(sc.acceptable_actions ?? []).map(actionLabel).join(" · ")} />}
                {sc && <Kv k="Regret" v={<span className="num">{fmt(sc.action_regret)} <span className="text-ink-3">expected utility</span></span>} />}
                <p className="mt-2 t-micro text-ink-3">{String(out.output.provenance.settings?.reply_source ?? "")}</p>
              </div>
              {sc && (
                <div className="mt-6">
                  <SectionTitle>Evaluation</SectionTitle>
                  <div className="flex flex-wrap gap-1.5">{(sc.failures.length ? sc.failures : ["pass"]).map((f) => <FailureTag key={f} id={f} info={meta?.failures[f]} />)}</div>
                  <p className="mt-2 t-small text-ink-2">{meta?.failures[sc.primary_failure]}</p>
                  <p className="mt-2 t-micro text-ink-3">{sc.decisive ? "Decisive item: top-1 agreement is scored." : "Not decisive: top-1 agreement is not scored; calibration and action are."} {sc.ambiguous ? "Genuinely ambiguous: confident answers count as overconfidence." : ""}</p>
                </div>
              )}
              <div className="mt-6">
                <SectionTitle>Model rationale</SectionTitle>
                <p className="t-small text-ink-2">{out.output.rationale}</p>
              </div>
            </>
          )}

          {scen?.latent && (
            <div className="mt-8 border-t-[3px] border-ink pt-4">
              <SectionTitle aside="researcher only — never shown to models">Latent state</SectionTitle>
              {Object.entries(scen.latent.hidden).map(([k, v]) => <Kv key={k} k={k.replaceAll("_", " ")} v={typeof v === "string" ? v : JSON.stringify(v)} />)}
              {scen.latent.why_hidden_matters && <p className="mt-2 t-small text-ink-2">{scen.latent.why_hidden_matters}</p>}
              <div className="mt-4">
                <Label>Why the reference looks like this</Label>
                <p className="mt-1 t-small">{scen.latent.reference.rationale}</p>
                <p className="mt-1 t-micro text-ink-3">{scen.latent.reference.source} · {scen.latent.reference.version} · human distribution: {scen.latent.reference.human_distribution ? "available" : "not collected"}</p>
              </div>
              <div className="mt-4">
                <Label className="mb-1">Expected utility of each move under the reference</Label>
                {obs && Object.entries(scen.latent.reference_actions.expected_utility).sort((a, b) => b[1] - a[1]).map(([a, v]) => (
                  <div key={a} className="grid grid-cols-[1fr_120px_44px] items-center gap-2 py-0.5 t-small">
                    <span className={cn(scen.latent!.reference_actions.acceptable.includes(a) ? "font-bold" : "text-ink-2")}>{actionLabel(a)}</span>
                    <span className="relative h-[7px] bg-wash"><span className="absolute top-0 h-full" style={{ left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%`, background: v >= 0 ? "var(--pg-ink)" : "var(--pg-accent)" }} /></span>
                    <span className="num text-right">{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>
      </section>

      {/* ---------------- model-wide diagnosis */}
      <section className="gutter mt-16">
        <div className="rule-heavy" />
        <div className="flex flex-wrap items-end justify-between gap-4 pt-5">
          <div>
            <h2 className="text-[32px] font-bold leading-tight tracking-[-0.035em]">Diagnosis across the corpus</h2>
            <p className="mt-1 t-lead max-w-[760px] text-ink-2">The local base model on all 53 hand-written scenarios and counterfactual variants. This is the failure profile the Data Intervention Lab targets.</p>
          </div>
          <Btn variant="outline" onClick={loadDiag} disabled={diagLoading}>{diagLoading ? "Scoring 53 scenarios…" : diag ? "Reload" : "Run diagnosis"}</Btn>
        </div>
        {diag && (
          <div className="mt-6">
            <ProvenanceBadge p={{ backend: diag.provenance.backend, model: diag.provenance.model, is_real_inference: diag.provenance.is_real_inference, label: (diag.provenance as { label?: string }).label ?? `Local inference · ${diag.provenance.model} · ${diag.created}` }} />
            <div className="mt-5 grid grid-cols-2 gap-6 md:grid-cols-5">
              <Stat label="Top-1 on decisive items" value={pct(diag.overall.top1_match.mean)} sub={`n=${diag.overall.top1_match.n}`} hot />
              <Stat label="Mean TVD to reference" value={fmt(diag.overall.tvd.mean)} sub={`95% CI ${fmt(diag.overall.tvd.lo)}–${fmt(diag.overall.tvd.hi)}`} />
              <Stat label="Overconfident when ambiguous" value={pct(diag.overall.overconfident_rate_on_ambiguous.mean)} sub={`n=${diag.overall.overconfident_rate_on_ambiguous.n}`} />
              <Stat label="Mean action regret" value={fmt(diag.overall.action_regret.mean)} />
              <Stat label="Harmful moves" value={pct(diag.overall.action_harmful.mean)} />
            </div>
            <div className="mt-8 overflow-x-auto">
              <table className="w-full min-w-[760px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2"><th className="py-2 font-medium">Capability</th><th className="font-medium">n</th><th className="font-medium">Top-1 (decisive)</th><th className="font-medium">TVD</th><th className="font-medium">Regret</th><th className="font-medium">Failure profile</th></tr></thead>
                <tbody>
                  {Object.entries(diag.by_capability).map(([c, v]) => (
                    <tr key={c} className="border-b border-line">
                      <td className="py-2 font-bold">{caps[c]?.label ?? c}</td>
                      <td className="num">{v.n}</td>
                      <td className="num">{pct(v.top1_match.mean)}</td>
                      <td className="num">{fmt(v.tvd.mean)}</td>
                      <td className="num">{fmt(v.action_regret.mean)}</td>
                      <td>
                        <div className="flex h-3 w-full max-w-[360px]">
                          {Object.entries(v.failure_counts).sort().map(([f, n]) => (
                            <span key={f} title={`${f}: ${n}`} style={{ width: `${(n / v.n) * 100}%`, background: f === "pass" ? "var(--pg-line)" : f === "misread" || f === "literalism" ? "var(--pg-accent)" : f === "over_interpretation" ? "#ff8a66" : f === "overconfident" ? "var(--pg-ink)" : "var(--pg-ink-3)" }} className="h-full border-r border-paper" />
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-2 flex flex-wrap gap-4 t-micro text-ink-2">
                {[["misread / literalism", "var(--pg-accent)"], ["over-interpretation", "#ff8a66"], ["overconfident", "var(--pg-ink)"], ["other", "var(--pg-ink-3)"], ["pass", "var(--pg-line)"]].map(([l, c]) => <span key={l} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5" style={{ background: c }} />{l}</span>)}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
