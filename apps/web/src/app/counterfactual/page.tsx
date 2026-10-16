"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, fmt, pct, type Dist, type ScenarioPayload } from "@/lib/api";
import { useAppState } from "@/components/pg/app-state";
import { ChatHeader, ContextNotes, Thread } from "@/components/pg/chat";
import { Btn, DistributionRows, DotLegend, ErrorNote, Label, Loading, PageHead, ProvenanceBadge, SectionTitle, Segmented, Stat, SyntheticTag } from "@/components/pg/primitives";
import { CellDots, Forest, InteractionPlot, SensitivityScatter, StackBar } from "@/components/pg/charts";
import { cn } from "@/lib/utils";

interface Preview { valid: boolean; violations?: string[]; levels: Record<string, string>; scenario?: ScenarioPayload & { latent?: ScenarioPayload["latent"] & { hidden: { lssm_terms?: { term: string; delta: Dist }[] } } } }
interface Design { design_id: string; domain: string; outcome_intent: string; factors: string[]; cells: { cell: string; assignment: Record<string, string>; levels: Record<string, string> }[]; excluded: { assignment: Record<string, string>; reasons: string[] }[]; n_units: number }
interface Effect { factor: string; level: string; reference_level: string; model: { estimate: number | null; lo: number | null; hi: number | null; significant?: boolean; n_clusters: number; note?: string }; hypothesis: number; direction_agrees: boolean | null }
interface Interaction { factors: string[]; levels: string[]; baseline: string[]; model: Effect["model"]; hypothesis: number }
interface CellRow { cell: string; assignment: Record<string, string>; n: number; y_mean: number; y_sd: number; y_ref: number; entropy: number; entropy_ref: number; top_counts: Record<string, number>; action_counts: Record<string, number> }
interface RunOut {
  experiment_id: string; runtime_s: number; provenance: { backend: string; model: string; is_real_inference: boolean; label: string };
  design: Design;
  analysis: { outcome_intent: string; n_units: number; cells: CellRow[]; main_effects: Effect[]; interactions: Interaction[];
    ols: { r2?: number; n?: number; se_type?: string; coefficients?: { term: string; coef: number; se: number; lo: number; hi: number; p: number }[]; error?: string };
    variance: { context: number; wording: number; residual: number; note: string }; sensitivity: { slope: number | null; intercept?: number; corr: number | null; mean_abs_entropy_gap?: number };
    human: { available: boolean; note?: string }; caveat: string };
  units: { cell: string; variant: number; rep: number; assignment: Record<string, string>; result: { probs: Dist; action: string } | null }[];
}

const PRESETS: { id: string; label: string; family: string; vary: Record<string, string[]>; base?: Record<string, string> }[] = [
  { id: "history_audience", label: "History × audience", family: "inv_let_you_know", vary: { shared_history: ["unspecified", "repeated_disappointment", "established_trust"], audience: ["private", "group_chat"] } },
  { id: "placebo", label: "History vs placebo", family: "inv_let_you_know", vary: { prior_pattern: ["none", "repeated"], placebo: ["none", "neutral"] } },
  { id: "rel_history", label: "Relationship × refusals", family: "inv_let_you_know", vary: { relationship: ["close_friend", "romantic_interest", "acquaintance"], prior_pattern: ["none", "repeated"] } },
  { id: "norms", label: "Stated norms × style", family: "inv_think_about_it", vary: { norms: ["unspecified", "indirect_norm", "literal_norm"], comm_style: ["unspecified", "direct", "avoidant"] } },
  { id: "same_page", label: "Same page: commitment × pattern", family: "rel_like_what_we_have", vary: { commitment: ["undefined", "explicit"], prior_pattern: ["none", "repeated"] } },
  { id: "check", label: "Manipulation check", family: "inv_well_see", vary: { directness: ["indirect", "explicit"], shared_history: ["unspecified", "established_trust"] } },
];

export default function CounterfactualLab() {
  const { meta, backend } = useAppState();
  const [family, setFamily] = useState("inv_let_you_know");
  const [base, setBase] = useState<Record<string, string>>({});
  const [vary, setVary] = useState<Record<string, string[]>>(PRESETS[0].vary);
  const [variant, setVariant] = useState(0);
  const [nVariants, setNVariants] = useState(4);
  const [outcome, setOutcome] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [defaults, setDefaults] = useState<Set<string>>(new Set());
  const [design, setDesign] = useState<Design | null>(null);
  const [designErr, setDesignErr] = useState<string | null>(null);
  const [run, setRun] = useState<RunOut | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [preset, setPreset] = useState("history_audience");

  const fam = meta?.families.find((f) => f.id === family);
  const domain = fam?.domain ?? "invitation";
  const factors = useMemo(() => (meta?.factors ?? []).filter((f) => f.domains.includes(domain)), [meta, domain]);
  const dom = meta?.domains[domain];
  const keyIntent = outcome ?? ({ invitation: "soft_decline", relationship: "avoiding_talk", emotional: "hurt_face_saving", family: "indirect_request" } as Record<string, string>)[domain];
  const fLabel = (id: string) => factors.find((f) => f.id === id)?.label ?? id;
  const lvLabel = (f: string, lv: string) => factors.find((x) => x.id === f)?.levels.find((l) => l.id === lv)?.label ?? lv;

  // the cell currently displayed = base levels with varied factors at their selected "view" level
  const [view, setView] = useState<Record<string, string>>({});
  const shown = useMemo(() => {
    const s: Record<string, string> = { ...base };
    for (const [f, lvls] of Object.entries(vary)) s[f] = view[f] && lvls.includes(view[f]) ? view[f] : lvls[0];
    return s;
  }, [base, vary, view]);

  useEffect(() => {
    const body = { family_id: family, levels: shown, variant };
    api<Preview>("/api/counterfactual/preview", { body, researcher: true }).then(setPreview).catch((e) => setErr(String(e)));
  }, [family, shown, variant]);

  useEffect(() => {
    api<Preview>("/api/counterfactual/preview", { body: { family_id: family, levels: {}, variant } }).then((p) => setDefaults(new Set(p.scenario?.observable.context ?? [])));
  }, [family, variant]);

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      api<Design>("/api/counterfactual/design", { body: { family_id: family, base, vary, n_variants: nVariants, reps: 1, outcome_intent: keyIntent } })
        .then((d) => { setDesign(d); setDesignErr(null); })
        .catch((e) => { setDesign(null); setDesignErr(String(e)); });
    }, 200);
  }, [family, base, vary, nVariants, keyIntent]);

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setPreset(id); setFamily(p.family); setVary(p.vary); setBase(p.base ?? {}); setView({}); setRun(null); setOutcome(null);
  };

  const toggleVary = (fid: string) => {
    setRun(null);
    setVary((v) => {
      const n = { ...v };
      if (n[fid]) {
        const keep = shown[fid];
        setBase((b) => ({ ...b, [fid]: keep }));
        delete n[fid];
      } else {
        if (Object.keys(n).length >= 3) return v;
        const f = factors.find((x) => x.id === fid)!;
        n[fid] = [f.default, ...f.levels.map((l) => l.id).filter((l) => l !== f.default)].slice(0, 3);
      }
      return n;
    });
  };
  const toggleLevel = (fid: string, lv: string) => {
    setRun(null);
    setVary((v) => {
      const cur = v[fid] ?? [];
      const nxt = cur.includes(lv) ? cur.filter((x) => x !== lv) : [...cur, lv];
      if (nxt.length < 2) return v;
      const f = factors.find((x) => x.id === fid)!;
      return { ...v, [fid]: f.levels.map((l) => l.id).filter((l) => nxt.includes(l)) };
    });
  };

  const doRun = async () => {
    setRunning(true); setErr(null); setPair(null);
    try {
      const r = await api<RunOut>("/api/counterfactual/run", { body: { spec: { family_id: family, base, vary, n_variants: nVariants, reps: 1, outcome_intent: keyIntent }, backend }, researcher: true });
      setRun(r);
      const cells = r.analysis.cells;
      if (cells.length >= 2) setPair([cells[0].cell, cells[cells.length - 1].cell]);
    } catch (e) { setErr(String(e)); }
    finally { setRunning(false); }
  };

  const scen = preview?.scenario;
  const obs = scen?.observable;
  const highlight = useMemo(() => new Set((obs?.context ?? []).filter((c, i) => i > 0 && !defaults.has(c))), [obs, defaults]);
  const cellName = (a: Record<string, string>) => Object.entries(a).map(([f, l]) => lvLabel(f, l)).join(" · ");
  const intentLabel = (id: string) => dom?.interpretations.find((i) => i.id === id)?.label ?? id;
  const an = run?.analysis;

  const pairStats = useMemo(() => {
    if (!run || !pair) return null;
    const by = (cell: string) => {
      const m: Record<number, number[]> = {};
      run.units.filter((u) => u.cell === cell && u.result).forEach((u) => { (m[u.variant] ??= []).push(u.result!.probs[run.design.outcome_intent] ?? 0); });
      return Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.reduce((a, b) => a + b, 0) / v.length]));
    };
    const a = by(pair[0]), b = by(pair[1]);
    const ks = Object.keys(a).filter((k) => k in b);
    const d = ks.map((k) => b[k] - a[k]);
    if (!d.length) return null;
    const mean = d.reduce((x, y) => x + y, 0) / d.length;
    let lo: number | null = null, hi: number | null = null;
    if (d.length >= 3) {
      let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const boots = Array.from({ length: 2000 }, () => { let s = 0; for (let i = 0; i < d.length; i++) s += d[Math.floor(rnd() * d.length)]; return s / d.length; }).sort((x, y) => x - y);
      lo = boots[50]; hi = boots[1949];
    }
    return { mean, lo, hi, n: d.length, d };
  }, [run, pair]);

  const interactionData = useMemo(() => {
    if (!run || run.design.factors.length < 2) return null;
    const [f1, f2] = run.design.factors;
    const l1 = vary[f1] ?? [], l2 = vary[f2] ?? [];
    const rows = l1.map((a) => {
      const row: Record<string, number | string> = { x: lvLabel(f1, a) };
      l2.forEach((b) => {
        const c = run.analysis.cells.find((x) => x.assignment[f1] === a && x.assignment[f2] === b);
        if (c) { row[`m_${b}`] = c.y_mean; row[`h_${b}`] = c.y_ref; }
      });
      return row;
    });
    const palette = ["var(--pg-accent)", "var(--pg-ink)", "#9a9a9a"];
    return { rows, f1, f2, series: l2.flatMap((b, i) => [{ key: `m_${b}`, label: `model · ${lvLabel(f2, b)}`, color: palette[i % 3] }, { key: `h_${b}`, label: `hypothesis · ${lvLabel(f2, b)}`, color: palette[i % 3], dashed: true }]) };
  }, [run, vary]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <PageHead
        n="02"
        title="Counterfactual Lab"
        lead={<>Test whether social context actually moves the model. Keep the message fixed, change one social fact at a time (relationship, history, audience…), and compare how far the model moves with how far the hypothesis says it should.</>}
        measures={["effect of each social factor", "interactions between factors", "sensitivity slope (model vs hypothesis)", "variance from context vs surface variants"]}
        right={<div className="flex flex-col items-end gap-2"><Label>Presets</Label><Segmented size="sm" value={preset} onChange={applyPreset} options={PRESETS.map((p) => ({ id: p.id, label: p.label }))} /></div>}
      />

      <section className="gutter grid grid-cols-1 gap-10 xl:grid-cols-[minmax(440px,0.9fr)_minmax(560px,1.1fr)]">
        {/* ============ LEFT: conversation & social context */}
        <div className="min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <Label>Utterance family (held constant)</Label>
              <select value={family} onChange={(e) => { setFamily(e.target.value); setBase({}); setView({}); setVary({}); setRun(null); setOutcome(null); }}
                className="mt-1 h-9 w-full max-w-[380px] border-[1.5px] border-ink bg-paper px-2 text-[14px] font-bold tracking-[-0.01em]">
                {(meta?.families ?? []).map((f) => <option key={f.id} value={f.id}>{f.title} — {f.domain}{f.split !== "train" ? ` (${f.split.replace("eval_", "held-out ")})` : ""}</option>)}
              </select>
            </div>
            <div className="text-right">
              <Label>Surface variant</Label>
              <Segmented size="sm" value={String(variant)} onChange={(v) => setVariant(Number(v))} options={[0, 1, 2].map((v) => ({ id: String(v), label: `v${v}`, hint: "names, paraphrase, order, punctuation" }))} className="mt-1" />
            </div>
          </div>

          {design && design.factors.length > 0 && (() => {
            const i = design.cells.findIndex((c) => Object.entries(c.assignment).every(([k, v]) => shown[k] === v));
            return (
              <p className="mt-5 t-small text-ink-2">
                <span className="font-bold text-ink">{i >= 0 ? `Previewing cell ${i + 1} of ${design.cells.length}` : "Previewing an excluded combination"}</span>
                {" · "}{cellName(Object.fromEntries(design.factors.map((f) => [f, shown[f]])))}
                <span className="text-ink-3"> · click a row in the design matrix to switch</span>
              </p>
            );
          })()}
          <div className="mt-2 border-[1.5px] border-ink p-5">
            {preview && !preview.valid ? (
              <div className="py-10 text-center"><p className="t-h3">Incoherent combination</p><p className="mt-2 t-small text-ink-2">{preview.violations?.join(" ")}</p><p className="mt-1 t-micro text-ink-3">Excluded from designs rather than rendered as a contradictory story.</p></div>
            ) : obs ? (
              <>
                <ChatHeader speaker={obs.speaker} relationship={obs.relationship_label} channel={obs.channel} participants={obs.participants} />
                <div className="mt-4"><ContextNotes context={obs.context} highlight={highlight} /></div>
                <div className="mt-5"><Thread history={obs.history} channel={obs.channel} /></div>
              </>
            ) : <Loading label="Rendering" />}
          </div>
          <p className="mt-2 t-micro text-ink-3"><mark className="bg-hot-wash px-0.5">Highlighted</mark> sentences are introduced by the current factor levels; everything else is held fixed. The outlined message is the utterance being interpreted.</p>

          {scen?.latent && (
            <div className="mt-6">
              <SectionTitle aside={<SyntheticTag>LSSM hypothesis · {scen.latent.reference.version}</SyntheticTag>}>Hypothesised reading of this cell</SectionTitle>
              {dom && <DistributionRows labels={dom.interpretations} reference={scen.latent.reference.distribution} model={null} />}
              <details className="mt-3">
                <summary className="cursor-pointer t-small font-bold">Audit trail: logit contributions</summary>
                <div className="mt-2 space-y-0.5">
                  {(scen.latent.hidden.lssm_terms ?? []).map((t, i) => (
                    <div key={i} className="grid grid-cols-[180px_1fr] gap-2 t-micro"><span className="mono">{t.term}</span><span className="text-ink-2">{Object.entries(t.delta).map(([k, v]) => `${intentLabel(k)} ${v >= 0 ? "+" : ""}${v.toFixed(1)}`).join(" · ")}</span></div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>

        {/* ============ RIGHT: experimental controls & results */}
        <div className="min-w-0">
          <SectionTitle aside={`${Object.keys(vary).length} of 3 compared`}><span className="num mr-2 text-hot">1</span>Choose what to test</SectionTitle>
          <p className="mb-3 max-w-[640px] t-small text-ink-2">
            Every social variable is either <span className="font-bold text-ink">Fixed</span>, with one level that is the same in every version of the conversation,
            or <span className="font-bold text-ink">Compare</span>, where you tick the levels to test against each other. Compare up to 3 variables.
          </p>
          <div className="border-t-[1.5px] border-ink">
            {factors.map((f) => {
              const varied = !!vary[f.id];
              const cur = shown[f.id] ?? f.default;
              const full = !varied && Object.keys(vary).length >= 3;
              return (
                <div key={f.id} className={cn("grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-2 border-b border-line py-2 sm:grid-cols-[150px_1fr_auto]", varied && "bg-hot-wash/40")}>
                  <div className="leading-tight">
                    <div className="text-[13.5px] font-bold tracking-[-0.01em]">{f.label}</div>
                    {f.group === "utterance" && <div className="t-micro text-ink-3">manipulation check</div>}
                  </div>
                  <div className="order-last col-span-2 flex flex-wrap gap-1 sm:order-none sm:col-span-1">
                    {f.levels.map((l) => {
                      if (!varied) {
                        return (
                          <button key={l.id} onClick={() => { setBase((b) => ({ ...b, [f.id]: l.id })); setRun(null); }}
                            className={cn("h-7 border-[1.5px] px-2 text-[12.5px] font-medium transition-colors", cur === l.id ? "border-ink bg-ink text-paper" : "border-line bg-paper text-ink hover:border-ink")}>
                            {l.label}
                          </button>
                        );
                      }
                      const on = !!vary[f.id]?.includes(l.id);
                      return (
                        <button key={l.id} onClick={() => toggleLevel(f.id, l.id)} aria-pressed={on}
                          title={on ? "Included in the experiment (keep at least 2)" : "Not included"}
                          className={cn("inline-flex h-7 items-center gap-1.5 border-[1.5px] px-2 text-[12.5px] font-medium transition-colors", on ? "border-hot bg-paper text-ink" : "border-line bg-paper text-ink-3 hover:border-ink")}>
                          <span aria-hidden className={cn("grid h-3 w-3 place-items-center border-[1.5px] text-[9px] leading-none text-paper", on ? "border-hot bg-hot" : "border-ink-3")}>{on ? "✓" : ""}</span>
                          {l.label}
                        </button>
                      );
                    })}
                  </div>
                  <Segmented size="sm" value={varied ? "compare" : "fixed"} onChange={(v) => { if ((v === "compare") !== varied) toggleVary(f.id); }}
                    options={[{ id: "fixed", label: "Fixed" }, { id: "compare", label: "Compare", disabled: full, hint: full ? "Already comparing 3 variables" : undefined }]} />
                </div>
              );
            })}
          </div>

          {/* design */}
          <div className="mt-8">
            <SectionTitle><span className="num mr-2 text-hot">2</span>Design matrix</SectionTitle>
            {design && design.factors.length > 0 ? (
              <>
                <p className="text-[16px] leading-snug tracking-[-0.01em]">
                  {design.factors.map((f, i) => (
                    <span key={f}>{i > 0 && " × "}<span className="num font-bold">{vary[f]?.length}</span> {fLabel(f).toLowerCase()} levels</span>
                  ))}
                  {" = "}<span className="num font-bold">{design.cells.length}</span> cells
                  {" × "}<span className="num font-bold">{nVariants}</span> surface variants
                  {" = "}<span className="num font-bold text-hot">{design.n_units}</span> prompts
                </p>
                <p className="mt-1 mb-3 max-w-[640px] t-small text-ink-2">
                  Each row is one <span className="font-bold text-ink">cell</span>: the same message with one combination of the compared levels; every fixed variable stays the same.
                  Each cell is then written {nVariants} ways (names, paraphrases, punctuation), and the model reads every version.
                  {design.excluded.length > 0 && ` ${design.excluded.length} combination${design.excluded.length > 1 ? "s" : ""} that would contradict each other ${design.excluded.length > 1 ? "are" : "is"} left out.`}
                  {" "}Click a row to see that cell on the left.
                </p>
              </>
            ) : (
              <p className="mb-3 t-small text-ink-2">Set at least one variable to <span className="font-bold text-ink">Compare</span> to build an experiment.</p>
            )}
            <div className="mb-3 flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 t-small"><span className="text-ink-2">Surface variants per cell</span>
                <Segmented size="sm" value={String(nVariants)} onChange={(v) => { setNVariants(Number(v)); setRun(null); }} options={[2, 3, 4, 6].map((n) => ({ id: String(n), label: String(n) }))} />
              </label>
              <label className="flex items-center gap-2 t-small"><span className="text-ink-2">Outcome</span>
                <select value={keyIntent} onChange={(e) => { setOutcome(e.target.value); setRun(null); }} className="h-7 border-[1.5px] border-ink bg-paper px-1.5 text-[12.5px] font-medium">
                  {dom?.interpretations.map((i) => <option key={i.id} value={i.id}>P({i.label})</option>)}
                </select>
              </label>
            </div>
            <ErrorNote error={designErr} />
            {design && design.factors.length > 0 && (
              <div className="max-h-[260px] overflow-auto quiet-scroll border-y-[1.5px] border-ink">
                <table className="w-full border-collapse t-small">
                  <thead className="sticky top-0 bg-paper"><tr className="border-b border-ink text-left text-ink-2">
                    <th className="py-1.5 pr-2 font-medium">#</th>
                    {design.factors.map((f) => <th key={f} className="pr-2 font-medium">{fLabel(f)}</th>)}
                    <th className="pr-2 text-right font-medium">Hypothesis</th><th className="text-right font-medium">Model</th>
                  </tr></thead>
                  <tbody>
                    {design.cells.map((c, i) => {
                      const r = an?.cells.find((x) => x.cell === c.cell);
                      return (
                        <tr key={c.cell} onClick={() => setView(c.assignment)} className={cn("cursor-pointer border-b border-line hover:bg-wash", Object.entries(c.assignment).every(([k, v]) => shown[k] === v) && "bg-wash")}>
                          <td className="num py-1 pr-2 text-ink-3">{i + 1}</td>
                          {design.factors.map((f) => <td key={f} className="pr-2">{lvLabel(f, c.assignment[f])}</td>)}
                          <td className="num pr-2 text-right text-ink-2">{r ? pct(r.y_ref) : "—"}</td>
                          <td className="num text-right font-bold text-hot">{r ? pct(r.y_mean) : ""}</td>
                        </tr>
                      );
                    })}
                    {design.excluded.map((x, i) => (
                      <tr key={`x${i}`} className="border-b border-line text-ink-3" title={x.reasons.join(" ")}>
                        <td className="py-1 pr-2">×</td>{design.factors.map((f) => <td key={f} className="pr-2 line-through">{lvLabel(f, x.assignment[f])}</td>)}<td colSpan={2} className="text-right t-micro">excluded</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-4">
              <span className="num t-h3 text-hot">3</span>
              <Btn variant="hot" onClick={doRun} disabled={running || !design || !design.factors.length || !design.cells.length}>{running ? "Running…" : `Run experiment (${design?.n_units ?? 0} prompts)`}</Btn>
              {running ? <Loading label={`Inference · ${backend}`} /> : run && <ProvenanceBadge p={run.provenance} />}
            </div>
            <ErrorNote error={err} />
            {!run && !running && <p className="mt-2 t-small text-ink-3">The Hypothesis column is the authored LSSM value for each cell. The Model column fills in after the run: the model&rsquo;s average P({intentLabel(keyIntent)}) across that cell&rsquo;s surface variants.</p>}
          </div>
        </div>
      </section>

      {/* ============ RESULTS */}
      {run && an && (
        <section className="gutter mt-14">
          <div className="rule-heavy" />
          <div className="flex flex-wrap items-end justify-between gap-4 pt-5">
            <div>
              <h2 className="text-[32px] font-bold leading-tight tracking-[-0.035em]">Results · P({intentLabel(an.outcome_intent)})</h2>
              <p className="mt-1 t-small text-ink-2">{an.n_units} prompts · {run.design.cells.length} cells · {nVariants} surface variants per cell · {run.runtime_s}s · experiment <span className="mono">{run.experiment_id}</span></p>
            </div>
            <ProvenanceBadge p={run.provenance} />
          </div>

          <div className="mt-6 grid grid-cols-2 gap-6 md:grid-cols-4">
            <Stat label="Variance explained by context" value={pct(an.variance.context)} sub="cell means (social factors)" hot />
            <Stat label="…by wording & names" value={pct(an.variance.wording)} sub="surface variants — should be ~0" />
            <Stat label="Sensitivity slope" value={fmt(an.sensitivity.slope)} sub="model vs hypothesis across cells (1 = matches)" />
            <Stat label="Calibration gap" value={fmt(an.sensitivity.mean_abs_entropy_gap)} sub="mean |entropy(model) − entropy(hyp.)|" />
          </div>

          <div className="mt-10 grid grid-cols-1 gap-12 xl:grid-cols-2">
            <div>
              <SectionTitle aside={<DotLegend modelLabel="Model mean ± sd" refLabel="Hypothesis (LSSM)" />}>Cell means</SectionTitle>
              <CellDots rows={an.cells.map((c) => ({ label: cellName(c.assignment), y: c.y_mean, sd: c.y_sd, ref: c.y_ref, n: c.n }))} />
            </div>
            <div>
              <SectionTitle aside="cluster bootstrap over variants · ink tick = hypothesis">Treatment effects</SectionTitle>
              <Forest rows={an.main_effects.map((e) => ({ label: `${fLabel(e.factor)}: ${lvLabel(e.factor, e.level)} vs ${lvLabel(e.factor, e.reference_level)}`, est: e.model.estimate, lo: e.model.lo, hi: e.model.hi, hyp: e.hypothesis, sig: e.model.significant }))} />
              {an.main_effects.some((e) => e.direction_agrees === false) && <p className="mt-2 t-small text-hot">Direction disagrees with the hypothesis for: {an.main_effects.filter((e) => e.direction_agrees === false).map((e) => `${fLabel(e.factor)}=${lvLabel(e.factor, e.level)}`).join(", ")}</p>}
              {an.interactions.length > 0 && (
                <div className="mt-6">
                  <Label className="mb-1">Interaction (difference-in-differences)</Label>
                  <Forest rows={an.interactions.map((x) => ({ label: `${lvLabel(x.factors[0], x.levels[0])} × ${lvLabel(x.factors[1], x.levels[1])}`, est: x.model.estimate, lo: x.model.lo, hi: x.model.hi, hyp: x.hypothesis, sig: x.model.significant }))} />
                </div>
              )}
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-12 xl:grid-cols-3">
            {interactionData && (
              <div>
                <SectionTitle>Interaction plot</SectionTitle>
                <InteractionPlot data={interactionData.rows} xLabel={fLabel(interactionData.f1)} series={interactionData.series} />
                <p className="t-micro text-ink-3">Solid = model, dashed = hypothesis; one line per level of {fLabel(interactionData.f2)}. Parallel lines = no interaction.</p>
              </div>
            )}
            <div>
              <SectionTitle>Context vs wording</SectionTitle>
              <StackBar parts={[{ label: "Social context", value: an.variance.context, color: "var(--pg-accent)" }, { label: "Wording & names", value: an.variance.wording, color: "var(--pg-ink)" }, { label: "Residual", value: an.variance.residual, color: "var(--pg-line)" }]} />
              <p className="mt-3 t-micro text-ink-3">{an.variance.note}</p>
            </div>
            <div>
              <SectionTitle aside={`r = ${fmt(an.sensitivity.corr)}`}>Sensitivity to context</SectionTitle>
              <SensitivityScatter points={an.cells.map((c) => ({ x: c.y_ref, y: c.y_mean, label: cellName(c.assignment) }))} slope={an.sensitivity.slope} intercept={an.sensitivity.intercept} />
              <p className="t-micro text-ink-3">Each point is a cell. Dashed: identity. A flat accent line means the model ignores context the hypothesis treats as informative.</p>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-12 xl:grid-cols-2">
            <div>
              <SectionTitle aside="paired by variant: same names & wording">Paired comparison</SectionTitle>
              {pair && (
                <div className="flex flex-wrap items-center gap-2 t-small">
                  <select value={pair[0]} onChange={(e) => setPair([e.target.value, pair[1]])} className="h-8 max-w-[45%] border-[1.5px] border-ink bg-paper px-1.5">{an.cells.map((c) => <option key={c.cell} value={c.cell}>{cellName(c.assignment)}</option>)}</select>
                  <span>→</span>
                  <select value={pair[1]} onChange={(e) => setPair([pair[0], e.target.value])} className="h-8 max-w-[45%] border-[1.5px] border-ink bg-paper px-1.5">{an.cells.map((c) => <option key={c.cell} value={c.cell}>{cellName(c.assignment)}</option>)}</select>
                </div>
              )}
              {pairStats && (
                <div className="mt-4 grid grid-cols-3 gap-4">
                  <Stat label="Mean paired Δ" value={`${pairStats.mean >= 0 ? "+" : "−"}${Math.abs(pairStats.mean * 100).toFixed(1)} pp`} hot />
                  <Stat label="95% bootstrap CI" value={pairStats.lo == null ? "n<3" : `[${(pairStats.lo * 100).toFixed(1)}, ${(pairStats.hi! * 100).toFixed(1)}]`} sub="percentage points" />
                  <Stat label="Pairs" value={pairStats.n} sub="variants present in both cells" />
                </div>
              )}
            </div>
            <div>
              <SectionTitle aside={an.ols.se_type}>Regression (OLS, main effects + 2-way interactions)</SectionTitle>
              {an.ols.error ? <p className="t-small text-ink-3">{an.ols.error}</p> : (
                <div className="max-h-[260px] overflow-auto quiet-scroll">
                  <table className="w-full t-small"><thead><tr className="border-b border-ink text-left text-ink-2"><th className="py-1 font-medium">Term</th><th className="text-right font-medium">β</th><th className="text-right font-medium">95% CI</th><th className="text-right font-medium">p</th></tr></thead>
                    <tbody>{an.ols.coefficients?.map((c) => (
                      <tr key={c.term} className="border-b border-line"><td className="py-1 pr-2 mono text-[11px]">{c.term}</td><td className="num text-right">{c.coef.toFixed(3)}</td><td className="num whitespace-nowrap pl-2 text-right text-ink-2">[{c.lo.toFixed(2)}, {c.hi.toFixed(2)}]</td><td className={cn("num text-right", c.p < 0.05 && "font-bold text-hot")}>{c.p < 0.001 ? "<.001" : c.p.toFixed(3)}</td></tr>
                    ))}</tbody></table>
                  <p className="mt-1 t-micro text-ink-3">R² = {fmt(an.ols.r2)} · n = {an.ols.n}</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-10 grid grid-cols-1 gap-x-10 gap-y-2 border-t border-line pt-4 t-small text-ink-2 md:grid-cols-2">
            <p><span className="font-bold text-ink">Human–model divergence.</span> {an.human.available ? "See table." : an.human.note}</p>
            <p><span className="font-bold text-ink">Scope.</span> {an.caveat} Bootstrap CIs resample surface variants (clusters); with fewer than 3 variants no CI is reported.</p>
          </div>
        </section>
      )}
    </div>
  );
}
