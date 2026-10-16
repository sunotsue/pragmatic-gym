"use client";

import { useEffect, useState } from "react";
import { API, api, fmt, pct, type Dist, type Message } from "@/lib/api";
import { useAppState } from "@/components/pg/app-state";
import { Thread } from "@/components/pg/chat";
import { Btn, DistributionRows, DotLegend, ErrorNote, Kv, Label, Loading, PageHead, SectionTitle, Segmented, Stat, SyntheticTag } from "@/components/pg/primitives";
import { cn } from "@/lib/utils";

interface Row { id: string; family_id: string; domain: string; capability: string; source: string; cluster: string | null; contrast_factor: string | null; factors: Record<string, string> | null; speaker: string; relationship: string; context: string[]; history: Message[]; reference: Dist; target_action: string; reason: string; tokens: number; base: { probs: Dist; action: string; failures: string[]; cross_entropy: number } | null; turn?: number; behaviour_action?: string; behaviour_response?: string; behaviour_reward_components?: Record<string, number> }
interface Diag { n: number; tokens: number; leakage: { n_leaks: number; examples: { id: string; pattern: string }[]; checked_patterns: string[] }; duplicates: { exact: number; "near_jaccard_0.9": number }; distribution: Record<string, Record<string, number>>; overlap: Record<string, { family_overlap: string[]; cell_overlap: number; near_duplicate_items: number; expected_family_overlap: boolean }>; factor_coverage: Record<string, Record<string, number>>; warnings: string[] }
interface Selection { selection_id: string; strategy: string; budget: number; tokens_used: number; n: number; info: string; rows: Row[]; diagnostics: Diag; include_context: boolean }
interface Compare { [k: string]: { n: number; tokens: number; mean_base_loss: number | null; base_failure_rate: number | null; n_clusters: number; n_contrast_pairs: number; diagnostics: Pick<Diag, "leakage" | "duplicates" | "distribution" | "warnings" | "factor_coverage">; error?: string } }
interface PoolDiag { available: boolean; n: number; model: string; primary: Record<string, number>; any: Record<string, number>; cross_entropy_by_domain: Record<string, number> }

const STRATS = [
  { id: "random", label: "Random" }, { id: "diversity", label: "Diversity" }, { id: "failure_targeted", label: "Failure-targeted" },
  { id: "counterfactual_contrast", label: "Counterfactual contrast" }, { id: "isolated", label: "Isolated (ablation)" },
];
const HYPOTHESES = [
  "Does training on context-rich indirect refusals improve pragmatic interpretation more than training on isolated refusal examples?",
  "Does failure-targeted selection produce greater generalization than random sampling at the same token budget?",
  "Do minimal-pair contrasts — the same conversation with one social fact changed — teach context sensitivity rather than surface cues?",
];

export default function DataLab() {
  const { meta } = useAppState();
  const [strategy, setStrategy] = useState("failure_targeted");
  const [target, setTarget] = useState<string>("");
  const [budget, setBudget] = useState(240000);
  const [sel, setSel] = useState<Selection | null>(null);
  const [cmp, setCmp] = useState<Compare | null>(null);
  const [pool, setPool] = useState<PoolDiag | null>(null);
  const [info, setInfo] = useState<{ info: Record<string, string>; pool_size: number; base_scored: number } | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [exported, setExported] = useState<{ manifest: Record<string, unknown>; path: string; sample_sft: unknown[]; sample_preference: unknown; downloads: Record<string, string> } | null>(null);

  useEffect(() => {
    api<PoolDiag>("/api/data/pool_diagnosis").then(setPool).catch(() => {});
    api<{ info: Record<string, string>; pool_size: number; base_scored: number }>("/api/data/strategies").then(setInfo).catch((e) => setErr(String(e)));
  }, []);

  const doSelect = async (s = strategy) => {
    setBusy("select"); setErr(null); setExported(null);
    try { const r = await api<Selection>("/api/data/select", { body: { strategy: s, budget, seed: 0, target_failure: target || null } }); setSel(r); setOpen(r.rows[0]?.id ?? null); }
    catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const doCompare = async () => {
    setBusy("compare"); setErr(null);
    try { setCmp(await api<Compare>("/api/data/compare", { body: { strategy, budget, seed: 0, target_failure: target || null } })); }
    catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const doExport = async () => {
    setBusy("export");
    try {
      const r = await api<NonNullable<typeof exported>>("/api/data/export", { body: { strategy, budget, seed: 0, target_failure: target || null } });
      setExported(r);
      const a = document.createElement("a");
      a.href = `${API}${r.downloads["sft.jsonl"]}`;
      a.click();
    }
    catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  useEffect(() => { doSelect(); doCompare(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const failureKeys = pool ? Object.entries(pool.any).sort((a, b) => b[1] - a[1]).map(([k]) => k).filter((k) => k !== "pass") : [];
  const domOf = (d: string) => meta?.domains[d];

  return (
    <div>
      <PageHead
        n="04"
        title="Data Intervention Lab"
        lead={<>Build training sets aimed at the model&rsquo;s mistakes. Each strategy picks examples from the same pool and stops at the same size, so the only difference between them is <em>which</em> examples they pick.</>}
        measures={["base-model failure profile", "what each strategy selects, and why", "leakage, duplicates and imbalance", "train/eval overlap"]}
      />

      <section className="gutter grid grid-cols-1 gap-10 xl:grid-cols-[1fr_1fr]">
        <div>
          <SectionTitle aside={pool?.available ? `${pool.model.split("/")[1]} on ${pool.n} pool items` : undefined}>Diagnosed failure profile (base model)</SectionTitle>
          {pool?.available ? (
            <div className="space-y-1.5">
              {Object.entries(pool.any).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                <div key={k} className="grid grid-cols-[170px_1fr_60px] items-center gap-3 t-small">
                  <span className={cn(k === "pass" ? "text-ink-2" : "font-bold")}>{k.replaceAll("_", " ")}</span>
                  <span className="relative h-3 bg-wash"><span className="absolute h-full" style={{ width: `${(v / pool.n) * 100}%`, background: k === "pass" ? "var(--pg-ink-3)" : "var(--pg-accent)" }} /></span>
                  <span className="num text-right">{pct(v / pool.n)}</span>
                </div>
              ))}
              <p className="pt-1 t-micro text-ink-3">Share of pool items with each flag (items can carry several). Measured once by local inference over the candidate pool (saved); used only by the failure-targeted strategy.</p>
            </div>
          ) : <p className="t-small text-ink-3">Base-model pool scores not found — run <span className="mono">scripts/score_pool.py</span>.</p>}
        </div>
        <div>
          <SectionTitle>Hypotheses this lab can test</SectionTitle>
          <ol className="space-y-3">
            {HYPOTHESES.map((h, i) => <li key={i} className="grid grid-cols-[28px_1fr] text-[15px] leading-snug"><span className="num font-medium text-hot">H{i + 1}</span><span>{h}</span></li>)}
          </ol>
          <p className="mt-3 t-small text-ink-2">Pool: {info?.pool_size ?? "…"} candidates from <span className="font-bold text-ink">training families only</span> (contrast clusters + multi-turn Soft No states). Unseen families, the family-subtext domain and every hand-written scenario are held out.</p>
        </div>
      </section>

      <section className="gutter mt-12">
        <div className="rule-heavy" />
        <div className="flex flex-wrap items-end gap-5 py-5">
          <div><Label className="mb-1">Selection strategy</Label><Segmented value={strategy} onChange={(v) => { setStrategy(v); doSelect(v); }} options={STRATS} /></div>
          <div>
            <Label className="mb-1">Target failure</Label>
            <select value={target} onChange={(e) => setTarget(e.target.value)} className="h-8 border-[1.5px] border-ink bg-paper px-2 text-[13px]" disabled={strategy !== "failure_targeted"}>
              <option value="">Any failure</option>
              {failureKeys.map((k) => <option key={k} value={k}>{k.replaceAll("_", " ")}</option>)}
            </select>
          </div>
          <div>
            <Label className="mb-1">Token budget</Label>
            <Segmented value={String(budget)} onChange={(v) => setBudget(Number(v))} options={[60000, 120000, 240000].map((b) => ({ id: String(b), label: `${b / 1000}k` }))} />
          </div>
          <Btn onClick={() => { doSelect(); doCompare(); }} disabled={!!busy}>{busy === "select" || busy === "compare" ? "Selecting…" : "Build selection"}</Btn>
          <Btn variant="outline" onClick={doExport} disabled={!!busy || !sel}>{busy === "export" ? "Exporting…" : "Export JSONL"}</Btn>
        </div>
        <ErrorNote error={err} />
        {info && <p className="max-w-[900px] t-small text-ink-2"><span className="font-bold text-ink">{STRATS.find((s) => s.id === strategy)?.label}.</span> {info.info[strategy]}</p>}

        {/* compare table */}
        {cmp && (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[900px] border-collapse t-small">
              <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2">
                <th className="py-2 font-medium">Strategy</th><th className="text-right font-medium">Examples</th><th className="text-right font-medium">Tokens</th><th className="text-right font-medium">Base loss</th><th className="text-right font-medium">Base failure rate</th><th className="text-right font-medium">Contrast pairs</th><th className="pl-4 font-medium">Domain mix</th><th className="pl-4 font-medium">Diagnostics</th>
              </tr></thead>
              <tbody>
                {STRATS.map((s) => {
                  const r = cmp[s.id];
                  if (!r || r.error) return <tr key={s.id} className="border-b border-line"><td className="py-2">{s.label}</td><td colSpan={7} className="text-ink-3">{r?.error}</td></tr>;
                  const doms = r.diagnostics.distribution.domain;
                  const tot = Object.values(doms).reduce((a, b) => a + b, 0);
                  return (
                    <tr key={s.id} onClick={() => { setStrategy(s.id); doSelect(s.id); }} className={cn("cursor-pointer border-b border-line hover:bg-wash", strategy === s.id && "bg-wash font-bold")}>
                      <td className="py-2">{s.label}</td>
                      <td className="num text-right">{r.n}</td>
                      <td className="num text-right">{(r.tokens / 1000).toFixed(0)}k</td>
                      <td className="num text-right">{fmt(r.mean_base_loss)}</td>
                      <td className="num text-right">{pct(r.base_failure_rate)}</td>
                      <td className="num text-right">{r.n_contrast_pairs}</td>
                      <td className="pl-4"><div className="flex h-2.5 w-[160px]">{Object.entries(doms).map(([d, n], i) => <span key={d} title={`${d}: ${n}`} style={{ width: `${(n / tot) * 100}%`, background: ["var(--pg-accent)", "var(--pg-ink)", "var(--pg-ink-3)", "#ffb399"][i % 4] }} className="h-full border-r border-paper" />)}</div></td>
                      <td className="pl-4 t-micro font-normal">
                        <span className={r.diagnostics.leakage.n_leaks ? "text-hot" : "text-ink-2"}>leaks {r.diagnostics.leakage.n_leaks}</span> · <span className="text-ink-2">dups {r.diagnostics.duplicates.exact}/{r.diagnostics.duplicates["near_jaccard_0.9"]}</span>
                        {r.diagnostics.warnings.length > 0 && <span className="text-hot"> · {r.diagnostics.warnings.length} warning{r.diagnostics.warnings.length > 1 ? "s" : ""}</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 t-micro text-ink-3">Base loss = mean cross-entropy of the base model against the reference on the selected items. Domain mix: invitation · relationship · emotional. The isolated arm has many duplicates by construction (utterances without context repeat).</p>
          </div>
        )}
      </section>

      {/* selection detail */}
      {sel && (
        <section className="gutter mt-10 grid grid-cols-1 gap-10 xl:grid-cols-[minmax(420px,1fr)_minmax(420px,1fr)]">
          <div className="min-w-0">
            <SectionTitle aside={`${sel.n} examples · ${(sel.tokens_used / 1000).toFixed(0)}k / ${sel.budget / 1000}k tokens`}>Why each example was included</SectionTitle>
            <div className="max-h-[720px] overflow-auto quiet-scroll border-t-[1.5px] border-ink">
              {sel.rows.map((r) => (
                <button key={r.id} onClick={() => setOpen(r.id)} className={cn("block w-full border-b border-line px-1 py-2.5 text-left transition-colors", open === r.id ? "bg-wash" : "hover:bg-wash/60")}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[13.5px] font-bold tracking-[-0.01em]">{r.relationship} · “{r.history[r.history.length - 1]?.text}”</span>
                    <span className="num shrink-0 t-micro text-ink-3">{r.source === "rollout" ? `rollout t${r.turn}` : r.family_id}</span>
                  </div>
                  <div className="mt-0.5 t-micro text-ink-2">{r.reason}</div>
                </button>
              ))}
            </div>
          </div>
          <div className="min-w-0">
            {(() => {
              const r = sel.rows.find((x) => x.id === open);
              if (!r) return null;
              const dom = domOf(r.domain);
              return (
                <div>
                  <SectionTitle aside={<SyntheticTag />}>Training example</SectionTitle>
                  <div className="border-[1.5px] border-ink p-4">
                    {sel.include_context ? (
                      <>
                        <ul className="mb-3 space-y-0.5 bg-wash px-3 py-2">{r.context.map((c, i) => <li key={i} className="text-[13.5px]">· {c}</li>)}</ul>
                        <Thread history={r.history} channel="private" />
                      </>
                    ) : (
                      <>
                        <p className="mb-2 t-micro text-hot">Isolated arm: the model sees only the utterance below — no relationship, history or context.</p>
                        <Thread history={r.history.slice(-1)} channel="private" />
                      </>
                    )}
                  </div>
                  <div className="mt-5">
                    <SectionTitle aside={<DotLegend modelLabel="Base model" refLabel="Training target (posterior)" />}>Target vs. base model</SectionTitle>
                    {dom && <DistributionRows labels={dom.interpretations} model={r.base?.probs} reference={r.reference} />}
                    <p className="mt-2 t-micro text-ink-3">Targets are the reference distribution given observable context (for rollout states, the environment&apos;s exact Bayes posterior) — never the sampled hidden intent. SFT samples 4 labels per example from it.</p>
                  </div>
                  <div className="mt-5">
                    <Kv k="Target move" v={<span className="font-bold">{dom?.actions.find((a) => a.id === r.target_action)?.label}</span>} />
                    <Kv k="Base model move" v={dom?.actions.find((a) => a.id === r.base?.action)?.label ?? "—"} />
                    <Kv k="Base failures" v={r.base?.failures.join(", ") || "none"} />
                    <Kv k="Capability" v={meta?.capabilities[r.capability]?.label} />
                    {r.contrast_factor && <Kv k="Contrast" v={`one-factor counterfactual of cluster ${r.cluster}: ${r.contrast_factor}`} />}
                    {r.source === "rollout" && <Kv k="Trajectory" v={<>behaviour move <span className="mono">{r.behaviour_action}</span> → <span className="mono">{r.behaviour_response}</span>; reward components {Object.entries(r.behaviour_reward_components ?? {}).filter(([, v]) => v).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ") || "0"}</>} />}
                    <Kv k="Provenance" v={<span className="mono">{r.id}</span>} />
                  </div>
                </div>
              );
            })()}
          </div>
        </section>
      )}

      {/* diagnostics */}
      {sel && (
        <section className="gutter mt-12 grid grid-cols-1 gap-10 lg:grid-cols-3">
          <div>
            <SectionTitle>Leakage</SectionTitle>
            <Stat label="Latent identifiers in model-visible text" value={sel.diagnostics.leakage.n_leaks} hot={sel.diagnostics.leakage.n_leaks > 0} sub={`checked: ${sel.diagnostics.leakage.checked_patterns.length} patterns incl. hidden-state keys, factor & intent ids`} />
          </div>
          <div>
            <SectionTitle>Duplicates & imbalance</SectionTitle>
            <Kv k="Exact duplicates" v={sel.diagnostics.duplicates.exact} />
            <Kv k="Near (Jaccard ≥ .9)" v={sel.diagnostics.duplicates["near_jaccard_0.9"]} />
            {sel.diagnostics.warnings.map((w) => <p key={w} className="mt-1 t-small text-hot">{w}</p>)}
            <div className="mt-2">{Object.entries(sel.diagnostics.distribution.target_interpretation).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => <Kv key={k} k={k.replaceAll("_", " ")} v={<span className="num">{pct(v / sel.n)}</span>} />)}</div>
          </div>
          <div>
            <SectionTitle>Train / evaluation overlap</SectionTitle>
            {Object.entries(sel.diagnostics.overlap).map(([k, v]) => (
              <Kv key={k} k={k.replace("eval_", "")} v={<span className="t-small">families: {v.family_overlap.length ? (v.expected_family_overlap ? `${v.family_overlap.length} (by design)` : <span className="text-hot">{v.family_overlap.join(", ")}</span>) : "none"} · cells {v.cell_overlap} · near-dups {v.near_duplicate_items}</span>} />
            ))}
            <p className="mt-2 t-micro text-ink-3">Split is at the family level: in-distribution eval shares families but no factor cell; other sets share nothing.</p>
          </div>
        </section>
      )}

      {exported && (
        <section className="gutter mt-10">
          <SectionTitle aside={<span className="mono">{exported.path}</span>}>Exported</SectionTitle>
          <p className="mb-3 t-small text-ink-2">
            Downloaded <span className="font-bold text-ink">sft.jsonl</span> ({String(exported.manifest.n_sft)} rows).{" "}
            <a href={`${API}${exported.downloads["sft.jsonl"]}`} className="font-bold text-ink underline underline-offset-4">Download SFT again</a>
            {" · "}
            <a href={`${API}${exported.downloads["preference.jsonl"]}`} className="font-bold text-ink underline underline-offset-4">Download preference pairs</a> ({String(exported.manifest.n_pref)} pairs)
          </p>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <pre className="max-h-[320px] overflow-auto quiet-scroll bg-wash p-3 mono text-[11px] leading-snug">{JSON.stringify(exported.manifest, null, 2)}</pre>
            <pre className="max-h-[320px] overflow-auto quiet-scroll bg-wash p-3 mono text-[11px] leading-snug">{JSON.stringify(exported.sample_sft[0], null, 2)}</pre>
          </div>
        </section>
      )}
      {busy === "select" && !sel && <div className="gutter mt-6"><Loading label="Selecting" /></div>}
    </div>
  );
}
