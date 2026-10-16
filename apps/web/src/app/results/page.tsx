"use client";

import { useEffect, useMemo, useState } from "react";
import { api, fmt, pct } from "@/lib/api";
import { useAppState } from "@/components/pg/app-state";
import { ErrorNote, Kv, Label, Loading, PageHead, SectionTitle, Segmented, SyntheticTag } from "@/components/pg/primitives";
import { cn } from "@/lib/utils";

interface MetricAgg { mean: number | null; seed_values: number[]; seed_sd?: number | null; item_ci?: [number | null, number | null] | null }
interface SplitAgg { [k: string]: MetricAgg | Record<string, number> }
interface Result {
  experiment_id: string; label: string; reference_note: string;
  meta: { base_model: string; budget: number; seeds_completed: number[]; arms: string[]; lora: Record<string, unknown>; train: Record<string, unknown>; eval_sizes: Record<string, number>; target_failure: string | null };
  summary: Record<string, { n_seeds: number; regression_accuracy: number; splits: Record<string, SplitAgg> }>;
  tests: Record<string, { treatment: string; control: string; metric?: string; diff_treat_minus_control: number; ci: [number, number]; p_value: number; p_holm?: number; n_items?: number }>;
  reference_baselines?: Record<string, Record<string, { cross_entropy: number; kl: number; top1_decisive: number }>>;
  train_meta: Record<string, ({ n_examples: number; train_tokens: number; steps: number; train_loss: number; runtime_s: number; seed: number; dataset_manifest?: { sha1: string; n_items: number; tokens_used: number } } | null)[]>;
}

const ARM_LABEL: Record<string, string> = { base: "Base (no SFT)", random: "Random", diversity: "Diversity", failure_targeted: "Failure-targeted", counterfactual_contrast: "Counterfactual contrast", isolated: "Isolated (no context)" };
const SPLIT_LABEL: Record<string, string> = { eval_id: "In-distribution (held-out cells)", eval_ood_family: "Unseen utterance families", eval_ood_domain: "Unseen domain (family subtext)", eval_handwritten: "Hand-written corpus" };
const METRICS: { id: string; label: string; better: "low" | "high"; fmt: (x: number | null) => string; note: string }[] = [
  { id: "kl", label: "Excess loss (KL)", better: "low", fmt: (x) => fmt(x, 3), note: "KL(reference ‖ model) in nats = cross-entropy minus the reference entropy floor." },
  { id: "top1_decisive", label: "Top-1 (decisive)", better: "high", fmt: (x) => pct(x, 1), note: "Agreement with the reference's top reading, only on items where the reference is decisive." },
  { id: "context_sensitivity_r", label: "Context sensitivity r", better: "high", fmt: (x) => fmt(x, 2), note: "Correlation between how the model's beliefs move and how the reference moves when one social fact changes." },
  { id: "sensitivity_ratio", label: "Sensitivity ratio", better: "high", fmt: (x) => fmt(x, 2), note: "Mean |Δ model| / mean |Δ reference| across one-factor contrasts. 1 = moves as much as the reference." },
  { id: "overconfident_on_ambiguous", label: "Overconfident (ambiguous)", better: "low", fmt: (x) => pct(x, 1), note: "Share of genuinely ambiguous items where the model puts > 70% on one reading." },
  { id: "action_regret", label: "Action regret", better: "low", fmt: (x) => fmt(x, 3), note: "Expected-utility gap between the chosen move and the best move under the reference uncertainty." },
  { id: "harmful_action_rate", label: "Harmful moves", better: "low", fmt: (x) => pct(x, 1), note: "Share of items where the chosen move has negative expected utility (e.g. pressure)." },
  { id: "unwarranted_deference_rate", label: "Unwarranted deference", better: "low", fmt: (x) => pct(x, 1), note: "Choosing an over-apologetic/deferential move that is not acceptable. Guards against style imitation." },
  { id: "paraphrase_tvd", label: "Paraphrase instability", better: "low", fmt: (x) => fmt(x, 3), note: "TVD between beliefs on the same cell rendered with different names/wording. Reference is 0 by construction." },
];

function verdict(t: { diff_treat_minus_control: number; p_holm?: number; p_value: number }) {
  const p = t.p_holm ?? t.p_value;
  if (p < 0.05) return t.diff_treat_minus_control < 0 ? { text: "Reject H₀ — treatment lower loss", hot: true } : { text: "Reject H₀ — treatment higher loss", hot: false };
  return { text: "No detectable difference", hot: false };
}

export default function Results() {
  const { meta } = useAppState();
  const [list, setList] = useState<{ experiment_id: string; label: string; meta: Result["meta"]; arms: string[] }[] | null>(null);
  const [eid, setEid] = useState<string | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [cfg, setCfg] = useState<{ config: Record<string, unknown>; commands: Record<string, string> } | null>(null);
  const [metric, setMetric] = useState("kl");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<typeof list>("/api/results").then((l) => { setList(l); const pick = l?.find((x) => x.experiment_id === "social-sft-v1") ?? l?.[0]; if (pick) setEid(pick.experiment_id); }).catch((e) => setErr(String(e)));
    api<typeof cfg>("/api/training/config").then(setCfg).catch(() => {});
  }, []);
  useEffect(() => { if (eid) api<Result>(`/api/results/${eid}`).then(setRes).catch((e) => setErr(String(e))); }, [eid]);

  const arms = res ? Object.keys(res.summary) : [];
  const splits = res ? Object.keys(res.summary[arms[0]]?.splits ?? {}) : [];
  const M = METRICS.find((m) => m.id === metric)!;
  const get = (a: string, s: string, k: string) => (res?.summary[a]?.splits[s]?.[k] as MetricAgg | undefined);

  const ranges = useMemo(() => {
    if (!res) return {} as Record<string, [number, number]>;
    const out: Record<string, [number, number]> = {};
    for (const s of splits) {
      const vals = arms.flatMap((a) => { const g = get(a, s, metric); return g ? [...(g.seed_values ?? []), g.mean ?? NaN] : []; }).filter((x) => Number.isFinite(x));
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const pad = (hi - lo) * 0.15 || 0.05;
      out[s] = [lo - pad, hi + pad];
    }
    return out;
  }, [res, metric]); // eslint-disable-line react-hooks/exhaustive-deps

  const caps = res ? Object.keys((res.summary[arms[0]]?.splits.eval_handwritten?.cross_entropy_by_capability as Record<string, number>) ?? {}) : [];

  return (
    <div>
      <PageHead
        n="05"
        title="Post-Training & Results"
        lead={<>Check whether the selected training data actually helped. Same base checkpoint, LoRA configuration, token budget and seeds; only the <em>selected experiences</em> differ.</>}
        measures={["fit to reference (pre-registered tests)", "context sensitivity", "calibration", "transfer to unseen utterances and domains"]}
        right={list && list.length > 0 ? <div className="flex flex-col items-end gap-1"><Label>Experiment</Label><select value={eid ?? ""} onChange={(e) => setEid(e.target.value)} className="h-8 border-[1.5px] border-ink bg-paper px-2 text-[13px] font-bold">{list.map((l) => <option key={l.experiment_id} value={l.experiment_id}>{l.experiment_id}</option>)}</select></div> : undefined}
      />
      <ErrorNote error={err} />
      {list && list.length === 0 && (
        <section className="gutter"><div className="border-[1.5px] border-dashed border-ink p-6"><p className="t-h3">No completed training runs yet</p><p className="mt-2 t-small text-ink-2">Nothing is shown until a real run finishes. Start one with <span className="mono">{cfg?.commands.run_all}</span>.</p></div></section>
      )}
      {!res && list && list.length > 0 && <div className="gutter"><Loading label="Loading results" /></div>}

      {res && (
        <>
          <section className="gutter">
            <div className="flex flex-wrap items-center gap-3 bg-ink px-4 py-2.5 text-paper">
              <span className="h-2.5 w-2.5 rounded-full bg-hot" />
              <span className="t-small font-bold">{res.label}</span>
              <span className="t-small text-paper/70">· seeds completed: {res.meta.seeds_completed.join(", ")} · budget {res.meta.budget / 1000}k tokens/arm · {Object.values(res.meta.eval_sizes).reduce((a, b) => a + b, 0)} eval items</span>
            </div>
            <p className="mt-2 t-micro text-ink-3">{res.reference_note}</p>
          </section>

          {/* hypothesis tests */}
          <section className="gutter mt-8">
            <SectionTitle aside="paired bootstrap over items (seed-averaged) · Holm–Bonferroni">Pre-registered comparisons</SectionTitle>
            <div className="grid grid-cols-1 gap-0 border-y-[1.5px] border-ink md:grid-cols-3">
              {(["H1_contrast_vs_random", "H2_context_vs_isolated", "H3_failure_vs_random"] as const).map((k, i) => {
                const t = res.tests[k];
                return (
                  <div key={k} className={cn("p-5", i < 2 && "md:border-r md:border-line")}>
                    <div className="num t-small text-hot">H{i + 1}</div>
                    <div className="mt-1 text-[18px] font-bold leading-tight tracking-[-0.025em]">{t ? `${ARM_LABEL[t.treatment]} vs ${ARM_LABEL[t.control]}` : ["Contrast vs random", "Context-rich vs isolated", "Failure-targeted vs random"][i]}</div>
                    {t ? (
                      <>
                        <div className="num mt-4 text-[34px] font-medium leading-none tracking-[-0.04em]">{t.diff_treat_minus_control >= 0 ? "+" : "−"}{Math.abs(t.diff_treat_minus_control).toFixed(3)}<span className="ml-1 text-[14px] text-ink-2">nats</span></div>
                        <div className="num mt-1 t-small text-ink-2">95% CI [{t.ci[0].toFixed(3)}, {t.ci[1].toFixed(3)}] · p = {t.p_value < 0.001 ? "<.001" : t.p_value.toFixed(3)}{t.p_holm != null && ` · Holm ${t.p_holm < 0.001 ? "<.001" : t.p_holm.toFixed(3)}`}</div>
                        <div className={cn("mt-3 inline-block px-2 py-[2px] t-small font-bold", verdict(t).hot ? "bg-hot text-paper" : "bg-wash")}>{verdict(t).text}</div>
                        <p className="mt-2 t-micro text-ink-3">Δ cross-entropy (treatment − control), pooled generalization sets, n = {t.n_items}. Negative = treatment better.</p>
                      </>
                    ) : <p className="mt-3 t-small text-ink-3">Planned — arm not yet trained.</p>}
                  </div>
                );
              })}
            </div>
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 t-small">
              {Object.entries(res.tests).filter(([k]) => k.startsWith("vs_base")).map(([k, t]) => (
                <span key={k}><span className="font-bold">{ARM_LABEL[t.treatment]}</span> vs base: <span className={cn("num", t.diff_treat_minus_control < 0 ? "text-ink" : "text-hot")}>{t.diff_treat_minus_control >= 0 ? "+" : "−"}{Math.abs(t.diff_treat_minus_control).toFixed(3)}</span> <span className="num text-ink-3">[{t.ci[0].toFixed(2)}, {t.ci[1].toFixed(2)}]</span></span>
              ))}
            </div>
          </section>

          {/* metric explorer */}
          <section className="gutter mt-12">
            <div className="rule-heavy" />
            <div className="flex flex-wrap items-end justify-between gap-4 pt-5">
              <h2 className="text-[32px] font-bold leading-tight tracking-[-0.035em]">{M.label}</h2>
              <Segmented size="sm" value={metric} onChange={setMetric} options={METRICS.map((m) => ({ id: m.id, label: m.label }))} />
            </div>
            <p className="mt-1 t-small text-ink-2">{M.note} <span className="text-ink-3">{M.better === "low" ? "Lower is better." : "Higher is better."} Small dots = individual seeds; large dot = mean.</span></p>
            <div className="mt-6 overflow-x-auto">
              <table className="w-full min-w-[980px] border-collapse">
                <thead><tr className="border-b-[1.5px] border-ink text-left t-small text-ink-2"><th className="w-[200px] py-2 font-medium">Arm</th>{splits.map((s) => <th key={s} className="px-3 font-medium">{SPLIT_LABEL[s]}</th>)}</tr></thead>
                <tbody>
                  {arms.map((a) => (
                    <tr key={a} className="border-b border-line">
                      <td className="py-3 pr-3"><div className="text-[14px] font-bold tracking-[-0.01em]">{ARM_LABEL[a] ?? a}</div><div className="t-micro text-ink-3">{res.summary[a].n_seeds} seed{res.summary[a].n_seeds > 1 ? "s" : ""}</div></td>
                      {splits.map((s) => {
                        const g = get(a, s, metric);
                        const [lo, hi] = ranges[s] ?? [0, 1];
                        const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
                        const best = arms.map((b) => get(b, s, metric)?.mean ?? NaN).filter(Number.isFinite);
                        const isBest = g?.mean != null && g.mean === (M.better === "low" ? Math.min(...best) : Math.max(...best));
                        return (
                          <td key={s} className="px-3">
                            <div className="flex items-center gap-3">
                              <div className="relative h-4 flex-1">
                                <div className="absolute top-1/2 h-px w-full bg-line" />
                                {g?.seed_values?.map((v, i) => <span key={i} className="absolute top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink-3" style={{ left: x(v) }} />)}
                                {g?.mean != null && <span className={cn("absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full", a === "base" ? "bg-paper ring-2 ring-ink" : "bg-hot")} style={{ left: x(g.mean) }} />}
                              </div>
                              <span className={cn("num w-[64px] text-right t-small", isBest && "font-bold")}>{M.fmt(g?.mean ?? null)}</span>
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  {res.reference_baselines && (metric === "kl" || metric === "top1_decisive") && (["uniform", "utterance_prior"] as const).map((b) => (
                    <tr key={b} className="border-b border-line bg-wash/60">
                      <td className="py-2.5 pr-3"><div className="text-[13.5px] font-bold tracking-[-0.01em] text-ink-2">{b === "uniform" ? "Uniform (no model)" : "Utterance-only prior"}</div><div className="t-micro text-ink-3">non-learned reference baseline</div></td>
                      {splits.map((s) => {
                        const v = res.reference_baselines![s]?.[b]?.[metric as "kl" | "top1_decisive"];
                        const [lo, hi] = ranges[s] ?? [0, 1];
                        const inRange = v != null && v >= lo && v <= hi;
                        return (
                          <td key={s} className="px-3">
                            <div className="flex items-center gap-3">
                              <div className="relative h-4 flex-1"><div className="absolute top-1/2 h-px w-full bg-line" />{inRange && <span className="absolute top-0 h-full w-[2px] bg-ink-2" style={{ left: `${((v! - lo) / (hi - lo)) * 100}%` }} />}</div>
                              <span className="num w-[64px] text-right t-small text-ink-2">{M.fmt(v ?? null)}</span>
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              {(metric === "kl" || metric === "top1_decisive") && <p className="mt-2 t-micro text-ink-3">Gray rows bound what is achievable without reading context. An arm that only matches the utterance-only prior has learned the words, not the social situation.</p>}
            </div>
          </section>

          {/* by capability + regression */}
          <section className="gutter mt-12 grid grid-cols-1 gap-12 xl:grid-cols-[1.3fr_0.7fr]">
            <div className="overflow-x-auto">
              <SectionTitle aside="hand-written corpus · cross-entropy (nats), lower is better">By capability</SectionTitle>
              <table className="w-full min-w-[640px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2"><th className="py-2 font-medium">Arm</th>{caps.map((c) => <th key={c} className="px-1 text-center font-medium" title={meta?.capabilities[c]?.label}>{meta?.capabilities[c]?.short ?? c}</th>)}</tr></thead>
                <tbody>
                  {arms.map((a) => {
                    const row = res.summary[a].splits.eval_handwritten?.cross_entropy_by_capability as Record<string, number>;
                    const base = res.summary.base?.splits.eval_handwritten?.cross_entropy_by_capability as Record<string, number> | undefined;
                    return (
                      <tr key={a} className="border-b border-line">
                        <td className="py-1.5 pr-2 font-bold">{ARM_LABEL[a]}</td>
                        {caps.map((c) => {
                          const d = base && a !== "base" ? row[c] - base[c] : 0;
                          return <td key={c} className="px-1 text-center"><span className="num inline-block w-full py-1" style={{ background: a === "base" ? "transparent" : d < 0 ? `rgba(255,59,0,${Math.min(0.55, -d * 1.2)})` : `rgba(17,17,17,${Math.min(0.25, d * 0.8)})` }}>{row[c]?.toFixed(2)}</span></td>;
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="mt-2 t-micro text-ink-3">Shading: improvement (accent) or regression (gray) relative to the base model.</p>
            </div>
            <div>
              <SectionTitle aside="40-item general MCQ tripwire">Regression on unrelated capabilities</SectionTitle>
              {arms.map((a) => (
                <div key={a} className="grid grid-cols-[150px_1fr_52px] items-center gap-3 py-1 t-small">
                  <span>{ARM_LABEL[a]}</span>
                  <span className="relative h-2.5 bg-wash"><span className="absolute h-full bg-ink" style={{ width: `${res.summary[a].regression_accuracy * 100}%` }} /></span>
                  <span className="num text-right">{pct(res.summary[a].regression_accuracy, 1)}</span>
                </div>
              ))}
            </div>
          </section>

          {/* metadata */}
          <section className="gutter mt-12 grid grid-cols-1 gap-12 xl:grid-cols-2">
            <div className="overflow-x-auto">
              <SectionTitle>Runs</SectionTitle>
              <table className="w-full min-w-[560px] border-collapse t-small">
                <thead><tr className="border-b-[1.5px] border-ink text-left text-ink-2"><th className="py-1.5 font-medium">Arm</th><th className="font-medium">Seed</th><th className="text-right font-medium">Items</th><th className="text-right font-medium">Train tokens</th><th className="text-right font-medium">Steps</th><th className="text-right font-medium">Loss</th><th className="text-right font-medium">Time</th><th className="pl-3 font-medium">Dataset</th></tr></thead>
                <tbody>
                  {Object.entries(res.train_meta).flatMap(([a, ms]) => ms.filter(Boolean).map((m) => (
                    <tr key={`${a}${m!.seed}`} className="border-b border-line">
                      <td className="py-1">{ARM_LABEL[a]}</td><td className="num">{m!.seed}</td><td className="num text-right">{m!.dataset_manifest?.n_items}</td><td className="num text-right">{(m!.train_tokens / 1000).toFixed(0)}k</td><td className="num text-right">{m!.steps}</td><td className="num text-right">{m!.train_loss.toFixed(3)}</td><td className="num text-right">{Math.round(m!.runtime_s)}s</td><td className="pl-3 mono text-[11px]">{m!.dataset_manifest?.sha1}</td>
                    </tr>
                  )))}
                </tbody>
              </table>
              <p className="mt-2 t-micro text-ink-3">Train tokens are measured with the tokenizer on the exported sequences (selection uses an estimate to hit the budget).</p>
            </div>
            <div>
              <SectionTitle aside={<SyntheticTag>configs/sft_lora.yaml</SyntheticTag>}>Matched configuration</SectionTitle>
              <Kv k="Base checkpoint" v={res.meta.base_model} />
              <Kv k="LoRA" v={<span className="mono break-all">{JSON.stringify(res.meta.lora)}</span>} />
              <Kv k="Training" v={<span className="mono break-all">{JSON.stringify(res.meta.train)}</span>} />
              <Kv k="Eval sets" v={Object.entries(res.meta.eval_sizes).map(([k, v]) => `${k.replace("eval_", "")} ${v}`).join(" · ")} />
              {cfg && (
                <div className="mt-4 space-y-1.5">
                  <Label>Reproduce</Label>
                  {Object.entries(cfg.commands).map(([k, v]) => <div key={k} className="bg-wash px-2.5 py-1.5"><span className="t-micro text-ink-2">{k.replaceAll("_", " ")}</span><div className="mono break-all">{v}</div></div>)}
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
