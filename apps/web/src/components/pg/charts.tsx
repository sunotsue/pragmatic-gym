"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ReferenceLine } from "recharts";
import { fmt } from "@/lib/api";

/** Horizontal forest plot: model estimate + CI (accent), hypothesis (ink tick). */
export function Forest({ rows, domain: fixed, label = "Δ probability" }: {
  rows: { label: string; est: number | null; lo: number | null; hi: number | null; hyp?: number | null; sig?: boolean }[];
  domain?: [number, number]; label?: string;
}) {
  const W = 100;
  const span = Math.max(0.05, ...rows.flatMap((r) => [r.est, r.lo, r.hi, r.hyp].filter((v): v is number => v != null).map(Math.abs))) * 1.25;
  const nice = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1].find((n) => n >= span) ?? 1;
  const domain: [number, number] = fixed ?? [-nice, nice];
  const x = (v: number) => ((v - domain[0]) / (domain[1] - domain[0])) * W;
  const clamp = (v: number) => Math.max(domain[0], Math.min(domain[1], v));
  return (
    <div>
      {rows.map((r, i) => (
        <div key={i} className="grid grid-cols-[minmax(160px,220px)_1fr_120px] items-center gap-3 border-b border-line py-2 last:border-0">
          <div className="t-small leading-tight">{r.label}</div>
          <div className="relative h-5">
            <div className="absolute top-1/2 h-px w-full bg-line" />
            <div className="absolute top-0 h-full w-px bg-ink" style={{ left: `${x(0)}%` }} />
            {r.lo != null && r.hi != null && (
              <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-hot" style={{ left: `${x(clamp(r.lo))}%`, width: `${x(clamp(r.hi)) - x(clamp(r.lo))}%` }} />
            )}
            {r.hyp != null && <div className="absolute top-[2px] h-4 w-[2px] bg-ink" style={{ left: `calc(${x(clamp(r.hyp))}% - 1px)` }} title={`hypothesis ${fmt(r.hyp)}`} />}
            {r.est != null && <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-hot" style={{ left: `${x(clamp(r.est))}%`, outline: r.sig ? "none" : "2px solid white" }} />}
          </div>
          <div className="num text-right t-small">
            <span className="text-hot">{r.est == null ? "—" : `${r.est >= 0 ? "+" : "−"}${Math.abs(r.est * 100).toFixed(1)}`}</span>
            {r.lo != null && r.hi != null && <span className="text-ink-3"> [{(r.lo * 100).toFixed(0)}, {(r.hi * 100).toFixed(0)}]</span>}
          </div>
        </div>
      ))}
      <div className="mt-1 grid grid-cols-[minmax(160px,220px)_1fr_120px] gap-3 t-micro text-ink-3">
        <span />
        <div className="flex justify-between"><span>{(domain[0] * 100).toFixed(0)} pp</span><span>{label}</span><span>+{(domain[1] * 100).toFixed(0)} pp</span></div>
        <span className="text-right">pp, 95% CI</span>
      </div>
    </div>
  );
}

/** One row per cell: model mean ± sd (accent) and hypothesis ring (ink). */
export function CellDots({ rows }: { rows: { label: string; y: number; sd: number; ref: number; n: number }[] }) {
  return (
    <div>
      {rows.map((r, i) => (
        <div key={i} className="grid grid-cols-[minmax(160px,260px)_1fr_90px] items-center gap-3 border-b border-line py-1.5 last:border-0">
          <div className="t-small leading-tight">{r.label}</div>
          <div className="relative h-5">
            <div className="absolute top-1/2 h-px w-full bg-line" />
            {[0, 0.25, 0.5, 0.75, 1].map((t) => <div key={t} className="absolute top-[30%] h-[40%] w-px bg-line" style={{ left: `${t * 100}%` }} />)}
            <div className="absolute top-1/2 h-[2px] -translate-y-1/2 bg-hot/50" style={{ left: `${Math.max(0, r.y - r.sd) * 100}%`, width: `${(Math.min(1, r.y + r.sd) - Math.max(0, r.y - r.sd)) * 100}%` }} />
            <div className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full" style={{ left: `${r.ref * 100}%`, boxShadow: "inset 0 0 0 2px var(--pg-ink)" }} />
            <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-hot" style={{ left: `${r.y * 100}%` }} />
          </div>
          <div className="num text-right t-small"><span className="text-hot">{(r.y * 100).toFixed(0)}%</span><span className="text-ink-3"> / {(r.ref * 100).toFixed(0)}%</span></div>
        </div>
      ))}
      <div className="mt-1 grid grid-cols-[minmax(160px,260px)_1fr_90px] gap-3 t-micro text-ink-3"><span /><div className="flex justify-between"><span>0%</span><span>50%</span><span>100%</span></div><span className="text-right">model / hyp.</span></div>
    </div>
  );
}

export function StackBar({ parts }: { parts: { label: string; value: number; color: string }[] }) {
  const tot = parts.reduce((a, b) => a + b.value, 0) || 1;
  return (
    <div>
      <div className="flex h-7 w-full">
        {parts.map((p) => <div key={p.label} style={{ width: `${(p.value / tot) * 100}%`, background: p.color }} className="h-full border-r-2 border-paper last:border-0" />)}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
        {parts.map((p) => (
          <span key={p.label} className="flex items-center gap-1.5 t-small"><span className="h-2.5 w-2.5" style={{ background: p.color }} />{p.label} <span className="num font-bold">{((p.value / tot) * 100).toFixed(0)}%</span></span>
        ))}
      </div>
    </div>
  );
}

export function InteractionPlot({ data, xLabel, series }: { data: Record<string, number | string>[]; xLabel: string; series: { key: string; label: string; dashed?: boolean; color: string }[] }) {
  return (
    <div className="h-[240px] w-full pb-5">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: -12 }}>
          <CartesianGrid stroke="var(--pg-line)" vertical={false} />
          <XAxis dataKey="x" tickLine={false} axisLine={{ stroke: "var(--pg-ink)" }} interval={0} padding={{ left: 24, right: 24 }} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} tickLine={false} axisLine={false} width={44} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ border: "1.5px solid #111", borderRadius: 0, fontSize: 12 }} />
          {series.map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "4 4" : undefined} dot={{ r: 4, fill: s.dashed ? "white" : s.color, stroke: s.color, strokeWidth: 2 }} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 text-center t-micro text-ink-3">{xLabel}</div>
    </div>
  );
}

export function SensitivityScatter({ points, slope, intercept }: { points: { x: number; y: number; label: string }[]; slope?: number | null; intercept?: number | null }) {
  const line = slope != null && intercept != null ? [{ x: 0, y: intercept }, { x: 1, y: intercept + slope }] : [];
  return (
    <div className="h-[230px] w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 18, left: -12 }}>
          <CartesianGrid stroke="var(--pg-line)" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} tickLine={false} label={{ value: "hypothesized P", position: "insideBottom", offset: -8, fontSize: 11 }} />
          <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} tickLine={false} width={44} />
          <ZAxis range={[60, 60]} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--pg-ink-3)" strokeDasharray="3 3" />
          {line.length > 0 && <ReferenceLine segment={line as [{ x: number; y: number }, { x: number; y: number }]} stroke="var(--pg-accent)" strokeWidth={1.5} />}
          <Tooltip cursor={false} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ border: "1.5px solid #111", borderRadius: 0, fontSize: 12 }} />
          <Scatter data={points} fill="var(--pg-accent)" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
