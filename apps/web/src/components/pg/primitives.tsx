"use client";

import { cn } from "@/lib/utils";
import type { Dist, Provenance } from "@/lib/api";
import { pct } from "@/lib/api";

/* ---------------------------------------------------------------- page head */
export function PageHead({ title, lead, measures, right }: { n?: string; title: string; lead?: React.ReactNode; measures?: string[]; right?: React.ReactNode }) {
  return (
    <section className="gutter pt-8 pb-7">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-[980px]">
          <h1 className="t-title">{title}</h1>
          {lead && <div className="t-lead mt-3 max-w-[900px] text-ink-2">{lead}</div>}
          {measures && (
            <p className="mt-4 max-w-[900px] t-small text-ink-2">
              <span className="mr-2 font-bold text-ink">Measures</span>
              {measures.map((m, i) => <span key={m}>{i > 0 && <> <span className="text-hot">·</span> </>}<span className="whitespace-nowrap">{m}</span></span>)}
            </p>
          )}
        </div>
        {right}
      </div>
    </section>
  );
}

export function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("t-small text-ink-2", className)}>{children}</div>;
}

export function SectionTitle({ children, aside, className }: { children: React.ReactNode; aside?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1", className)}>
      <h3 className="t-h3">{children}</h3>
      {aside && <div className="t-small text-ink-2">{aside}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- buttons */
export function Btn({ children, onClick, disabled, variant = "solid", className, type = "button", title }: {
  children: React.ReactNode; onClick?: () => void; disabled?: boolean; variant?: "solid" | "outline" | "ghost" | "hot";
  className?: string; type?: "button" | "submit"; title?: string;
}) {
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "inline-flex h-9 items-center justify-center gap-2 px-4 text-[14px] font-bold tracking-[-0.01em] transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        variant === "solid" && "bg-ink text-paper hover:bg-[#2a2a2a]",
        variant === "hot" && "bg-hot text-paper hover:bg-[#e63500]",
        variant === "outline" && "border-[1.5px] border-ink bg-paper text-ink hover:bg-wash",
        variant === "ghost" && "text-ink-2 hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "md", className }: {
  value: T; options: { id: T; label: string; disabled?: boolean; hint?: string }[]; onChange: (v: T) => void; size?: "sm" | "md"; className?: string;
}) {
  return (
    <div className={cn("inline-flex flex-wrap border-[1.5px] border-ink", className)}>
      {options.map((o, i) => (
        <button
          key={o.id}
          title={o.hint}
          disabled={o.disabled}
          onClick={() => onChange(o.id)}
          className={cn(
            "font-medium tracking-[-0.01em] transition-colors disabled:opacity-30",
            size === "sm" ? "h-7 px-2.5 text-[12.5px]" : "h-8 px-3 text-[13px]",
            i > 0 && "border-l-[1.5px] border-ink",
            value === o.id ? "bg-ink text-paper" : "bg-paper text-ink hover:bg-wash",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- provenance */
export function ProvenanceBadge({ p, className }: { p?: Partial<Provenance> | null; className?: string }) {
  if (!p) return null;
  const kind = p.backend === "fixture" ? "fixture" : p.backend === "replay" || p.backend === "recorded" ? "replay" : p.backend === "policy" ? "policy" : p.is_real_inference ? "live" : "fixture";
  return (
    <span className={cn("inline-flex items-center gap-2 t-micro", className)} title={p.prompt_hash ? `prompt ${p.prompt_hash}` : undefined}>
      {kind === "live" && <span className="h-2.5 w-2.5 rounded-full bg-hot" />}
      {kind === "replay" && <span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-hot" />}
      {kind === "policy" && <span className="h-2.5 w-2.5 border-[1.5px] border-ink" />}
      {kind === "fixture" && <span className="fixture-stripes h-2.5 w-2.5" />}
      <span className={cn(kind === "fixture" ? "font-bold text-ink" : "text-ink-2")}>
        {kind === "fixture" ? "FIXTURE — " : ""}{p.label ?? `${p.backend} · ${p.model}`}
        {p.latency_ms ? <span className="num text-ink-3"> · {(p.latency_ms / 1000).toFixed(1)}s</span> : null}
      </span>
    </span>
  );
}

export function SyntheticTag({ children = "Synthetic · authored reference, not human data" }: { children?: React.ReactNode }) {
  return <span className="inline-flex items-center gap-1.5 border border-ink-3 px-1.5 py-[1px] t-micro text-ink-2">{children}</span>;
}

/* ---------------------------------------------------------------- dot distributions */
const N_DOTS = 20;

export function DotRow({ p, target: refP, dotSize = 9, gap = 3, n = N_DOTS }: { p?: number | null; target?: number | null; dotSize?: number; gap?: number; n?: number }) {
  const fill = p == null ? 0 : Math.round(p * n);
  const ring = refP == null ? -1 : Math.round(refP * n);
  return (
    <div className="flex" style={{ gap }}>
      {Array.from({ length: n }).map((_, i) => {
        const filled = i < fill;
        const inRef = i < ring;
        return (
          <span
            key={i}
            className="pg-dot block rounded-full"
            style={{
              width: dotSize,
              height: dotSize,
              background: filled ? "var(--pg-accent)" : "transparent",
              boxShadow: inRef ? `inset 0 0 0 1.5px var(--pg-ink)` : filled ? "none" : `inset 0 0 0 1px var(--pg-line)`,
            }}
          />
        );
      })}
    </div>
  );
}

export function DistributionRows({ labels, model, reference, highlight, compact = false }: {
  labels: { id: string; label: string; literal?: boolean }[]; model?: Dist | null; reference?: Dist | null; highlight?: string | null; compact?: boolean;
}) {
  return (
    <div className="space-y-2.5">
      {labels.map((l) => (
        <div key={l.id} className={cn("grid items-center gap-x-3", compact ? "grid-cols-[1fr_auto]" : "grid-cols-[1fr_auto] gap-y-1 sm:grid-cols-[minmax(108px,1fr)_auto_auto]")}>
          <div className={cn("t-small leading-tight", highlight === l.id ? "text-ink font-bold" : "text-ink")}>
            {l.label}
            {l.literal && <span className="ml-1.5 t-micro text-ink-3">literal</span>}
          </div>
          <div className={cn(!compact && "order-last col-span-2 sm:order-none sm:col-span-1")}><DotRow p={model?.[l.id]} target={reference?.[l.id]} dotSize={compact ? 6 : 7.5} gap={compact ? 2 : 2.5} /></div>
          {!compact && (
            <div className="num w-[74px] whitespace-nowrap text-right t-small">
              <span className="text-hot">{model ? pct(model[l.id]) : "—"}</span>
              {reference && <span className="text-ink-3"> / {pct(reference[l.id])}</span>}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function DotLegend({ modelLabel = "Model", refLabel = "Reference (authored)" }: { modelLabel?: string; refLabel?: string }) {
  return (
    <div className="flex flex-wrap items-center gap-4 t-micro text-ink-2">
      <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-hot" />{modelLabel}</span>
      <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ boxShadow: "inset 0 0 0 1.5px var(--pg-ink)" }} />{refLabel}</span>
      <span className="text-ink-3">each dot = 5%</span>
    </div>
  );
}

/* ---------------------------------------------------------------- stat */
export function Stat({ label, value, sub, hot }: { label: string; value: React.ReactNode; sub?: React.ReactNode; hot?: boolean }) {
  return (
    <div>
      <div className="t-small text-ink-2">{label}</div>
      <div className={cn("num mt-1 text-[30px] font-medium leading-none tracking-[-0.035em]", hot && "text-hot")}>{value}</div>
      {sub && <div className="t-micro mt-1.5 text-ink-3">{sub}</div>}
    </div>
  );
}

export function FailureTag({ id, info }: { id: string; info?: string }) {
  const pass = id === "pass";
  return (
    <span title={info} className={cn("inline-flex items-center gap-1.5 px-2 py-[2px] t-small font-bold", pass ? "bg-wash text-ink" : "bg-hot-wash text-[#b32900]")}>
      <span className={cn("h-1.5 w-1.5 rounded-full", pass ? "bg-ink" : "bg-hot")} />
      {id.replaceAll("_", " ")}
    </span>
  );
}

export function Loading({ label = "Running" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 t-small text-ink-2">
      <span className="flex gap-1">{[0, 1, 2].map((i) => <span key={i} className="pg-pulse h-1.5 w-1.5 rounded-full bg-hot" style={{ animationDelay: `${i * 0.15}s` }} />)}</span>
      {label}
    </span>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="border-l-[3px] border-hot bg-hot-wash px-3 py-2 t-small text-[#8a2300]">{error}</div>;
}

export function Kv({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[130px_1fr] gap-3 border-b border-line py-1.5 t-small last:border-0">
      <span className="text-ink-2">{k}</span>
      <span className={cn("text-ink", mono && "mono break-all")}>{v}</span>
    </div>
  );
}
