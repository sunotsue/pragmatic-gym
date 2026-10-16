"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAppState } from "./app-state";
import { cn } from "@/lib/utils";
import { TABS } from "@/lib/tabs";



export function Wordmark() {
  return (
    <Link href="/" className="group flex items-end gap-3">
      <span aria-hidden className="grid grid-cols-4 gap-[3px] pb-[5px]">
        {[1, 0.75, 0.45, 0.2, 0.2, 0.55, 1, 0.35].map((o, i) => (
          <span key={i} className="block h-[7px] w-[7px] rounded-full" style={{ background: i < 4 ? "var(--pg-accent)" : "var(--pg-ink)", opacity: o }} />
        ))}
      </span>
      <span className="text-[27px] font-medium leading-none tracking-[-0.045em]">Pragmatic Gym</span>
    </Link>
  );
}

export function Header() {
  const path = usePathname();
  const { apiError } = useAppState();
  return (
    <header className="gutter pt-7">
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <Wordmark />
        <div className="flex flex-wrap items-center gap-5 t-small">
          {apiError && (
            <span className="flex items-center gap-2 text-hot"><span className="h-2 w-2 rounded-full bg-hot" /> Backend offline — start <code className="mono">make dev</code></span>
          )}
        </div>
      </div>
      <div className="rule-heavy mt-5" />
      <nav className="flex flex-wrap gap-x-8 gap-y-2 py-3">
        {TABS.map((t) => {
          const active = path === t.href || !!path?.startsWith(`${t.href}/`);
          return (
            <Link key={t.href} href={t.href} className={cn("group flex items-baseline gap-2 text-[15px] font-bold tracking-[-0.015em] transition-colors", active ? "text-ink" : "text-ink-3 hover:text-ink")}>
              <span className={cn("num text-[12px] font-medium", active ? "text-hot" : "text-ink-3")}>{t.n}</span>
              {t.label}
            </Link>
          );
        })}
      </nav>
      <div className="rule" />
    </header>
  );
}
