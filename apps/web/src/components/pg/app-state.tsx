"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { api } from "@/lib/api";

export interface Backend { backend: string; available: boolean; model?: string; reason?: string | null; note?: string }
export interface Meta {
  capabilities: Record<string, { label: string; short: string }>;
  domains: Record<string, { id: string; title: string; interpretations: { id: string; label: string; description: string; literal: boolean }[]; actions: { id: string; label: string; example: string; style: string[] }[]; utility: Record<string, Record<string, number>> }>;
  factors: { id: string; label: string; description: string; levels: { id: string; label: string }[]; default: string; domains: string[]; group: string }[];
  families: { id: string; domain: string; title: string; split: string; utterance: string; explicit: string; factors: string[] }[];
  envs: { id: string; title: string; subtitle: string; flagship: boolean; domain: string; capabilities: string[] }[];
  failures: Record<string, string>;
  reward_components: Record<string, string>;
  versions: Record<string, string>;
}

interface State {
  backend: string;
  setBackend: (v: string) => void;
  backends: Backend[];
  meta: Meta | null;
  apiError: string | null;
}

const Ctx = createContext<State | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [backend, setBackendRaw] = useState("");
  const [backends, setBackends] = useState<Backend[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    api<{ default_backend: string; backends: Backend[] }>("/api/status")
      .then((s) => {
        setBackends(s.backends);
        let saved = "";
        try { saved = localStorage.getItem("pg.backend") ?? ""; } catch {}
        const ok = s.backends.find((b) => b.backend === saved && b.available);
        setBackendRaw(ok ? saved : s.default_backend);
      })
      .catch((e) => setApiError(String(e)));
    api<Meta>("/api/meta").then(setMeta).catch((e) => setApiError(String(e)));
  }, []);

  const setBackend = (v: string) => { setBackendRaw(v); try { localStorage.setItem("pg.backend", v); } catch {} };

  return <Ctx.Provider value={{ backend, setBackend, backends, meta, apiError }}>{children}</Ctx.Provider>;
}

export function useAppState() {
  const c = useContext(Ctx);
  if (!c) throw new Error("AppStateProvider missing");
  return c;
}
