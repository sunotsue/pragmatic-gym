"use client";

import { cn } from "@/lib/utils";
import type { Message } from "@/lib/api";

const CHANNEL: Record<string, string> = { private: "Private messages", group_chat: "Group chat", in_person: "In person" };

export function Avatar({ name, size = 28, you }: { name: string; size?: number; you?: boolean }) {
  const initials = name === "The group" ? "G" : name.replace(/[^A-Za-z ]/g, "").split(" ").map((x) => x[0]).join("").slice(0, 2) || "·";
  return (
    <span
      className={cn("inline-flex shrink-0 items-center justify-center font-bold tracking-[-0.02em]", you ? "bg-ink text-paper" : "bg-wash text-ink")}
      style={{ width: size, height: size, fontSize: size * 0.4 }}
    >
      {initials}
    </span>
  );
}

export function ChatHeader({ speaker, relationship, channel, participants }: { speaker: string; relationship: string; channel: string; participants: string[] }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line pb-3">
      <div className="flex items-center gap-3">
        {channel === "group_chat" ? (
          <div className="flex -space-x-1.5">{participants.slice(0, 4).map((p) => <Avatar key={p} name={p} size={26} />)}</div>
        ) : (
          <Avatar name={speaker} size={34} />
        )}
        <div>
          <div className="text-[16px] font-bold leading-tight tracking-[-0.02em]">
            {channel === "group_chat" ? participants.join(", ") : speaker}
          </div>
          <div className="t-micro text-ink-2">{relationship} · {CHANNEL[channel] ?? channel}</div>
        </div>
      </div>
    </div>
  );
}

export function ContextNotes({ context, highlight, title = "What you know" }: { context: string[]; highlight?: Set<string>; title?: string }) {
  if (!context.length) return null;
  return (
    <div className="bg-wash px-4 py-3">
      <div className="t-micro mb-1.5 text-ink-2">{title}</div>
      <ul className="space-y-1">
        {context.map((c, i) => {
          const hot = highlight?.has(c);
          return (
            <li key={i} className={cn("relative pl-3 text-[14px] leading-snug", hot ? "text-ink" : "text-ink")}>
              <span className={cn("absolute left-0 top-[0.55em] h-[5px] w-[5px] rounded-full", hot ? "bg-hot" : "bg-ink-3")} />
              {hot ? <mark className="bg-hot-wash px-0.5 text-ink">{c}</mark> : c}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function Thread({ history, channel, emphasizeLast = true, pendingFrom, annotations }: {
  history: Message[]; channel: string; emphasizeLast?: boolean; pendingFrom?: string | null;
  annotations?: Record<number, React.ReactNode>;
}) {
  let lastTime: string | null | undefined = null;
  return (
    <div className="flex flex-col gap-1.5">
      {history.map((m, i) => {
        const showTime = m.time && m.time !== lastTime;
        lastTime = m.time;
        const prev = history[i - 1];
        const showName = !m.is_you && (channel === "group_chat" || m.sender !== prev?.sender || showTime);
        const isLast = i === history.length - 1;
        const isTarget = emphasizeLast && isLast && !m.is_you;
        const system = m.sender === "—" || m.text.startsWith("(");
        return (
          <div key={i}>
            {showTime && <div className="t-micro my-3 text-center text-ink-3">{m.time}</div>}
            {system ? (
              <div className="my-1 text-center t-small italic text-ink-3">{m.sender !== "—" && !m.is_you ? `${m.sender}: ` : ""}{m.text}</div>
            ) : (
              <div className={cn("flex items-end gap-2", m.is_you ? "justify-end" : "justify-start")}>
                {!m.is_you && <span className="w-7">{showName && <Avatar name={m.sender.replace(" (private)", "")} size={28} />}</span>}
                <div className={cn("max-w-[78%]", m.is_you && "flex flex-col items-end")}>
                  {showName && channel === "group_chat" && <div className="t-micro mb-0.5 text-ink-2">{m.sender}</div>}
                  <div
                    className={cn(
                      "px-3.5 py-2 text-[15px] leading-[1.38] tracking-[-0.01em]",
                      m.is_you ? "bg-ink text-paper" : "bg-wash text-ink",
                      isTarget && "outline outline-2 outline-offset-2 outline-hot",
                    )}
                  >
                    {m.text}
                  </div>
                  {annotations?.[i]}
                </div>
              </div>
            )}
          </div>
        );
      })}
      {pendingFrom && (
        <div className="flex items-end gap-2">
          <span className="w-7"><Avatar name={pendingFrom} size={28} /></span>
          <div className="flex gap-1 bg-wash px-3.5 py-3">{[0, 1, 2].map((i) => <span key={i} className="pg-pulse h-1.5 w-1.5 rounded-full bg-ink-3" style={{ animationDelay: `${i * 0.15}s` }} />)}</div>
        </div>
      )}
    </div>
  );
}
