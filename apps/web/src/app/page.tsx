import Link from "next/link";
import { TABS } from "@/lib/tabs";

/** Original dot drawing: two overlapping speech bubbles — what is said / what is meant. */
function BubbleDots() {
  const cols = 9, rows = 15, pitch = 31, r = 11;
  const dots: { x: number; y: number; kind: "said" | "meant" | "both" }[] = [];
  const inBubble = (c: number, rr: number, x0: number, y0: number, w: number, h: number, tailLeft: boolean) => {
    const inBox = c >= x0 && c < x0 + w && rr >= y0 && rr < y0 + h;
    const corner = (c === x0 || c === x0 + w - 1) && (rr === y0 || rr === y0 + h - 1);
    const tail = tailLeft ? (rr === y0 + h && c === x0 + 1) || (rr === y0 + h + 1 && c === x0) : (rr === y0 + h && c === x0 + w - 2) || (rr === y0 + h + 1 && c === x0 + w - 1);
    return (inBox && !corner) || tail;
  };
  for (let rr = 0; rr < rows; rr++)
    for (let c = 0; c < cols; c++) {
      const said = inBubble(c, rr, 0, 1, 6, 5, true);
      const meant = inBubble(c, rr, 3, 6, 6, 6, false);
      if (said || meant) dots.push({ x: c, y: rr, kind: said && meant ? "both" : said ? "said" : "meant" });
    }
  // a few "missing" dots inside the meant bubble: inference is incomplete
  const gaps = new Set(["5,8", "7,9", "4,10", "8,7", "6,11"]);
  return (
    <svg width={cols * pitch} height={rows * pitch} viewBox={`0 0 ${cols * pitch} ${rows * pitch}`} aria-hidden className="max-w-full">
      {dots.map((d, i) => {
        const cx = d.x * pitch + pitch / 2, cy = d.y * pitch + pitch / 2;
        if (d.kind === "said") return <circle key={i} cx={cx} cy={cy} r={r} fill="var(--pg-ink)" />;
        if (gaps.has(`${d.x},${d.y}`)) return <circle key={i} cx={cx} cy={cy} r={r - 0.75} fill="none" stroke="var(--pg-accent)" strokeWidth={1.5} />;
        return <circle key={i} cx={cx} cy={cy} r={r} fill="var(--pg-accent)" />;
      })}
    </svg>
  );
}

const QUESTIONS = [
  ["What can't models do?", "Which social and pragmatic capabilities do current LLMs lack — and where exactly do they fail: inference, calibration, or action?"],
  ["Which experiences teach it?", "Which data and interaction experiences are causally useful for teaching those capabilities, at a fixed token budget?"],
  ["Does it generalize?", "Which post-training interventions improve the capability on unseen relationships, unseen phrasings and unseen domains — without regressions?"],
];

export default function Home() {
  return (
    <div className="pb-16">
      <section className="gutter grid grid-cols-1 gap-10 pt-12 lg:grid-cols-[1fr_auto]">
        <div className="max-w-[1040px]">
          <h1 className="t-display">Language is more than what is said.</h1>
          <p className="mt-6 text-[clamp(22px,2.6vw,38px)] font-medium leading-[1.08] tracking-[-0.03em]">
            The same words can mean different things depending on context, intent, and what is at stake. Pragmatic Gym explores how
            models acquire social intelligence, and which training experiences actually produce it.
          </p>
        </div>
        <div className="hidden justify-self-end pt-2 lg:block"><BubbleDots /></div>
      </section>

      <section className="gutter mt-16">
        <div className="rule-heavy" />
        <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-6">
          {TABS.map((t, i) => (
            <Link key={t.href} href={t.href} className="group border-line py-5 pr-6 xl:border-l xl:pl-5 xl:first:border-l-0 xl:first:pl-0">
              <div className="num t-small text-hot">{t.n} · {t.loop}</div>
              <div className="mt-1.5 text-[19px] font-bold leading-tight tracking-[-0.025em] group-hover:underline group-hover:decoration-[1.5px] group-hover:underline-offset-4">{t.label}</div>
              <p className="mt-2 t-small text-ink-2">
                {[
                  "Find where a model misreads a message, and whether the reading, the confidence or the next move is what goes wrong.",
                  "Keep the message fixed, change one social fact at a time, and measure how far the model\u2019s reading moves.",
                  "Play out a conversation with someone whose intention is hidden, and see whether reading them well pays off.",
                  "Turn a diagnosed failure into training data. Every strategy gets the same token budget.",
                  "Fine-tune on each dataset and check what actually improved, on conversations the model has never seen.",
                  "Why data research belongs inside the scientific loop, and the team I\u2019d build to run it.",
                ][i]}
              </p>
            </Link>
          ))}
        </div>
        <div className="rule" />
      </section>

      <section className="gutter mt-16 grid grid-cols-1 gap-10 lg:grid-cols-[360px_1fr]">
        <div>
          <h2 className="text-[36px] font-bold leading-[1.1] tracking-[-0.035em]">Research questions</h2>
          <p className="mt-3 t-lead text-ink-2">The structure and selection of training experiences, studied as a mechanism for producing new model capabilities.</p>
        </div>
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          {QUESTIONS.map(([q, a], i) => (
            <div key={q}>
              <div className="num text-[44px] font-medium leading-none tracking-[-0.04em] text-hot">{i + 1}</div>
              <div className="mt-3 text-[20px] font-bold leading-tight tracking-[-0.025em]">{q}</div>
              <p className="mt-2 text-[15px] leading-relaxed text-ink-2">{a}</p>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
