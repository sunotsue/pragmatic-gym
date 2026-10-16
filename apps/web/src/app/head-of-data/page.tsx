import { PageHead, SectionTitle } from "@/components/pg/primitives";

const BUILD: [string, string][] = [
  ["Build & Lead the Data Research Team", "Hire researchers and engineers; establish the pipelines, infrastructure, and experimental workflows."],
  ["Co-Define the Research Agenda", "Work with model researchers to formulate hypotheses, identify behavioral failure modes, and prioritize scientific questions."],
  ["Design the Data Interventions", "Determine what to collect, synthesize, or buy, and define quality in terms of the behaviors and capabilities we want to develop."],
  ["Train & Evaluate", "Run post-training experiments and develop evaluations that measure behavioral fidelity, generalization, and capability improvement."],
  ["Attribute Capability Gains to Data", "Isolate which data interventions work, why they work, and whether their effects generalize across models and settings."],
];

const CHAIN = ["Lab researcher", "Lab procurement", "Vendor sales", "Vendor researcher", "Data production"];
const INTENT = [10, 7, 5, 3, 1]; // dots of research intent left at each hand-off

const LOOP = ["Hypothesis", "Data intervention", "Training", "Evaluation", "Attribution"];

function IntentDots({ n }: { n: number }) {
  return (
    <div className="flex gap-[3px]" aria-label={`${n} of 10`}>
      {Array.from({ length: 10 }).map((_, i) => (
        <span key={i} className="block h-[7px] w-[7px] rounded-full" style={i < n ? { background: "var(--pg-accent)" } : { boxShadow: "inset 0 0 0 1px var(--pg-line)" }} />
      ))}
    </div>
  );
}

function Arrow({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden className={`flex items-center justify-center text-ink-3 ${className}`}>
      <span className="hidden text-[18px] md:inline">→</span>
      <span className="text-[18px] md:hidden">↓</span>
    </div>
  );
}

function BrokenModel() {
  return (
    <div>
      <div className="flex flex-col items-stretch md:flex-row md:items-center">
        {CHAIN.map((c, i) => (
          <div key={c} className="flex flex-col md:flex-1 md:flex-row md:items-center">
            {i > 0 && <Arrow className="py-1 md:w-8 md:shrink-0 md:py-0" />}
            <div className="flex-1 border-[1.5px] border-ink px-3 py-3">
              <div className="text-[14px] font-bold leading-tight tracking-[-0.01em]">{c}</div>
              <div className="mt-2"><IntentDots n={INTENT[i]} /></div>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-2 t-micro text-ink-3">
        <span className="block h-[7px] w-[7px] rounded-full bg-hot" /> research intent that survives each hand-off
      </div>
    </div>
  );
}

function IntegratedModel() {
  return (
    <div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {[["Model researchers", "Hypotheses and architectures"], ["Data researchers", "Data design and measurement"]].map(([t, s]) => (
          <div key={t} className="border-[1.5px] border-ink px-4 py-3">
            <div className="text-[15px] font-bold tracking-[-0.01em]">{t}</div>
            <div className="t-small text-ink-2">{s}</div>
          </div>
        ))}
      </div>
      <div aria-hidden className="grid grid-cols-2 text-center text-[18px] text-ink-3"><span>↓</span><span className="hidden sm:inline">↓</span></div>

      <div className="border-[1.5px] border-ink bg-wash px-4 pb-4 pt-3">
        <div className="t-small font-bold">Joint experimental loop</div>
        <div className="mt-3 flex flex-col items-stretch md:flex-row md:items-center">
          {LOOP.map((s, i) => (
            <div key={s} className="flex flex-col md:flex-1 md:flex-row md:items-center">
              {i > 0 && <Arrow className="py-0.5 md:w-6 md:shrink-0 md:py-0" />}
              <div className="flex-1 border-[1.5px] border-ink bg-paper px-2.5 py-2 text-center text-[13.5px] font-bold tracking-[-0.01em]">{s}</div>
            </div>
          ))}
        </div>
        <div className="mt-2 flex items-center gap-2 t-micro text-ink-2">
          <span aria-hidden className="text-[15px] text-hot">↺</span> attribution feeds the next hypothesis
        </div>
      </div>

      <div aria-hidden className="text-center text-[18px] text-ink-3">↓</div>
      <div className="bg-hot px-4 py-3 text-center text-[16px] font-bold tracking-[-0.015em] text-paper">Measured capability improvement</div>
    </div>
  );
}

export default function HeadOfData() {
  return (
    <div className="pb-16">
      <PageHead
        n="00"
        title="Head of Data"
      />

      <section className="gutter grid grid-cols-1 gap-10 lg:grid-cols-[360px_1fr]">
        <div>
          <h2 className="text-[36px] font-bold leading-[1.1] tracking-[-0.035em]">Thesis</h2>
        </div>
        <p className="max-w-[900px] text-[clamp(20px,2vw,28px)] font-medium leading-[1.2] tracking-[-0.025em]">
          Treating data as a downstream support function is a well-known structural failure mode. Data researchers should be embedded
          in the scientific loop, not operating as an internal vendor.
        </p>
      </section>

      <section className="gutter mt-14">
        <div className="rule-heavy" />
        <div className="grid grid-cols-1 gap-14 pt-6">
          <div className="min-w-0">
            <SectionTitle>The broken model</SectionTitle>
            <BrokenModel />
            <p className="mt-6 t-lead text-ink">Research intent gets diluted across organizational boundaries.</p>
            <p className="mt-2 max-w-[640px] t-small text-ink-2">
              The problem isn&rsquo;t simply communication overhead. It&rsquo;s that the person designing the data is often several degrees
              removed from the scientific hypothesis the data is supposed to test.
            </p>
          </div>
          <div className="min-w-0 max-w-[1000px]">
            <SectionTitle>The research-integrated model</SectionTitle>
            <IntegratedModel />
          </div>
        </div>
      </section>

      <section className="gutter mt-16 grid grid-cols-1 gap-10 lg:grid-cols-[360px_1fr]">
        <div>
          <h2 className="text-[36px] font-bold leading-[1.1] tracking-[-0.035em]">What I want to build</h2>
        </div>
        <ol className="max-w-[900px] border-t-[1.5px] border-ink">
          {BUILD.map(([title, body], i) => (
            <li key={title} className="grid grid-cols-[40px_1fr] gap-2 border-b border-line py-3.5">
              <span className="num text-[20px] font-medium leading-tight text-hot">{i + 1}</span>
              <div>
                <div className="text-[17px] font-bold leading-snug tracking-[-0.015em]">{title}</div>
                <div className="mt-0.5 text-[15px] leading-snug text-ink-2">{body}</div>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="gutter mt-16 grid grid-cols-1 gap-10 lg:grid-cols-[360px_1fr]">
        <div>
          <h2 className="text-[36px] font-bold leading-[1.1] tracking-[-0.035em]">How I&rsquo;d measure success</h2>
        </div>
        <div className="max-w-[900px]">
          <p className="t-lead text-ink-3 line-through decoration-[1.5px]">Not tasks produced, annotations completed, or datasets procured.</p>
          <p className="mt-4 text-[clamp(22px,2.4vw,34px)] font-bold leading-[1.15] tracking-[-0.03em]">
            Capability improvement attributable to data, per unit of research effort and compute.
          </p>
        </div>
      </section>
    </div>
  );
}
