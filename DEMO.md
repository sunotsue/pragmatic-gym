# Live demo — 6 minutes

**Setup (before the talk).**
- Run `make dev` and open http://localhost:3000 at 1440 wide.
- Header: *Inference = Local · Qwen2.5-0.5B (recorded)*. The app serves saved model outputs, so every click is instant and nothing runs live.
- Have the Counterfactual Lab open in a second tab.

Every number below is a real Qwen2.5-0.5B-Instruct output, recorded by `scripts/record_demo.py` and replayed by the app. If you swap in a frontier model via the header (`ANTHROPIC_WORKSPACE_ID` set), the *pattern* is the thing to show, not these exact values.

---

### 0:00–1:00 · A surprising pragmatic failure (Explorer)

1. Open **01 Capability Explorer → Indirect requests → The Soft No**.
   - *"Jordan, a close friend, skipped the last two invitations: tired once, replied a day late once. You invite them to your birthday. They say: 'That sounds fun. I'll let you know!'"*
2. Click **Run model**.
   - The accent dots show the model's reading; the rings show the authored reference.
   - The model's top reading is **"Interested, will likely come" (≈41%)**. It gives **≈10% to "Polite no"**, which the reference rates most likely (45%).
   - Failure tags: **misread**.
3. Point to *What it does*: the chosen move is *Clarify, low pressure*, a reasonable move.
   - **Say:** *"The reading is wrong but the action is fine. That's why I score inference, calibration and action separately. They're different capabilities and they fail differently."*
4. Toggle the counterfactual variant *"Jordan has always followed through"* and run it. Now the model moves: "interested" goes from 41% to **60%**, which is correct (failure tag: *pass*).
   - **Say:** *"So it can use history, but only good news. Two avoided invitations didn't register as a soft no; a record of reliability registers immediately. That's an optimism asymmetry in reading social history, and a testable hypothesis about what's missing from its training experiences."*

### 1:00–2:30 · Isolate the context effect (Counterfactual Lab)

1. Open **02 Counterfactual Lab**, preset **History × audience**.
   - Click levels in the right column; the conversation on the left re-renders instantly, with the manipulated sentences highlighted.
   - **Say:** *"The utterance is fixed. Only who-said-it-after-what changes. Names, paraphrases and punctuation are randomised surface variants, and they're my clusters for inference."*
2. Click **Run experiment (24 prompts)**. It takes about 15 s and the result is real.
   - **Sensitivity slope ≈ 0.06.** The hypothesis moves P(polite no) from 6% to 53% across cells; the model stays at 6–9%.
   - **Context vs wording.** In this run wording and names explain about as much variance as social history does. In the 6-variant run (*prior pattern × placebo*), **wording explained 53% and context 6%**.
   - Forest plot: *repeated disappointment* +2 pp for the model vs +25 pp hypothesised.
3. Optional: switch the preset to **Manipulation check**. Making the utterance explicit ("Thanks, but I'm going to pass on this one") moves P(polite no) by **+21 pp** against +71 pp hypothesised, while *established trust* moves it −0.5 pp against −12.5 pp. The model partly reads words; it hardly reads relationships.
   - **Say:** *"These are randomised interventions on model behaviour, not claims about humans. The ink ticks are an authored hypothesis, and there's a slot for human distributions."*

### 2:30–4:00 · Consequences (Social RL Playground)

1. Open **03 Social RL Playground → The Soft No**, seed 7, *Conversational history = Repeatedly*.
2. Click **Bayes policy**. The reference agent acts on the exact observable posterior (rings). The friend replies; the reward decomposes per turn.
3. Click **Reveal hidden intent**. The friend was, say, *Genuinely undecided*. The agent never sees this.
4. **Reset** and play **Press for an answer** yourself. Friction jumps, the boundary penalty fires, and the friend may withdraw or give a *coerced yes* (a negative outcome).
5. Scroll to **Reward audit → Run audit**.
   - **Say:** *"I treat my own reward as a hypothesis to be falsified. Deferential, always-clarify and always-repair policies are hacking probes. A greedy Bayes policy barely beat 'always clarify', so I added a planner that only sees the observable posterior. It beats every constant move in all five environments. Inference pays, but through multi-turn consequences. That finding is in RESEARCH §6."*

### 4:00–5:30 · Turn the failure into an intervention (Data Intervention Lab)

1. Open **04 Data Intervention Lab**.
   - The failure profile of the base model on 1,703 pool items (live): mostly **misread**, then underconfident and literal readings.
2. Click the strategies; the comparison table updates at the same **240k-token budget**.
   - **Counterfactual contrast** selects whole clusters: the same conversation with one social fact changed, ranked by how much that fact should move the reading.
   - **Failure-targeted** shows an honest artefact: it over-selects manipulation-check items (a 61% polite-no imbalance warning).
   - **Isolated** strips relationship and history from the same experiences. It's the ablation for *"does context-rich data teach more than isolated refusals?"*
3. Open an example: you see *why it was included*, the base model's dots against the target posterior, and the leakage, duplicate and overlap checks (0 leaks, no family overlap with held-out sets).
4. Click **Export JSONL** to get SFT prompt/completion rows plus preference pairs, with provenance.

### 5:30–7:00 · Did it work, and what's next (Results)

1. Open **05 Post-Training & Results** (`social-sft-v1`). The black bar says *REAL MEASUREMENT*. Every arm uses the same checkpoint, LoRA config and budget, and 3 seeds.
2. Read the **pre-registered comparisons** H1–H3 (Δ cross-entropy with 95% CI and Holm-corrected p). Then switch the metric to **Context sensitivity r** and **Sensitivity ratio**, and show the grey **non-learned baselines**.
   - **Say:** *"The first thing SFT learns is humility. Loss drops partly because the model stops being confidently wrong. The question that matters is whether beliefs start moving with context. That's the sensitivity row, and that's where the arms differ."* Then the result, 3 seeds:
   - **Contrast beats random:** −0.016 nats, Holm p < .001.
   - **Context-rich beats isolated:** −0.005, p = .024.
   - **Failure-targeting is worse and introduces overconfidence.**
   - The contrast arm's beliefs track context about 1.5× better on unseen utterances (r 0.40 vs 0.27).
   - **Nothing transfers to the unseen family domain, and no arm beats uniform on the hand-written corpus. That gap is the research agenda.**
3. Close on FUTURE_RESEARCH:
   - **Say:** *"The contribution isn't a dataset. It's a method for discovering a missing capability, finding which experiences teach it, and proving the change is real. Next: human reference distributions, frontier-scale sensitivity curves, and RL from consequences in calibrated simulators."*

---

**If something breaks.**
- **Backend offline:** the header shows it. Run `make dev`.
- **"Not recorded" message:** pick a preset in the Counterfactual Lab, or compare a single variable on its own.
- **No network:** everything except Hugging Face downloads works offline. The fixture backend is clearly labelled and should only be used to show UI flow, never to quote numbers.
