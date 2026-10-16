# Pragmatic Gym — Research design

> **Thesis.** Social and pragmatic competence won't come from simply showing models more conversations. It can be taught with **targeted, consequence-bearing experiences**. *Which* experiences are causally useful is itself an empirical, data-centric research question.

## 1. Research questions

1. **Capability gap.** Which social-pragmatic capabilities do current LMs lack? Is the failure in *inference* (what is meant), *calibration* (how sure to be), or *action* (what to do next)?
2. **Experience selection.** Which training experiences are causally useful for teaching these capabilities, at a fixed token budget?
3. **Generalization.** Do improvements transfer to unseen relationships, unseen phrasings and an unseen social domain, and do they come without regressions or mere stylistic imitation?

## 2. Constructs, and why they're scored separately

| Construct | Operationalization | Why it's separate |
|---|---|---|
| Inference | Distribution over a discrete set of readings (e.g. *interested · undecided · polite no · keeping options open*). Cross-entropy / KL to the reference, and top-1 agreement **only on decisive items** (reference top ≥ .45 with a ≥ .10 margin). | A model can read the situation correctly and still act badly, and vice versa. |
| Calibration | Confidence gap (model confidence − reference mass on its top choice); entropy gap; overconfidence rate on *genuinely ambiguous* items (normalized reference entropy ≥ .80 and model top > .70). | Ambiguous messages shouldn't be forced into one interpretation. |
| Action | Expected-utility **regret** of the chosen move under the reference distribution; harmful-move rate (EU < −.15). The utilities U[move][intent] are an authored hypothesis (`social/domains.py`). | Good moves are good *under uncertainty*. A low-pressure clarification can be optimal even when the reading is unsure. |
| Interaction outcome | Return and outcome in the RL environments (plan confirmed, graceful close, coerced yes, withdrawal, …). | Outcomes depend on the interlocutor, not only on the agent's single move. |

The four are related and empirically dissociable. On the Soft No, the base model chooses a reasonable move (low-pressure clarification) while assigning only 7% to the polite-no reading.

## 3. Scenario and environment design

### 3.1 Hand-written corpus (`data/scenarios/handwritten.yaml`)

- 24 scenario families: 3 per capability across 8 capabilities.
- 29 counterfactual variants, each changing one observable fact: the history of follow-through, a scalar term ("some" → "all"), a joking frame, a new shared fact, and so on.
- Each scenario specifies:
  - the observable context and dialogue;
  - the relationship;
  - an authored reference distribution with a written **rationale**;
  - a concrete hidden state (one plausible latent truth, for the researcher view only);
  - why that hidden state matters for action.
- Every counterfactual must move the reference (TVD ≥ .08, enforced by a test). Some moves are deliberately subtle.
- All personal-life domains: friendship, dating, family, group chats. No workplace hierarchy.
- **Not human consensus.** These are synthetic, authored judgments, used as a held-out test set written by a different process from the training data.

### 3.2 Latent Social State Model (LSSM, `social/lssm.py`)

A transparent log-linear hypothesis:

    logit(intent | context) = base[utterance family] + Σ effect[factor = level] + Σ authored interactions

It has three roles:
1. the hypothesis that Counterfactual Lab results are compared against;
2. the prior from which an environment samples the hidden intent;
3. the source of **training targets as posteriors given observable context**. Training never uses the sampled hidden intent, because one-hot targets would teach overconfidence.

The LSSM is versioned (`lssm-v0.3-authored`) and every cell exposes an audit trail of logit contributions.

### 3.3 Factors (`social/factors.py`)

The 12 manipulable factors are:
- relationship and familiarity;
- shared history and emotional stakes;
- communication style and conversational history;
- audience and explicitly stated norms;
- information asymmetry and commitment expectations;
- directness (a manipulation check that changes the words);
- social pressure.

There is also a **placebo control**: an irrelevant detail of similar length, whose hypothesized effect is zero.

- Levels render as explicit context sentences with paraphrases.
- *Cultural context is only ever an explicitly stated norm* ("In your friend group, 'I'll let you know' is a gentle no"). It is never inferred from nationality or identity.
- Incoherent combinations (a close friend who is a stranger; a first interaction with a history of refusals) are **excluded and reported**, never rendered.

### 3.4 Environments (`environments/`)

- **API.** `reset(seed, options) → (obs, info)` and `step(a) → (obs, r, terminated, truncated, info)`.
- **Environments.** The Soft No and Are We on the Same Page? are the flagships; Group Chat, Family Subtext and It's Fine are also fully runnable.
- **Hidden state:** intent sampled from the LSSM prior, patience, friction.
- **Interlocutor:** an authored stochastic policy P(response | intent, move). Responses can reveal the intent, end the episode, or raise friction. Repeated clarification costs more each time.
- **Observable posterior.** Because the interlocutor policy is known, the environment computes the **exact Bayes posterior** over intents from observable events only. Clarification and over-assumption rewards condition on *this*, so an agent is never penalized for not knowing what it couldn't know.
- **Reward components** (weights configurable, every step logged):

  | Component | What it rewards or penalizes |
  |---|---|
  | pragmatic understanding | Log score of the submitted belief on the true intent, relative to uniform. A proper scoring rule. |
  | boundary respect | Penalizes pressure, more heavily when the intent is a refusal. |
  | clarification | Rewards the first clarification under real uncertainty; penalizes repeats. |
  | excessive assumption | Penalizes acting certain while the posterior is uncertain. |
  | social friction | Penalizes the friction a move creates. |
  | repair | Rewards moves that reduce built-up friction. |
  | task completion | Terminal outcome value, given the true intent. |

- **Message text is stored but never scored.**

## 4. The pre-registered experiment (`training/experiment.py`)

**Question.** Holding the base checkpoint, LoRA configuration, optimizer, seeds and *token budget* fixed, does the *kind* of experience selected change what the model learns?

| Element | Specification |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct`, chosen because full experiments with 3 seeds run on a laptop in about 2 h |
| Candidate pool | 1,703 items, **training families only**:<br>• 7 utterance families in 3 domains, generated as contrast clusters (an anchor plus one-factor neighbors)<br>• 240 multi-turn Soft No decision points, each with the environment's exact posterior as target |
| Arms | **random** (control) · **diversity** (k-center over TF-IDF) · **failure-targeted** (highest base-model loss, ≤ 2 per cluster) · **counterfactual-contrast** (whole clusters, ranked by how much a single factor moves the reference) · **isolated** (the random arm's experiences with relationship and history stripped). Plus **base**. |
| Budget | 240k estimated tokens per arm; train tokens measured afterwards are reported per run |
| Targets | Interpretation: 4 letters sampled from the reference posterior per item, each with a different option order (Monte-Carlo soft labels). Move: the best move under the reference. |
| Training | LoRA r = 16, α = 32, all attention + MLP projections, lr 2e-4 cosine, 1 epoch, batch 8, loss on the answer letter only (TRL `SFTTrainer`) |
| Evaluation | Next-token letter distribution, averaged over **all cyclic option orders**. Option order is always permuted in training, so position can't be learned and style can't score. |
| Eval sets (family-level split) | **ID:** held-out *cells* of training families (no shared factor assignment, unseen variants), 224 items<br>**OOD-family:** unseen utterances in seen domains ("We'll see!", "We don't need a label", "all good 👍"), 168<br>**OOD-domain:** the entire family-subtext domain, 168<br>**Hand-written:** 53<br>Plus a 40-item general-knowledge regression tripwire |

**Hypotheses** (primary outcome: mean cross-entropy to the reference on the pooled generalization sets — OOD-family ∪ OOD-domain ∪ hand-written, n = 389):

- **H1 (contrast).** Counterfactual-contrast < random. H₀: equal.
- **H2 (context).** Random (context-rich) < isolated (same experiences, context stripped). H₀: equal.
- **H3 (failure-targeting).** Failure-targeted < random. H₀: equal.

**Analysis.**
- Predictions are averaged over seeds per item.
- Paired bootstrap over items (4,000 resamples) gives the CI and a two-sided p.
- Holm–Bonferroni across H1–H3.
- Seed-level values are reported next to every mean.

**Secondary outcomes:**
- KL (excess loss);
- top-1 on decisive items;
- **context sensitivity**, i.e. the correlation r between how the model's beliefs move and how the reference moves under one-factor changes, and the **sensitivity ratio**, mean |Δmodel| / mean |Δreference|;
- paraphrase instability;
- overconfidence on ambiguous items;
- action regret and harmful-move rate;
- **unwarranted deference** (choosing an over-apologetic move that isn't acceptable);
- regression accuracy.

**Reference baselines (non-learned).**
- *Uniform.*
- *Utterance-only prior*: the hypothesis for the bare words, ignoring all context.

An arm that only reaches the utterance prior has learned the words, not the social situation.

## 5. Results

*This section is generated from `experiments/results/social-sft-v1.json` (see "Results" below). The Results view in the app renders the same file.*

<!-- RESULTS -->
**Experiment `social-sft-v1`** — REAL MEASUREMENT — local LoRA fine-tuning and evaluation of Qwen/Qwen2.5-0.5B-Instruct. Seeds completed: 0, 1, 2. Budget 240k tokens per arm. Eval items: ID (held-out cells) 224, Unseen families 168, Unseen domain 168, Hand-written 53.

**Pre-registered tests** (Δ cross-entropy, treatment − control, pooled generalization sets; negative = treatment better):

| Hypothesis | Δ (nats) | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| H1 Contrast vs Random | -0.016 | [-0.022, -0.010] | 0.000 | 0.000 |
| H2 Random vs Isolated | -0.005 | [-0.009, -0.001] | 0.012 | 0.024 |
| H3 Failure-targeted vs Random | +0.026 | [+0.003, +0.051] | 0.030 | 0.030 |

vs base: Contrast -0.239 [-0.27, -0.21]; Diversity -0.219 [-0.25, -0.19]; Failure-targeted -0.197 [-0.24, -0.15]; Isolated -0.218 [-0.25, -0.19]; Random -0.223 [-0.26, -0.19]

**Excess loss KL(reference‖model), nats — lower is better** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.538 | 0.533 | 0.282 | 0.626 |
| Contrast | 0.277 | 0.199 | 0.166 | 0.293 |
| Diversity | 0.262 | 0.237 | 0.183 | 0.268 |
| Failure-targeted | 0.463 | 0.230 | 0.177 | 0.471 |
| Isolated | 0.271 | 0.202 | 0.219 | 0.270 |
| Random | 0.279 | 0.213 | 0.195 | 0.275 |
| *Uniform (no model)* | 0.324 | 0.256 | 0.233 | 0.270 |
| *Utterance-only prior* | 0.328 | 0.214 | 0.172 | 0.270 |

**Top-1 agreement on decisive items — higher is better** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.19 | 0.12 | 0.22 | 0.22 |
| Contrast | 0.51 | 0.68 | 0.64 | 0.31 |
| Diversity | 0.54 | 0.49 | 0.51 | 0.33 |
| Failure-targeted | 0.46 | 0.68 | 0.66 | 0.33 |
| Isolated | 0.47 | 0.60 | 0.38 | 0.26 |
| Random | 0.46 | 0.62 | 0.56 | 0.32 |
| *Uniform (no model)* | 0.18 | 0.07 | 0.12 | 0.39 |
| *Utterance-only prior* | 0.28 | 0.49 | 0.51 | 0.39 |

**Context sensitivity r (Δmodel vs Δreference under one-factor changes) — higher is better** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.26 | 0.26 | 0.09 | — |
| Contrast | 0.59 | 0.40 | 0.08 | — |
| Diversity | 0.63 | 0.44 | 0.08 | — |
| Failure-targeted | 0.13 | 0.10 | 0.02 | — |
| Isolated | 0.21 | 0.05 | 0.08 | — |
| Random | 0.43 | 0.27 | 0.13 | — |

**Sensitivity ratio |Δmodel| / |Δreference| — 1 = moves as much as the reference** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.52 | 0.46 | 0.66 | — |
| Contrast | 0.23 | 0.19 | 0.23 | — |
| Diversity | 0.25 | 0.21 | 0.25 | — |
| Failure-targeted | 0.11 | 0.11 | 0.27 | — |
| Isolated | 0.06 | 0.07 | 0.10 | — |
| Random | 0.13 | 0.11 | 0.16 | — |

**Action regret (expected utility) — lower is better** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.053 | 0.047 | 0.063 | 0.060 |
| Contrast | 0.053 | 0.048 | 0.000 | 0.044 |
| Diversity | 0.073 | 0.052 | 0.037 | 0.095 |
| Failure-targeted | 0.069 | 0.038 | 0.000 | 0.138 |
| Isolated | 0.075 | 0.046 | 0.026 | 0.098 |
| Random | 0.200 | 0.185 | 0.007 | 0.100 |

**Overconfident on ambiguous items — lower is better** (mean over seeds; seed values in the app)

| Arm | ID (held-out cells) | Unseen families | Unseen domain | Hand-written |
|---|---|---|---|---|
| Base | 0.00 | 0.00 | 0.00 | 0.19 |
| Contrast | 0.00 | 0.00 | 0.00 | 0.00 |
| Diversity | 0.00 | 0.00 | 0.00 | 0.00 |
| Failure-targeted | 0.25 | 0.17 | 0.00 | 0.13 |
| Isolated | 0.00 | 0.00 | 0.00 | 0.00 |
| Random | 0.00 | 0.00 | 0.00 | 0.00 |

**Regression tripwire** (40 general MCQ): Base 0.800, Contrast 0.792, Diversity 0.767, Failure-targeted 0.808, Isolated 0.792, Random 0.717

<!-- /RESULTS -->

### 5.1 Interpretation (3 seeds, real runs)

1. **Every SFT arm makes the model less confidently wrong.** Each arm cuts KL against the base model by about 0.20–0.24 nats on the generalization sets. Much of that is calibration: the base model is *worse than uniform* on three of the four sets.
2. **H1 supported: contrast beats random.** Counterfactual-contrast clusters beat random selection at the same token budget: Δ = −0.016 nats [−0.022, −0.010], Holm p < .001. The effect is small, but it shows up where the hypothesis says it should:
   - the contrast arm's beliefs **move with context**, with context sensitivity r = 0.59 in-distribution and 0.40 on unseen families, against 0.43 and 0.27 for random;
   - its sensitivity ratio is 0.23 against 0.13.
3. **H2 supported, narrowly: context-rich beats isolated.** Context-rich examples beat the same experiences with context stripped (Δ = −0.005 [−0.009, −0.001], Holm p = .024). The mechanism differs sharply:
   - the isolated arm learns near-flat beliefs (sensitivity ratio 0.06; r = 0.05 on unseen families);
   - the context-rich arms learn beliefs that respond to the situation.
4. **H3 reversed: failure-targeting is worse.** Selecting the base model's highest-loss items does *worse* than random (Δ = +0.026 [+0.003, +0.051], Holm p = .030) and is the only arm that **introduces overconfidence** (25% overconfident on ambiguous in-distribution items, against 0% for every other arm).
   - This matches the reviewer-pass diagnosis (§6 T4): high base loss here marks *extreme labels* (manipulation-check items), not *useful experiences*.
   - Lesson: "train on what the model gets wrong" is not a free lunch when the loss is measured against a distributional target.
5. **Exploratory (not pre-registered): diversity selection** matches contrast on context sensitivity (r = 0.63 / 0.44) and has the best hand-written KL. Unlike contrast, it wasn't predicted, so treat it as a hypothesis for the next study.
6. **Limits of transfer:**
   - **No arm transfers context-reading to the unseen family-subtext domain** (r ≈ 0.08 everywhere).
   - **On the hand-written corpus no arm meaningfully beats a uniform predictor** (best 0.268 against 0.270).
   - What is learned is domain-bound and partly tied to the generator's phrasing. That is the most important negative result, and the motivation for human data and broader experience distributions (FUTURE_RESEARCH §1–2).
7. **Regression tripwire:** 0.72–0.81 against 0.80 for base. The random arm's dip (−3 of 40 items, averaged) is within the noise of a 40-item check; a real capability battery is needed before any claim about regressions.


## 6. Skeptical-reviewer pass: threats to validity, and what was changed

Written as a NeurIPS/ICLR reviewer would read it, after the system worked end to end.

**T1. Circularity: the model is trained toward, and evaluated against, the same authored hypothesis.**
- **The problem.** Template targets and template references both come from the LSSM, so in-distribution gains measure "learned the LSSM", not "learned human pragmatics".
- **Changes made:**
  - The hand-written corpus, whose references come from per-scenario authored judgment and not from the LSSM, is a separate eval set that is never trained on.
  - The unseen *domain* (family subtext) shares no utterances with training.
  - All UI and paper language says "agreement with the hypothesis" for template sets.
  - A human-annotation pipeline is defined (`scripts/export_annotation_batch.py`, `data/eval/human_annotations.schema.json`): 100-point allocations over readings, observable text only. The analysis code already reports human–model divergence when the file exists.
- **Remaining.** Both references share one author. Human distributions from a consented, explicitly described population are the necessary next step (FUTURE_RESEARCH §2).

**T2. Simulator and reward misspecification.**
- **The finding.** The audit found that with the belief-scoring term removed, a *greedy* Bayes policy doesn't beat the best *constant* move in 3 of 5 environments ("always acknowledge and repair" in It's Fine, "always reassure and schedule" in Family Subtext, "always clarify" in the Soft No). Read naively, the action reward is dominated by a universally safe move.
- **Changes made:**
  - Added a POMDP planner (Monte-Carlo lookahead on the known interlocutor model, sampling intents from the observable posterior only).
  - With action reward alone, it beats the best constant move in **all five** environments: Soft No 1.08 vs 0.89, Same Page 0.95 vs 0.60, Group Chat 0.87 vs 0.80, Family 1.24 vs 0.97, It's Fine 1.16 vs 0.93 (n = 150 episodes).
  - This is now a test (`test_inference_pays_even_without_belief_reward`).
  - The UI audit includes deferential and over-apologizing policies as hacking probes, and they score poorly.
  - The so-called "oracle" is also greedy, so it was relabeled "greedy oracle". It is *not* an upper bound.
- **Remaining.**
  - Margins are modest, so inference pays mostly through multi-turn dynamics.
  - The response tables are authored and untested against human behavior.
  - Simulator validity is the weakest link: the environment is a controlled testbed, not a model of people.

**T3. Confounding in contextual interventions.**
- **The problem.** Adding a factor sentence adds length, salience and position as well as information.
- **Changes made:**
  - A **placebo factor** (an irrelevant detail of similar length; hypothesized effect zero) so "more text" can be separated from "informative context".
  - Surface variants randomize names, paraphrases, sentence order and the utterance's surface form, and serve as **clusters** for inference.
  - Paired within-variant contrasts.
  - Incoherent cells are excluded rather than rendered. OLS handles the resulting imbalance.
- **Measured result.** On the base model, the placebo effect is −0.4 pp (CI −0.8 to +0.1). History, however, has no effect either, which is the finding (§5).
- **Remaining.** Factor *wording* is still author-chosen. A stronger design would sample factor paraphrases from several writers.

**T4. Data leakage and selection artifacts.**
- **Leakage controls:**
  - split at the family level (no family overlap with OOD or hand-written sets; no shared *cell* with ID);
  - targets from posteriors, never hidden intents;
  - a leakage scanner for latent identifiers in model-visible text;
  - near-duplicate checks across splits.
- **The selection artifact.** Failure-targeted selection over-selects the explicit-utterance *manipulation-check* items and one domain: 61% polite-no targets, flagged by the diagnostics. High base-model loss partly reflects how extreme the label is, not how useful the item is.
- **Changes made.** Imbalance warnings are surfaced in the UI and saved with every run. Arm sizes in items are reported, since matched tokens don't mean matched examples: the isolated arm has 2× more, shorter items.

**T5. Teaching style or label priors instead of inference.**
- **What was ruled out.** Letter-probability scoring removes text style. The deference-bait metric catches "always pick the polite-sounding option".
- **What remained.** A subtler failure surfaced in the pilot: **SFT can lower cross-entropy simply by becoming uniform**. The pilot random arm reached top-1 0.49, but its sensitivity ratio was 0.07, meaning its beliefs barely moved with context.
- **Changes made:**
  - The headline is no longer cross-entropy alone.
  - KL is reported next to **non-learned baselines** (uniform, utterance-only prior).
  - **Context sensitivity r and the sensitivity ratio** are primary secondary outcomes.
  - The isolated arm (H2) directly tests whether context is used.
  - The pilot also showed eval-time letter-position bias inflating action regret. Action distributions are now averaged over *all* cyclic orders.

**T6 (power).** Three seeds and about 389 generalization items. CIs are bootstrapped over items with seed-averaged predictions, so they understate seed variance; seed values are therefore shown separately. Effects smaller than about 0.02 nats are not resolvable.

## 7. Limitations

- **Synthetic data.** All scenarios and references are synthetic. Nothing here measures human preferences or human psychology.
- **Cultural variation.** Norms are explicit, situational sentences. Real communities vary in ways this can't capture, and generalizing "norms" from these templates would be wrong.
- **Annotation disagreement.** It's modeled as reference *distributions*, but the distributions are authored, not measured, and inter-annotator structure is unknown.
- **Simulator validity.** The interlocutor tables are hypotheses. Agents that exploit them may not transfer.
- **Distribution shift.** Template-to-hand-written is a real shift but narrow: one author, English, text messaging.
- **Scale.** A 0.5B model is a compute-bounded testbed. Effects at larger scale may differ in size and even in sign.
- **Live frontier inference.** The Claude backend is implemented but couldn't run in the build environment (workspace-scoped key). No frontier-model numbers are reported.
