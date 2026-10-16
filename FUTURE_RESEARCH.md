# From prototype to research program

Pragmatic Gym is a first vertical slice. The program it points to treats **training experiences as the object of study**: which experiences, structured how, produce which capabilities, and why. Below is how that could become a sequence of publishable work. It connects to work on human-like intelligence, machine psychology, pragmatic reasoning (e.g. the *On the Same Wavelength?* evaluation of pragmatic reasoning with RSA) and information-theoretic views of learning.

## 1. Measure the gap properly (months 0–2)

- **Human reference distributions.** Run the annotation exporter on counterfactual designs for the Soft No and Same Page families: ≥ 20 annotators per cell, 100-point allocations, observable text only.
  - Recruit from explicitly described, consented populations. Report the population by its stated communication norms, never as a proxy for identity.
  - Fit the LSSM to these data (it is already log-linear) and replace authored references with fitted ones.
- **Frontier diagnosis.** Run the same factorial designs on frontier models (backend already implemented) and on open models from 0.5B to 70B.
  - Key quantity: the **context-sensitivity slope**, i.e. how much the model's reading moves per unit of human-measured movement, as a function of scale.
  - Prediction: wording sensitivity falls faster with scale than context insensitivity does.
- *Paper 1: "Do language models read the relationship? Factorial estimates of context sensitivity in indirect speech."*

## 2. Which experiences teach it? (months 2–6)

- **Experience-selection laws.** At matched token budgets, compare contrastive minimal pairs, failure-targeted, diverse, multi-turn rollouts and isolated utterances. Map capability per token as a function of *experience structure*, not volume.
- **Causal data attribution.** Use influence approximations and leave-cluster-out retraining to identify which contrast clusters drive transfer to unseen domains. Test whether an information-theoretic criterion predicts usefulness ahead of time, e.g. the expected reduction in posterior entropy about the speaker's intent, which connects to rate-distortion views of "what to learn".
- **Counterfactual data augmentation as an intervention.** Generate contrast clusters for any utterance with a teacher, validate them against human data, and measure whether self-generated contrasts work as well as authored ones.
- *Paper 2: "Contrast, not volume: selecting interaction experiences that teach pragmatic inference."*

## 3. Learning from consequences (months 4–9)

- **RL in the environment.** Use GRPO or PPO on the Social RL environments, with rewards whose components are fitted to human ratings of interaction outcomes rather than authored.
  - Compare SFT on posterior targets with RL from consequence, and with both combined.
- **Simulator validity as a research object.** Calibrate interlocutor response models against human role-play data. Measure the sim-to-human gap with held-out human interlocutors.
  - Publish the reward audit methodology: constant, stylistic, greedy and planner baselines as a standard sanity suite for social reward functions.
- **Belief-state agents.** Agents that explicitly maintain and update a posterior over the other person's intent, trained with proper-scoring-rule rewards on their beliefs. This tests whether explicit belief tracking improves sample efficiency.
- *Paper 3: "Reward functions for social interaction are hypotheses: an audit suite and calibrated simulators."*

## 4. Generalization and safety of the capability (months 6–12)

- **Longitudinal common ground.** Multi-session relationships where history accumulates across episodes, a natural extension of the shared-history factor.
- **Manipulation risk.** Better pragmatic inference also enables better persuasion. Measure boundary-respecting behavior as capability grows, and treat it as a first-class outcome (the boundary-respect and coerced-yes outcomes are already tracked).
- **Regression and transfer.** Run a broader capability battery so pragmatic post-training doesn't trade off reasoning or honesty.

## What this program needs

- A data-centric research lead who owns the loop: hypothesis, environment, experience design, training, measurement.
- Modest compute: everything up to 8B fits on a small cluster.
- A human-data pipeline with consent and population documentation.
- Collaborators in pragmatics and computational cognitive science (RSA, theory of mind) and in RL theory (information-directed exploration, value of information).

The deliverable is not a dataset. It's **evidence about which experiences create which capabilities**, plus the tools that make that question answerable for capabilities beyond pragmatics.
