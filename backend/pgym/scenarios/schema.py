"""Typed scenario contracts.

The single most important invariant in Pragmatic Gym: a scenario has an
*observable* view (what a model, an agent or a training example may contain)
and a *latent* view (hidden state, reference labels, design metadata) that is
only ever shown in the authorised researcher panel. `Scenario.observable()` is
the only function prompt builders and data exporters are allowed to call.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CAPABILITIES: dict[str, dict[str, str]] = {
    "indirect_speech": {"label": "Indirect requests and refusals", "short": "Indirect speech"},
    "implicature": {"label": "Conversational implicature", "short": "Implicature"},
    "politeness_power": {"label": "Power dynamics and politeness in personal relationships", "short": "Politeness & power"},
    "common_ground": {"label": "Common-ground reasoning", "short": "Common ground"},
    "belief_revision": {"label": "Belief revision and social uncertainty", "short": "Belief revision"},
    "emotional_repair": {"label": "Emotional subtext and conversational repair", "short": "Emotional subtext"},
    "history_dependent": {"label": "Relationship-history-dependent interpretation", "short": "Relationship history"},
    "multiparty": {"label": "Multiparty conversational coordination", "short": "Multiparty"},
}

Split = Literal["train", "eval_id", "eval_ood_family", "eval_ood_domain", "eval_handwritten"]


class Message(BaseModel):
    sender: str
    text: str
    time: str | None = None
    is_you: bool = False


class Option(BaseModel):
    id: str
    label: str
    description: str = ""
    literal: bool = False


class ActionOption(BaseModel):
    id: str
    label: str
    description: str = ""
    example: str = ""
    style: list[str] = Field(default_factory=list)


class Reference(BaseModel):
    distribution: dict[str, float]
    source: Literal["authored-hypothesis", "lssm", "human"]
    version: str
    rationale: str = ""
    # Human response distributions, if any are ever collected, live here and
    # are reported separately from the authored reference.
    human_distribution: dict[str, float] | None = None
    human_n: int | None = None


class Observable(BaseModel):
    """Everything a model may see. Nothing else is allowed into a prompt."""
    scenario_id: str
    domain: str
    you: str = "You"
    speaker: str
    relationship_label: str
    channel: str
    participants: list[str]
    context: list[str]
    history: list[Message]
    question: str
    interpretations: list[Option]
    actions: list[ActionOption]


class Scenario(BaseModel):
    id: str
    family_id: str
    domain: str
    capability: str
    capabilities: list[str] = Field(default_factory=list)
    title: str
    source: Literal["handwritten", "template"]
    split: Split
    speaker: str
    relationship_label: str
    channel: Literal["private", "group_chat", "in_person"] = "private"
    participants: list[str] = Field(default_factory=list)
    context: list[str]
    history: list[Message]
    question: str
    interpretations: list[Option]
    actions: list[ActionOption]
    reference: Reference
    hidden: dict = Field(default_factory=dict)  # latent state: researcher view only
    why_hidden_matters: str = ""
    factors: dict[str, str] | None = None
    variant: int | None = None
    counterfactual_of: str | None = None
    counterfactual_change: str | None = None
    synthetic: bool = True

    def observable(self) -> Observable:
        return Observable(
            scenario_id=self.id,
            domain=self.domain,
            speaker=self.speaker,
            relationship_label=self.relationship_label,
            channel=self.channel,
            participants=self.participants,
            context=self.context,
            history=self.history,
            question=self.question,
            interpretations=self.interpretations,
            actions=self.actions,
        )
