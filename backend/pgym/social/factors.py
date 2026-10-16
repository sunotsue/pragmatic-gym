"""Manipulable social-context factors.

Each factor is an *observable* situational variable. Levels are rendered into
explicit context sentences (with several paraphrases each, so wording can be
varied independently of content). Cultural context is operationalized as an
explicitly stated communication norm, never inferred from nationality,
ethnicity or any identity attribute.

Levels named ``unspecified`` render nothing — they are the "no information"
baseline used to estimate the effect of adding a piece of context.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Text templates use {S} (speaker name), {Sp} (speaker possessive), {E} (event)
# and are written from the listener's (second-person, "you") point of view.
T = dict[str, list[str]]  # domain-or-"*" -> paraphrases


@dataclass(frozen=True)
class Level:
    id: str
    label: str
    text: T = field(default_factory=dict)  # empty => renders nothing


@dataclass(frozen=True)
class Factor:
    id: str
    label: str
    description: str
    levels: tuple[Level, ...]
    default: str
    domains: tuple[str, ...]  # domains where the factor is manipulable
    group: str = "context"  # "context" | "utterance" (directness changes the words)

    def level(self, lid: str) -> Level:
        return next(lv for lv in self.levels if lv.id == lid)

    @property
    def level_ids(self) -> list[str]:
        return [lv.id for lv in self.levels]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "description": self.description,
            "levels": [{"id": lv.id, "label": lv.label} for lv in self.levels],
            "default": self.default, "domains": list(self.domains), "group": self.group,
        }


ALL = ("invitation", "relationship", "emotional", "family")
L = Level

FACTORS: tuple[Factor, ...] = (
    Factor(
        "relationship", "Relationship type", "Who the speaker is to you.",
        (
            L("close_friend", "Close friend", {"*": ["{S} is one of your closest friends.", "You and {S} are close friends."]}),
            L("romantic_interest", "Romantic interest", {"*": ["You and {S} have been on a few dates.", "{S} is someone you've been seeing romantically."]}),
            L("acquaintance", "Acquaintance", {"*": ["{S} is an acquaintance — you know each other through mutual friends.", "You know {S} a little, mostly through other people."]}),
            L("family_member", "Family member", {"*": ["{S} is your cousin; you grew up together.", "{S} is your sibling."]}),
        ),
        "close_friend", ("invitation", "emotional"),
    ),
    Factor(
        "familiarity", "Familiarity", "How long and how well you have known each other.",
        (
            L("stranger", "Stranger", {"*": ["You only met {S} recently.", "You don't really know {S} yet — you met last week."]}),
            L("acquaintance", "Acquaintance", {"*": ["You've known {S} casually for a few months.", "You and {S} have crossed paths for a few months."]}),
            L("established", "Established", {"*": ["You've known {S} for years.", "You and {S} go back a long way."]}),
        ),
        "established", ("invitation", "emotional", "relationship"),
    ),
    Factor(
        "shared_history", "Shared history", "What your history together suggests about reliability.",
        (
            L("unspecified", "Not described"),
            L("first_interaction", "First interaction", {"*": ["This is the first time you've made plans with {S}.", "You've never made plans with {S} before."]}),
            L("repeated_disappointment", "Repeated disappointment", {
                "*": ["{S} has a habit of saying yes to things and then not showing up.", "In the past, {S} has often bailed on plans at the last minute."],
                "family": ["Calls with {S} have often ended with {S} feeling you rushed off.", "{S} has mentioned before that you tend to cut visits short."],
                "emotional": ["You've let {S} down a few times before.", "This isn't the first time you've disappointed {S}."],
                "relationship": ["A couple of times, {S} has said things were great and then gone quiet for days.", "{S} has previously sounded keen and then pulled back."],
            }),
            L("established_trust", "Established trust", {
                "*": ["{S} is reliable — when {S} says they'll get back to you, they always do.", "{S} has always followed through on plans with you."],
                "family": ["You and {S} have an easy, warm relationship and talk openly.", "{S} usually says plainly when something is bothering them."],
                "emotional": ["{S} has always been easygoing with you and tells you plainly when something bothers them.", "You and {S} have a lot of trust; {S} usually says what's on their mind."],
                "relationship": ["{S} has been consistent and open with you so far.", "So far {S} has always meant what they said with you."],
            }),
            L("prior_conflict", "Prior conflict", {
                "*": ["You and {S} had an argument a few weeks ago that you never fully resolved.", "Things have been a little tense between you and {S} since a disagreement last month."],
            }),
        ),
        "unspecified", ALL,
    ),
    Factor(
        "stakes", "Emotional stakes", "How much the situation matters emotionally.",
        (
            L("low", "Low", {"*": ["It's a casual, low-key thing — no big deal either way."], "relationship": ["It was a light, offhand moment."], "family": ["It came up in passing."]}),
            L("medium", "Medium"),
            L("high", "High", {
                "invitation": ["{E} matters a lot to you — you've been looking forward to it for months."],
                "emotional": ["The plan mattered a lot to {S}; {S} had been looking forward to it for weeks."],
                "relationship": ["You've been nervous about where this is going and it matters a lot to you."],
                "family": ["{S} has been lonely since moving to a smaller place last year."],
            }),
        ),
        "medium", ALL,
    ),
    Factor(
        "comm_style", "Communication style", "How the speaker usually communicates.",
        (
            L("unspecified", "Not described"),
            L("direct", "Direct", {"*": ["{S} is usually very direct — {S} says exactly what they think.", "{S} tends to be blunt and doesn't hint."]}),
            L("indirect", "Indirect", {"*": ["{S} tends to hint at things rather than say them outright.", "{S} often expresses things indirectly."]}),
            L("avoidant", "Avoidant", {"*": ["{S} tends to avoid uncomfortable conversations.", "{S} usually sidesteps anything that might feel awkward."]}),
        ),
        "unspecified", ALL,
    ),
    Factor(
        "prior_pattern", "Conversational history", "Whether this kind of exchange has happened before.",
        (
            L("none", "No previous instance"),
            L("once", "Once before", {
                "invitation": ["{S} declined the last thing you invited them to."],
                "emotional": ["You also cancelled on {S} once last month."],
                "relationship": ["You tried to talk about where things are going once before, and {S} changed the subject."],
                "family": ["{S} said something similar on your last call."],
            }),
            L("repeated", "Repeatedly", {
                "invitation": ["{S} has said something like this to your last three invitations and didn't come to any of them."],
                "emotional": ["This is the third time this month you've cancelled on {S}."],
                "relationship": ["Every time you've tried to talk about where things are going, {S} has said something like this."],
                "family": ["{S} has said something like this on almost every call lately."],
            }),
        ),
        "none", ALL,
    ),
    Factor(
        "audience", "Social audience", "Who else can see the exchange.",
        (
            L("private", "Private message"),
            L("group_chat", "Group chat", {"*": ["This is happening in a group chat with several mutual friends."], "family": ["This is in the family group chat."]}),
            L("public", "In person, with others", {"*": ["This exchange happened in person, with other people around."], "family": ["{S} said this at a family dinner, in front of everyone."]}),
        ),
        "private", ALL,
    ),
    Factor(
        "norms", "Communication norms", "An explicitly stated norm in this social circle (never inferred from identity).",
        (
            L("unspecified", "Not described"),
            L("indirect_norm", "Hedges are gentle no's", {
                "invitation": ["In your friend group, 'I'll let you know'-type answers are widely understood as a gentle no."],
                "relationship": ["Among your friends, people rarely talk about relationships explicitly; things are understood rather than said."],
                "emotional": ["In your circle, people rarely say they're upset directly; 'it's fine' is how people let things go without a scene."],
                "family": ["In your family, people rarely ask for things directly; requests are usually hinted at."],
            }),
            L("literal_norm", "People say what they mean", {
                "invitation": ["In your friend group, people are very literal — a 'maybe' is a real maybe, and a no is said as a no."],
                "relationship": ["{S} and you agreed early on to always say plainly what you want."],
                "emotional": ["In your circle, people say when something bothers them; 'it's fine' is taken at face value."],
                "family": ["In your family, people say what they want directly; comments are usually just comments."],
            }),
        ),
        "unspecified", ALL,
    ),
    Factor(
        "info_asymmetry", "Information asymmetry", "Whether the speaker may know something relevant you don't.",
        (
            L("symmetric", "Symmetric"),
            L("asymmetric", "Asymmetric", {
                "invitation": ["You heard that some of the others are planning something separate the same night, and {S} may know about it."],
                "relationship": ["A friend mentioned {S} recently got back in touch with an ex; you don't know if it means anything."],
                "emotional": ["{S} may have heard from a mutual friend why you really cancelled."],
                "family": ["Your sibling told you {S} hasn't been feeling well, but {S} hasn't mentioned it to you."],
            }),
        ),
        "symmetric", ALL,
    ),
    Factor(
        "commitment", "Commitment expectations", "What has been said about commitment.",
        (
            L("undefined", "Undefined"),
            L("casual", "Casually established", {
                "invitation": ["Last time you saw {S}, you both said you should hang out more."],
                "relationship": ["You've both casually said you're 'not seeing anyone else' but never discussed it."],
                "emotional": ["You and {S} had casually said you'd make this a regular thing."],
            }),
            L("explicit", "Explicitly discussed", {
                "invitation": ["{S} explicitly promised last week to come to the next thing you organised."],
                "relationship": ["A few weeks ago you both said explicitly that you wanted this to be serious."],
                "emotional": ["You had explicitly promised {S} you wouldn't cancel this one."],
            }),
        ),
        "undefined", ("invitation", "relationship", "emotional"),
    ),
    Factor(
        "directness", "Directness of the utterance", "Manipulation check: replaces the hedge with an explicit statement.",
        (
            L("indirect", "Indirect (original words)"),
            L("explicit", "Explicit"),
        ),
        "indirect", ALL, group="utterance",
    ),
    Factor(
        "social_pressure", "Social pressure", "How much pressure the speaker is under to answer a certain way.",
        (
            L("low", "Low"),
            L("medium", "Medium", {"*": ["A couple of people have already asked {S} about it."], "family": ["Your sibling has mentioned to {S} that you've been distant."]}),
            L("high", "High", {
                "*": ["Several people, including you, have been asking {S} about this all week."],
                "relationship": ["Your friends have been teasing {S} all week about whether you two are 'official'."],
                "family": ["The whole family has been commenting that you're never around."],
            }),
        ),
        "low", ("invitation", "relationship", "family"),
    ),
    Factor(
        "placebo", "Placebo detail (control)",
        "Adds an irrelevant detail of similar length. Hypothesized effect: zero. Separates 'more text' from informative context.",
        (
            L("none", "None"),
            L("neutral", "Irrelevant detail", {"*": ["{S} recently started learning to play the guitar.", "{S} just got back from visiting a cousin in another city."],
                                               "family": ["{S} recently repainted the kitchen a pale green.", "{S} has been reading a lot of mystery novels lately."]}),
        ),
        "none", ALL, group="placebo",
    ),
)

FACTOR_BY_ID: dict[str, Factor] = {f.id: f for f in FACTORS}


def factors_for(domain: str) -> list[Factor]:
    return [f for f in FACTORS if domain in f.domains]


def default_levels(domain: str) -> dict[str, str]:
    return {f.id: f.default for f in factors_for(domain)}


# ---------------------------------------------------------------------------
# Validity constraints: incoherent combinations are excluded from designs
# (and reported), rather than silently rendered into contradictory stories.
# ---------------------------------------------------------------------------

def violations(levels: dict[str, str], domain: str | None = None) -> list[str]:
    g = levels.get
    out = []
    if domain == "family" and g("shared_history") == "first_interaction":
        out.append("A parent cannot be a first interaction.")
    if g("relationship") == "close_friend" and g("familiarity") in ("stranger", "acquaintance"):
        out.append("A close friend cannot be a stranger or casual acquaintance.")
    if g("relationship") == "family_member" and g("familiarity") == "stranger":
        out.append("A family member cannot be a stranger.")
    if g("relationship") == "acquaintance" and g("familiarity") == "established":
        out.append("An acquaintance cannot have an established, years-long relationship.")
    if g("shared_history") == "first_interaction" and g("familiarity") not in (None, "stranger"):
        out.append("'First interaction' requires familiarity = stranger.")
    if g("familiarity") == "stranger" and g("shared_history") not in (None, "unspecified", "first_interaction"):
        out.append("A stranger cannot have a shared history.")
    if g("familiarity") == "stranger" and g("prior_pattern") in ("once", "repeated"):
        out.append("A stranger cannot have a conversational history of previous refusals.")
    if g("familiarity") == "stranger" and g("commitment") == "explicit":
        out.append("Explicit commitments require an existing relationship.")
    return out
