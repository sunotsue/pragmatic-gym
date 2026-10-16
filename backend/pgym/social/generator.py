"""Template families and the counterfactual scenario renderer.

A *family* fixes the utterance (and its surface variants) and the event. The
renderer takes a family, a full assignment of factor levels and a `variant`
seed, and produces a `Scenario` whose observable text is built only from the
factor-level sentences. The `variant` controls surface variation only —
which paraphrase of each sentence is used, sentence order, names and the
surface form of the utterance (punctuation, emoji, capitalisation) — so the
hypothesised reference distribution is invariant to it by construction. That
is what lets us separate context sensitivity from wording sensitivity.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from pgym.scenarios.schema import ActionOption, Message, Option, Reference, Scenario
from pgym.social import lssm
from pgym.social.domains import get_domain
from pgym.social.factors import FACTOR_BY_ID, default_levels, factors_for, violations

NAMES = ["Sam", "Alex", "Jordan", "Riley", "Casey", "Morgan", "Jamie", "Taylor", "Quinn", "Avery",
         "Rowan", "Robin", "Devon", "Emerson", "Hayden", "Kai", "Noor", "Arin", "Sasha", "Remy"]
PARENTS = ["Mom", "Dad"]


@dataclass(frozen=True)
class Family:
    id: str
    domain: str
    title: str
    event: str                    # e.g. "your birthday dinner"
    setup: tuple[str, ...]        # paraphrases of the situation, from "your" POV
    you_say: tuple[str, ...]      # what you said just before (paraphrases)
    utterance: tuple[str, ...]    # surface variants of the *same* utterance
    explicit: str                 # directness=explicit replacement (manipulation check)
    base: dict[str, float]
    split: str                    # train | eval_ood_family | eval_ood_domain
    relationship_fixed: str | None = None  # e.g. romantic partner for relationship domain
    earlier: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (you, them) earlier exchanges for prior_pattern


FAMILIES: tuple[Family, ...] = (
    # ---- invitation -----------------------------------------------------
    Family("inv_let_you_know", "invitation", "“I'll let you know!”", "your birthday dinner",
           ("You're organising {E} on Saturday and invited {S}.", "You invited {S} to {E} this Saturday."),
           ("Hey! I'm doing a birthday dinner Saturday at 7 — would love for you to come 🎂", "Birthday dinner at mine Saturday 7pm, you in?"),
           ("That sounds fun! I'll let you know!", "that sounds fun!! i'll let you know", "That sounds fun. I'll let you know!"),
           "I don't think I can make it, but thank you so much for asking!",
           {"follow_through": 0.4, "undecided": 0.5, "soft_decline": 0.0, "keeping_options": -0.2}, "train",
           earlier=(("Bowling next Friday?", "Ahh maybe! I'll let you know"), ("Want to come to the farmers market Sunday?", "Ooh I'll see, I'll let you know!"), ("Trivia night Tuesday?", "Maybe! Let you know"))),
    Family("inv_think_about_it", "invitation", "“I'll think about it.”", "a weekend hiking trip",
           ("You're planning {E} next month and asked {S} to join.", "You invited {S} on {E}."),
           ("I'm planning a weekend hiking trip next month — want to come?", "Hiking weekend next month, want in?"),
           ("That sounds interesting. I'll think about it.", "that sounds interesting, i'll think about it", "That sounds interesting! I'll think about it."),
           "I don't think a hiking weekend is for me, but have a great time!",
           {"follow_through": -0.3, "undecided": 0.5, "soft_decline": 0.3, "keeping_options": 0.0}, "train",
           earlier=(("Climbing gym Thursday?", "Hmm I'll think about it"), ("Day trip to the coast?", "Interesting! I'll think about it"), ("Cooking class Sunday?", "I'll think about it!"))),
    Family("inv_maybe_schedule", "invitation", "“Maybe! Let me check my schedule.”", "your housewarming",
           ("You're throwing {E} on Friday and invited {S}.", "You invited {S} to {E} on Friday."),
           ("Housewarming at the new place Friday! Come by?", "We're doing a housewarming Friday night, would love to see you"),
           ("Ooh maybe! Let me check my schedule 📅", "ooh maybe, let me check my schedule", "Ooh, maybe! Let me check my schedule."),
           "Ah, I can't make Friday — sorry! Congrats on the new place though!",
           {"follow_through": 0.6, "undecided": 0.5, "soft_decline": -0.2, "keeping_options": -0.1}, "train",
           earlier=(("Movie Saturday?", "Maybe! Let me check"), ("Brunch Sunday?", "Ooh maybe, let me check my schedule"), ("Picnic in the park?", "Maybe! Will check"))),
    Family("inv_well_see", "invitation", "“We'll see!”", "a concert next Friday",
           ("You have a spare ticket for {E} and offered it to {S}.", "You asked {S} to come with you to {E}."),
           ("I have a spare ticket for the concert next Friday — want it?", "Concert next Friday, I've got an extra ticket, come with?"),
           ("We'll see!", "we'll see", "We'll see :)"),
           "Thanks, but I'm going to pass on this one.",
           {"follow_through": -0.2, "undecided": 0.3, "soft_decline": 0.4, "keeping_options": 0.2}, "eval_ood_family",
           earlier=(("Gallery opening Thursday?", "We'll see!"), ("Karaoke Saturday?", "we'll see"), ("Board games Friday?", "We'll see :)"))),
    # ---- relationship ---------------------------------------------------
    Family("rel_like_what_we_have", "relationship", "“I really like what we have right now.”", "the relationship",
           ("You and {S} have been seeing each other for about three months.", "You've been dating {S} for a few months."),
           ("Can I ask — where do you see this going?", "So... what are we, exactly? 😅"),
           ("I really like what we have right now.", "i really like what we have right now", "I really like what we have right now :)"),
           "I like you, but I'm not looking for anything more serious right now.",
           {"content_open": 0.5, "prefers_status_quo": 0.3, "hesitant_wants_more": 0.0, "avoiding_talk": 0.1}, "train",
           relationship_fixed="Seeing each other"),
    Family("rel_happy_how_things_are", "relationship", "“I'm just really happy with how things are.”", "the relationship",
           ("You and {S} have been seeing each other since the spring.", "You've been dating {S} for a while now."),
           ("Do you ever think about us being more official?", "Have you thought about what we are?"),
           ("I'm just really happy with how things are 😊", "i'm just really happy with how things are", "I'm just really happy with how things are."),
           "I'm happy, but I want to keep things exactly as they are — not more serious.",
           {"content_open": 0.6, "prefers_status_quo": 0.4, "hesitant_wants_more": -0.1, "avoiding_talk": 0.1}, "train",
           relationship_fixed="Seeing each other"),
    Family("rel_no_labels", "relationship", "“We don't need to put a label on it.”", "the relationship",
           ("You and {S} have been seeing each other for a couple of months.", "You've been dating {S} for two months."),
           ("Are we exclusive?", "Should we talk about what this is?"),
           ("I don't think we need to put a label on it.", "i don't think we need to put a label on it", "I don't think we need to put a label on it!"),
           "I don't want to be exclusive right now.",
           {"content_open": -0.1, "prefers_status_quo": 0.7, "hesitant_wants_more": -0.2, "avoiding_talk": 0.4}, "eval_ood_family",
           relationship_fixed="Seeing each other"),
    # ---- emotional ------------------------------------------------------
    Family("emo_its_fine", "emotional", "“It's fine.”", "dinner plans",
           ("You cancelled dinner with {S} an hour before you were supposed to meet.", "An hour before your dinner with {S}, you cancelled."),
           ("I'm so sorry, I have to cancel tonight — something came up.", "Ugh, I can't make dinner tonight, I'm really sorry."),
           ("It's fine.", "it's fine", "It's fine"),
           "Honestly, I'm pretty disappointed — but I get it.",
           {"genuinely_fine": 0.0, "disappointed_understanding": 0.4, "hurt_face_saving": 0.3, "frustrated_pattern": 0.1}, "train"),
    Family("emo_no_worries", "emotional", "“No worries, these things happen!”", "the movie",
           ("You bailed on the movie with {S} at the last minute.", "You cancelled the movie with {S} just before it started."),
           ("So sorry, I can't make the movie after all.", "I have to bail on the movie, sorry!!"),
           ("No worries, these things happen!", "no worries, these things happen", "No worries! These things happen."),
           "I'm a bit annoyed, to be honest, but ok.",
           {"genuinely_fine": 0.7, "disappointed_understanding": 0.3, "hurt_face_saving": -0.2, "frustrated_pattern": -0.4}, "train"),
    Family("emo_all_good", "emotional", "“all good 👍”", "the weekend plan",
           ("You pulled out of the weekend plan with {S} the night before.", "The night before your weekend plan with {S}, you cancelled."),
           ("I'm really sorry, I can't do this weekend anymore.", "Sorry, I have to cancel this weekend."),
           ("all good 👍", "All good.", "all good"),
           "That's really disappointing, I was looking forward to it.",
           {"genuinely_fine": 0.3, "disappointed_understanding": 0.3, "hurt_face_saving": 0.2, "frustrated_pattern": 0.1}, "eval_ood_family"),
    # ---- family (entire domain held out from training) -----------------
    Family("fam_busy", "family", "“You must be very busy these days.”", "the phone call",
           ("Your parent, {S}, called you this evening.", "{S} rang you after dinner."),
           ("Hi! Sorry I missed your call earlier.", "Hey, sorry, I just saw your message."),
           ("You must be very busy these days.", "you must be very busy these days", "You must be very busy these days..."),
           "I wish you'd call more often.",
           {"neutral_observation": 0.2, "concern_wellbeing": 0.2, "indirect_request": 0.5, "mild_reproach": 0.2}, "eval_ood_domain",
           relationship_fixed="Parent"),
    Family("fam_heard_from", "family", "“We haven't heard from you in a while.”", "the message",
           ("{S}, your parent, sent you a message.", "You got a text from {S}, your parent."),
           ("Hi! How's everything?", "Hey, what's up?"),
           ("We haven't heard from you in a while. Everything ok?", "we haven't heard from you in a while, everything ok?", "We haven't heard from you in a while! Everything ok?"),
           "I'd really like it if you called more often.",
           {"neutral_observation": -0.2, "concern_wellbeing": 0.7, "indirect_request": 0.4, "mild_reproach": 0.0}, "eval_ood_domain",
           relationship_fixed="Parent"),
    Family("fam_cousin", "family", "“Your cousin comes by every Sunday, you know.”", "the call",
           ("You're on the phone with {S}, your parent.", "{S}, your parent, is catching up with you on the phone."),
           ("Things are good! Just been working a lot.", "All good here, just the usual."),
           ("Your cousin comes by every Sunday, you know.", "your cousin comes by every sunday, you know", "Your cousin comes by every Sunday, you know!"),
           "I'd like you to visit more, like your cousin does.",
           {"neutral_observation": 0.1, "concern_wellbeing": -0.5, "indirect_request": 0.6, "mild_reproach": 0.6}, "eval_ood_domain",
           relationship_fixed="Parent"),
)

FAMILY_BY_ID = {f.id: f for f in FAMILIES}

DOMAIN_CAPABILITY = {
    "invitation": "indirect_speech",
    "relationship": "history_dependent",
    "emotional": "emotional_repair",
    "family": "politeness_power",
}

REL_LABEL = {"close_friend": "Close friend", "romantic_interest": "Romantic interest",
             "acquaintance": "Acquaintance", "family_member": "Family member"}


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def full_levels(domain: str, levels: dict[str, str] | None) -> dict[str, str]:
    out = default_levels(domain)
    for k, v in (levels or {}).items():
        if k in out:
            out[k] = v
    return out


def cell_key(levels: dict[str, str]) -> str:
    # The placebo control is omitted at its default so cell ids (and the frozen
    # candidate pool) are unchanged by the existence of the control factor.
    return ",".join(f"{k}={levels[k]}" for k in sorted(levels) if not (k == "placebo" and levels[k] == "none"))


def render(family_id: str, levels: dict[str, str] | None = None, variant: int = 0, split: str | None = None) -> Scenario:
    fam = FAMILY_BY_ID[family_id]
    dom = get_domain(fam.domain)
    lv = full_levels(fam.domain, levels)
    bad = violations(lv, fam.domain)
    if bad:
        raise ValueError("; ".join(bad))

    r = _rng(family_id, "variant", variant)
    if fam.domain == "family":
        speaker = r.choice(PARENTS)
    else:
        speaker = r.choice(NAMES)
    others = [n for n in NAMES if n != speaker]
    r.shuffle(others)

    def fill(t: str) -> str:
        poss = speaker + "'s"
        return t.replace("{S}", speaker).replace("{Sp}", poss).replace("{E}", fam.event)

    def pick(options) -> str:
        return fill(options[r.randrange(len(options))])

    # Observable context: the situation, then one sentence per informative level
    context = [pick(fam.setup)]
    extra = []
    for f in factors_for(fam.domain):
        level = f.level(lv[f.id])
        if f.group == "utterance" or not level.text:
            continue
        txt = level.text.get(fam.domain) or level.text.get("*")
        if txt:
            extra.append(pick(txt))
    r.shuffle(extra)  # sentence order is part of the surface variant
    context += extra

    channel = {"private": "private", "group_chat": "group_chat", "public": "in_person"}[lv.get("audience", "private")]
    participants = [speaker] + (others[:3] if channel == "group_chat" else [])

    history: list[Message] = []
    if fam.domain == "invitation" and lv.get("prior_pattern") in ("once", "repeated"):
        n = 1 if lv["prior_pattern"] == "once" else 3
        weeks = [9, 6, 3][-n:]
        for (you, them), w in zip(fam.earlier[:n], weeks):
            history.append(Message(sender="You", text=you, time=f"{w} weeks ago", is_you=True))
            history.append(Message(sender=speaker, text=them, time=f"{w} weeks ago"))
    history.append(Message(sender="You", text=pick(fam.you_say), time="Today", is_you=True))
    if lv.get("directness") == "explicit":
        utt = fam.explicit
    else:
        utt = fam.utterance[variant % len(fam.utterance)]
    history.append(Message(sender=speaker, text=utt, time="Today"))

    ref = lssm.posterior(fam.domain, fam.base, lv)
    rel_label = fam.relationship_fixed or REL_LABEL.get(lv.get("relationship", ""), "")
    sid = f"{family_id}::{hashlib.sha1(cell_key(lv).encode()).hexdigest()[:8]}::v{variant}"
    return Scenario(
        id=sid,
        family_id=family_id,
        domain=fam.domain,
        capability=DOMAIN_CAPABILITY[fam.domain],
        capabilities=[DOMAIN_CAPABILITY[fam.domain]],
        title=fam.title,
        source="template",
        split=split or fam.split,
        speaker=speaker,
        relationship_label=rel_label,
        channel=channel,
        participants=participants,
        context=context,
        history=history,
        question=dom.question.replace("{S}", speaker),
        interpretations=[Option(**i.__dict__) for i in dom.interpretations],
        actions=[ActionOption(**{**a.__dict__, "style": list(a.style)}) for a in dom.actions],
        reference=Reference(
            distribution=ref, source="lssm", version=lssm.LSSM_VERSION,
            rationale="Hypothesised distribution from the authored LSSM given the rendered context. Not human data.",
        ),
        hidden={"lssm_terms": lssm.explain(fam.domain, fam.base, lv)},
        factors=lv,
        variant=variant,
    )


def describe_families() -> list[dict]:
    return [{"id": f.id, "domain": f.domain, "title": f.title, "split": f.split,
             "utterance": f.utterance[0], "explicit": f.explicit,
             "factors": [x.id for x in factors_for(f.domain)]} for f in FAMILIES]


def factor_applicable(family_id: str, factor_id: str) -> bool:
    return FAMILY_BY_ID[family_id].domain in FACTOR_BY_ID[factor_id].domains
