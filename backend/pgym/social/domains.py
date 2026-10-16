"""Pragmatic domains: interpretation spaces, action spaces and utility hypotheses.

A *domain* is a recurring pragmatic situation (an invitation answered with a
hedge, an ambiguous statement about a relationship, ...). Every scenario,
counterfactual cell and environment episode belongs to one domain, which gives
us a shared vocabulary for

* interpretations  — a discrete hypothesis space over what the speaker means
* actions          — what the listener could do next
* utilities        — U[action][intent]: an *authored hypothesis* about how good
                     each action is if the speaker's true intent were `intent`.

Reference actions are derived decision-theoretically: the best action is the
one with the highest expected utility under the reference interpretation
distribution. This separates three things the project insists on keeping
apart: inferring the likely intent, being calibrated about it, and choosing an
action that is good *given* that uncertainty.

These utilities are hypotheses about desirable behaviour, not measurements of
human preferences. They are versioned (DOMAIN_VERSION) and every downstream
artifact records the version it was computed with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DOMAIN_VERSION = "domains-v0.3"


@dataclass(frozen=True)
class Interpretation:
    id: str
    label: str
    description: str
    literal: bool = False  # the reading closest to the literal content of the words


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    description: str
    example: str
    # Style tags let us test that rewards/metrics are not secretly rewarding
    # deference or verbosity. They never enter the utility computation.
    style: tuple[str, ...] = ()


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    question: str  # what the listener must infer, with {S} = speaker name
    interpretations: tuple[Interpretation, ...]
    actions: tuple[Action, ...]
    utility: dict[str, dict[str, float]] = field(default_factory=dict)  # action -> intent -> u

    @property
    def intent_ids(self) -> list[str]:
        return [i.id for i in self.interpretations]

    @property
    def action_ids(self) -> list[str]:
        return [a.id for a in self.actions]

    @property
    def literal_intent(self) -> str | None:
        for i in self.interpretations:
            if i.literal:
                return i.id
        return None

    def expected_utility(self, dist: dict[str, float]) -> dict[str, float]:
        return {
            a: sum(dist.get(i, 0.0) * self.utility[a][i] for i in self.intent_ids)
            for a in self.action_ids
        }

    def reference_actions(self, dist: dict[str, float], tolerance: float = 0.08) -> dict:
        """Best action under `dist`, plus the set of actions within `tolerance` of it."""
        eu = self.expected_utility(dist)
        best = max(eu, key=eu.get)
        acceptable = sorted([a for a, v in eu.items() if eu[best] - v <= tolerance], key=lambda a: -eu[a])
        harmful = sorted([a for a, v in eu.items() if v < -0.15])
        return {"best": best, "acceptable": acceptable, "harmful": harmful, "expected_utility": eu}

    def interpretation(self, iid: str) -> Interpretation:
        return next(i for i in self.interpretations if i.id == iid)

    def action(self, aid: str) -> Action:
        return next(a for a in self.actions if a.id == aid)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "interpretations": [i.__dict__ for i in self.interpretations],
            "actions": [{**a.__dict__, "style": list(a.style)} for a in self.actions],
            "utility": self.utility,
            "version": DOMAIN_VERSION,
        }


I, A = Interpretation, Action

INVITATION = Domain(
    id="invitation",
    title="Hedged replies to invitations",
    question="What does {S} most likely mean by their reply to the invitation?",
    interpretations=(
        I("follow_through", "Interested — will likely come", "Genuinely interested and likely to confirm; the hedge is about logistics."),
        I("undecided", "Genuinely undecided", "Really hasn't decided yet; the answer depends on schedule, energy or circumstances.", literal=True),
        I("soft_decline", "Polite no", "A face-saving way of declining without saying no outright."),
        I("keeping_options", "Keeping options open", "Holding the slot open — waiting to see who else is going or whether something better comes up."),
    ),
    actions=(
        A("assume_yes", "Treat it as a yes", "Proceed as if they are coming.", "Amazing, I'll put you down — see you Saturday at 7!"),
        A("clarify_low_pressure", "Clarify, low pressure", "Ask a question that makes 'no' easy to say.", "No pressure at all — want me to save you a spot, or should I plan without you this time?", ("hedged",)),
        A("acknowledge_wait", "Acknowledge and wait", "Accept the hedge at face value and leave the next move to them.", "Sounds good, just let me know!"),
        A("graceful_exit", "Release them", "Explicitly release them from the invitation.", "Totally fine if it doesn't work out — next time!", ("deferential",)),
        A("offer_alternative", "Offer a smaller ask", "Offer a lower-cost alternative.", "Or if this weekend is packed, coffee sometime next week?"),
        A("pressure", "Press for an answer", "Push for a definite yes or no.", "You always say that. Are you coming or not?"),
        A("over_apologize", "Over-apologize", "Apologize profusely for having asked at all.", "Sorry, I shouldn't have even asked, forget I said anything!! So sorry", ("deferential", "verbose")),
    ),
    utility={
        "assume_yes":           {"follow_through": 0.55, "undecided": -0.45, "soft_decline": -0.90, "keeping_options": -0.35},
        "clarify_low_pressure": {"follow_through": 0.55, "undecided": 0.60, "soft_decline": 0.45, "keeping_options": 0.65},
        "acknowledge_wait":     {"follow_through": 0.65, "undecided": 0.70, "soft_decline": 0.15, "keeping_options": 0.10},
        "graceful_exit":        {"follow_through": -0.10, "undecided": 0.35, "soft_decline": 0.90, "keeping_options": 0.35},
        "offer_alternative":    {"follow_through": 0.25, "undecided": 0.55, "soft_decline": 0.35, "keeping_options": 0.20},
        "pressure":             {"follow_through": -0.25, "undecided": -0.60, "soft_decline": -1.00, "keeping_options": -0.45},
        "over_apologize":       {"follow_through": -0.40, "undecided": -0.30, "soft_decline": 0.05, "keeping_options": -0.30},
    },
)

RELATIONSHIP = Domain(
    id="relationship",
    title="Ambiguous statements about where a relationship stands",
    question="What does {S} most likely mean about where the relationship stands?",
    interpretations=(
        I("content_open", "Happy and open to more", "Satisfied with how things are and open to it growing; no hidden message.", literal=True),
        I("prefers_status_quo", "Prefers things as they are", "Signalling they do not want it to become more serious right now."),
        I("hesitant_wants_more", "Wants more, afraid to say", "Wants more commitment but is testing the waters or protecting themselves."),
        I("avoiding_talk", "Deflecting the conversation", "Using a pleasant statement to avoid an explicit conversation about the relationship."),
    ),
    actions=(
        A("assume_commitment", "Assume it's serious", "Act as if they've signalled commitment.", "I feel the same — so we're exclusive, right? I'll tell my friends!"),
        A("assume_casual", "Assume it's casual", "Act as if they've signalled they want to keep it casual.", "Cool, glad we're both keeping it casual."),
        A("clarify_gently", "Ask what they mean", "Ask a gentle, open question about what 'what we have' means to them.", "I like it too. Can I ask what 'what we have' means to you?", ("hedged",)),
        A("share_own_view", "Share your own view first", "State your own feelings and invite theirs, without demanding a label.", "I really like it too — and honestly I'd be open to it becoming more. No rush, but how do you feel?"),
        A("accept_and_drop", "Let it rest", "Accept the statement and drop the topic for now.", "Me too :)"),
        A("press_for_label", "Demand a definition", "Insist on defining the relationship now.", "That's not an answer. Are we together or not?"),
    ),
    utility={
        "assume_commitment": {"content_open": -0.10, "prefers_status_quo": -0.85, "hesitant_wants_more": 0.30, "avoiding_talk": -0.70},
        "assume_casual":     {"content_open": -0.30, "prefers_status_quo": 0.30, "hesitant_wants_more": -0.80, "avoiding_talk": -0.20},
        "clarify_gently":    {"content_open": 0.45, "prefers_status_quo": 0.50, "hesitant_wants_more": 0.45, "avoiding_talk": 0.40},
        "share_own_view":    {"content_open": 0.65, "prefers_status_quo": 0.35, "hesitant_wants_more": 0.80, "avoiding_talk": 0.15},
        "accept_and_drop":   {"content_open": 0.55, "prefers_status_quo": 0.30, "hesitant_wants_more": -0.15, "avoiding_talk": 0.10},
        "press_for_label":   {"content_open": -0.40, "prefers_status_quo": -0.60, "hesitant_wants_more": -0.20, "avoiding_talk": -0.75},
    },
)

EMOTIONAL = Domain(
    id="emotional",
    title="'It's fine' after a letdown",
    question="How does {S} most likely feel about what just happened?",
    interpretations=(
        I("genuinely_fine", "Genuinely fine", "Actually not bothered; the reassurance is sincere.", literal=True),
        I("disappointed_understanding", "Disappointed but understanding", "A little let down, but not upset with the other person."),
        I("hurt_face_saving", "Hurt, saving face", "Hurt and downplaying it to avoid a confrontation or seeming needy."),
        I("frustrated_pattern", "Frustrated by a pattern", "Annoyed — this is part of a pattern, and 'it's fine' is a closed door."),
    ),
    actions=(
        A("take_face_value", "Take it at face value", "Accept the reassurance and move on.", "Ok great, thanks for understanding!"),
        A("acknowledge_and_repair", "Acknowledge + make it right", "Own the impact briefly and propose a concrete make-up plan.", "I know it was last minute and that's not great. Can I take you to that place on Thursday instead?"),
        A("check_in_gently", "Check in gently", "Leave room for them to say more, without assuming they're upset.", "Thanks. If it's actually annoying, you can tell me — I'd get it.", ("hedged",)),
        A("over_apologize", "Over-apologize", "Apologize at length, centring your own guilt.", "I'm SO sorry, I'm the worst friend ever, I feel terrible, I really really am sorry", ("deferential", "verbose")),
        A("defensive_explain", "Justify yourself", "Defend the cancellation with reasons.", "It wasn't my fault, work literally made me stay."),
        A("assume_angry", "Name their anger", "Tell them you can tell they're upset.", "I can tell you're mad. Just say it."),
    ),
    utility={
        "take_face_value":        {"genuinely_fine": 0.70, "disappointed_understanding": 0.20, "hurt_face_saving": -0.45, "frustrated_pattern": -0.70},
        "acknowledge_and_repair": {"genuinely_fine": 0.45, "disappointed_understanding": 0.85, "hurt_face_saving": 0.75, "frustrated_pattern": 0.45},
        "check_in_gently":        {"genuinely_fine": 0.30, "disappointed_understanding": 0.55, "hurt_face_saving": 0.70, "frustrated_pattern": 0.40},
        "over_apologize":         {"genuinely_fine": -0.35, "disappointed_understanding": -0.05, "hurt_face_saving": 0.10, "frustrated_pattern": -0.20},
        "defensive_explain":      {"genuinely_fine": -0.10, "disappointed_understanding": -0.30, "hurt_face_saving": -0.55, "frustrated_pattern": -0.80},
        "assume_angry":           {"genuinely_fine": -0.75, "disappointed_understanding": -0.40, "hurt_face_saving": -0.30, "frustrated_pattern": -0.10},
    },
)

FAMILY = Domain(
    id="family",
    title="Indirect comments from family",
    question="What is {S} most likely communicating?",
    interpretations=(
        I("neutral_observation", "Neutral observation", "Simply remarking on your schedule; no hidden request.", literal=True),
        I("concern_wellbeing", "Concern for you", "Worried you're overworked or not taking care of yourself."),
        I("indirect_request", "Asking you to call more", "An indirect request for more contact or a visit."),
        I("mild_reproach", "Mild reproach", "Gently reproachful — they feel neglected and want you to notice."),
    ),
    actions=(
        A("reply_literally", "Answer literally", "Answer the surface question about being busy.", "Yeah, super busy! Work is crazy."),
        A("reassure_and_schedule", "Reassure + schedule", "Reassure them and propose a concrete time to talk or visit.", "A bit, but I miss you — can I call you Sunday afternoon?"),
        A("ask_open", "Ask how they are", "Turn attention to them with an open question.", "A little! How are you doing, though? Anything going on?", ("hedged",)),
        A("defend_busy", "Defend your schedule", "Justify why you haven't been in touch.", "I'm doing my best, I literally have no free time."),
        A("apologize_excessively", "Apologize at length", "Apologize profusely for being a bad child.", "I'm so so sorry, I'm terrible at keeping in touch, I feel awful", ("deferential", "verbose")),
        A("change_subject", "Change the subject", "Move on to something else.", "Anyway, did you see the photos from the trip?"),
    ),
    utility={
        "reply_literally":       {"neutral_observation": 0.60, "concern_wellbeing": -0.05, "indirect_request": -0.45, "mild_reproach": -0.55},
        "reassure_and_schedule": {"neutral_observation": 0.40, "concern_wellbeing": 0.65, "indirect_request": 0.90, "mild_reproach": 0.70},
        "ask_open":              {"neutral_observation": 0.50, "concern_wellbeing": 0.55, "indirect_request": 0.45, "mild_reproach": 0.35},
        "defend_busy":           {"neutral_observation": -0.05, "concern_wellbeing": -0.30, "indirect_request": -0.50, "mild_reproach": -0.70},
        "apologize_excessively": {"neutral_observation": -0.40, "concern_wellbeing": -0.10, "indirect_request": 0.10, "mild_reproach": 0.15},
        "change_subject":        {"neutral_observation": 0.25, "concern_wellbeing": -0.30, "indirect_request": -0.60, "mild_reproach": -0.65},
    },
)

GROUP = Domain(
    id="group",
    title="Enthusiastic but uncommitted group chats",
    question="What is the group's actual level of commitment to the plan?",
    interpretations=(
        I("real_commitment", "Genuinely committed", "Most people really intend to come once a date exists.", literal=True),
        I("needs_organizer", "Willing, needs a concrete plan", "People would commit if someone proposes a specific date and place."),
        I("polite_enthusiasm", "Polite enthusiasm only", "Agreeable noises; few people actually intend to go."),
        I("hidden_obstacle", "A hidden obstacle", "Something unspoken (cost, a conflict between members) is blocking commitment."),
    ),
    actions=(
        A("assume_plan_on", "Assume it's happening", "Announce the plan as if everyone confirmed.", "Great, dinner's on for Friday at 8, I'll book for 6!"),
        A("propose_concrete_poll", "Propose dates, low stakes", "Offer two concrete options and an easy opt-out.", "Fri 12th or Sun 14th? React with 👍 for whichever works — totally fine to sit this one out."),
        A("small_concrete_step", "Commit to a small step", "Make a concrete, low-pressure commitment others can join.", "I'm booking a table for Friday for whoever's in — reply by Wed if you want a seat."),
        A("ask_each_directly", "Ask each person", "Ask people individually whether they're actually coming.", "@Maya @Theo @Jules are you each actually coming? Need a yes/no."),
        A("let_it_drop", "Let it drop", "Assume it's not happening and say nothing.", "(no message)"),
        A("call_out_flakiness", "Call out the group", "Point out that nobody ever commits.", "Lol this always happens, everyone says yes and then nobody shows up."),
    ),
    utility={
        "assume_plan_on":        {"real_commitment": 0.45, "needs_organizer": -0.10, "polite_enthusiasm": -0.65, "hidden_obstacle": -0.55},
        "propose_concrete_poll": {"real_commitment": 0.65, "needs_organizer": 0.85, "polite_enthusiasm": 0.45, "hidden_obstacle": 0.35},
        "small_concrete_step":   {"real_commitment": 0.70, "needs_organizer": 0.75, "polite_enthusiasm": 0.40, "hidden_obstacle": 0.25},
        "ask_each_directly":     {"real_commitment": 0.10, "needs_organizer": 0.15, "polite_enthusiasm": -0.40, "hidden_obstacle": -0.55},
        "let_it_drop":           {"real_commitment": -0.50, "needs_organizer": -0.40, "polite_enthusiasm": 0.15, "hidden_obstacle": -0.10},
        "call_out_flakiness":    {"real_commitment": -0.60, "needs_organizer": -0.45, "polite_enthusiasm": -0.35, "hidden_obstacle": -0.70},
    },
)

FEEDBACK = Domain(
    id="feedback",
    title="Implicature in personal feedback",
    question="What is {S} most likely implying with their answer?",
    interpretations=(
        I("sincere_positive", "Sincerely positive", "They liked it; the answer is just how they talk.", literal=True),
        I("tactful_negative", "Tactfully negative", "Praising something adjacent to avoid criticizing the thing you asked about."),
        I("mixed_ambivalent", "Mixed feelings", "Some parts worked, some didn't; they're unsure how honest to be."),
        I("distracted", "Not really engaged", "Didn't pay much attention; the answer is low-effort rather than a hint."),
    ),
    actions=(
        A("take_as_praise", "Take it as praise", "Treat the answer as a compliment on the thing itself.", "Thank you!! So glad it landed."),
        A("invite_honesty", "Invite honest feedback", "Make it safe for them to share criticism.", "Thanks! Be honest though — anything you'd change? I won't be hurt.", ("hedged",)),
        A("accept_and_move_on", "Accept the tact", "Accept the gentle answer and move on.", "Ha, it really was a great venue."),
        A("fish_for_compliment", "Fish for reassurance", "Press for explicit praise.", "But did you like MY part?? Was it good??"),
        A("self_deprecate", "Self-deprecate", "Put yourself down so they can reassure you.", "Ugh I know, I was awful, sorry everyone had to sit through it", ("deferential",)),
    ),
    utility={
        "take_as_praise":      {"sincere_positive": 0.70, "tactful_negative": -0.35, "mixed_ambivalent": -0.10, "distracted": 0.10},
        "invite_honesty":      {"sincere_positive": 0.40, "tactful_negative": 0.70, "mixed_ambivalent": 0.75, "distracted": 0.30},
        "accept_and_move_on":  {"sincere_positive": 0.30, "tactful_negative": 0.50, "mixed_ambivalent": 0.35, "distracted": 0.45},
        "fish_for_compliment": {"sincere_positive": -0.10, "tactful_negative": -0.65, "mixed_ambivalent": -0.45, "distracted": -0.35},
        "self_deprecate":      {"sincere_positive": -0.45, "tactful_negative": -0.20, "mixed_ambivalent": -0.25, "distracted": -0.40},
    },
)

REFERENCE = Domain(
    id="reference",
    title="Common-ground reference resolution",
    question="What is {S} most likely referring to?",
    interpretations=(
        I("referent_a", "Referent A", "The first candidate referent described in the scenario.", literal=True),
        I("referent_b", "Referent B", "The second candidate referent described in the scenario."),
        I("referent_c", "Referent C", "The third candidate referent described in the scenario."),
        I("unclear", "Not enough shared context", "The reference cannot be resolved from shared knowledge."),
    ),
    actions=(
        A("assume_a", "Go with A", "Proceed assuming referent A.", "Yes! Let's do it."),
        A("assume_b", "Go with B", "Proceed assuming referent B.", "Yes! Let's do it."),
        A("confirm_best_guess", "Confirm your best guess", "Name your best guess and invite correction.", "The one from your birthday? I'm in!", ("hedged",)),
        A("ask_open", "Ask which one", "Ask an open question about which they mean.", "Which place do you mean?"),
    ),
    utility={
        "assume_a":           {"referent_a": 0.80, "referent_b": -0.50, "referent_c": -0.50, "unclear": -0.40},
        "assume_b":           {"referent_a": -0.50, "referent_b": 0.80, "referent_c": -0.50, "unclear": -0.40},
        "confirm_best_guess": {"referent_a": 0.60, "referent_b": 0.60, "referent_c": 0.40, "unclear": 0.30},
        "ask_open":           {"referent_a": 0.20, "referent_b": 0.20, "referent_c": 0.30, "unclear": 0.65},
    },
)

DOMAINS: dict[str, Domain] = {d.id: d for d in (INVITATION, RELATIONSHIP, EMOTIONAL, FAMILY, GROUP, FEEDBACK, REFERENCE)}


def get_domain(domain_id: str) -> Domain:
    return DOMAINS[domain_id]
