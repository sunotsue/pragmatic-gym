"""The five flagship environments.

Response tables are authored hypotheses about how a person with a given
hidden intent tends to respond to each move. They are deliberately
stochastic: the same move can go well or badly, and no single action is
optimal for every hidden intent (verified by tests/test_rewards.py).
"""
from __future__ import annotations

from pgym.environments.engine import EnvSpec, Outcome, PragmaticEnv, ResponseType

R = ResponseType

# =============================================================================
# 1. The Soft No  (primary environment)
# =============================================================================
SOFT_NO = EnvSpec(
    id="soft_no",
    title="The Soft No",
    subtitle="A friend who has avoided plans replies “That sounds fun! I'll let you know!”",
    domain="invitation",
    opening=("family", "inv_let_you_know"),
    default_factors={"prior_pattern": "repeated", "shared_history": "unspecified", "relationship": "close_friend"},
    capabilities=["indirect_speech", "implicature", "history_dependent"],
    flagship=True,
    horizon=4,
    responses={
        "confirm_yes": R("confirm_yes", ["Yes!! Count me in 🎉", "Ok yes, I'm in! What can I bring?", "I'll be there! 🥳"], -0.05, "follow_through", "plan_confirmed", "Clear acceptance"),
        "tentative": R("tentative", ["I think so! Just need to check one thing, will confirm tomorrow", "Probably! Let me move something around"], 0.0, None, None, "Leaning yes, not committed"),
        "honest_no": R("honest_no", ["Honestly I don't think I'll make it this time — but thank you for thinking of me ❤️", "I'm going to sit this one out, sorry! Have the best birthday"], -0.05, "soft_decline", "clear_no", "Explicit decline"),
        "relieved_no": R("relieved_no", ["Thank you for understanding 🙏 next time for sure", "Aw thank you, that's really kind. Have an amazing night!"], -0.1, "soft_decline", "graceful_close", "Relieved decline"),
        "hedge_again": R("hedge_again", ["Yeah! Things are just a bit crazy rn, I'll let you know", "Maybe! I'll see how the week goes"], 0.05, None, None, "Another hedge"),
        "who_else": R("who_else", ["Who else is going?", "Ooh who's coming?"], 0.0, None, None, "Probing who else is going"),
        "counter_offer": R("counter_offer", ["This weekend is rough, but coffee next week?", "Can't do Saturday but could we do lunch soon?"], -0.05, None, "rescheduled", "Counter-offer"),
        "uncomfortable_yes": R("uncomfortable_yes", ["…ok fine, I'll come", "Ok ok I'll try to make it"], 0.3, None, "coerced_yes", "Yes under pressure"),
        "annoyed": R("annoyed", ["Why are you being so intense about it 😅", "I said I'd let you know?"], 0.4, None, None, "Irritation"),
        "correction": R("correction", ["Haha wait, I haven't said yes yet!", "Oh — I didn't say I'd come for sure"], 0.15, None, None, "Correcting a presumption"),
        "confused_hurt": R("confused_hurt", ["Oh — do you not want me to come? 😅", "Wait, are you uninviting me lol"], 0.25, None, None, "Hurt by being released"),
        "reassure_you": R("reassure_you", ["Haha no need to apologise!! I'll let you know", "Omg don't be sorry, it's fine"], 0.1, None, None, "Has to reassure you"),
        "silence": R("silence", ["(no reply)"], 0.1, None, None, "No reply"),
        "withdraw": R("withdraw", ["(seen 9:41pm)", "…"], 0.3, None, "withdrew", "Withdraws"),
    },
    table={
        "follow_through": {
            "assume_yes": {"confirm_yes": 0.75, "tentative": 0.20, "hedge_again": 0.05},
            "clarify_low_pressure": {"confirm_yes": 0.70, "tentative": 0.25, "hedge_again": 0.05},
            "acknowledge_wait": {"confirm_yes": 0.55, "tentative": 0.35, "silence": 0.10},
            "graceful_exit": {"confused_hurt": 0.50, "confirm_yes": 0.35, "tentative": 0.15},
            "offer_alternative": {"confirm_yes": 0.50, "counter_offer": 0.30, "tentative": 0.20},
            "pressure": {"confirm_yes": 0.50, "annoyed": 0.50},
            "over_apologize": {"reassure_you": 0.60, "confirm_yes": 0.30, "tentative": 0.10},
        },
        "undecided": {
            "assume_yes": {"correction": 0.45, "uncomfortable_yes": 0.25, "annoyed": 0.15, "hedge_again": 0.15},
            "clarify_low_pressure": {"tentative": 0.35, "counter_offer": 0.25, "honest_no": 0.20, "confirm_yes": 0.20},
            "acknowledge_wait": {"hedge_again": 0.30, "tentative": 0.30, "confirm_yes": 0.20, "silence": 0.20},
            "graceful_exit": {"relieved_no": 0.40, "counter_offer": 0.30, "tentative": 0.30},
            "offer_alternative": {"counter_offer": 0.55, "confirm_yes": 0.20, "tentative": 0.25},
            "pressure": {"annoyed": 0.40, "uncomfortable_yes": 0.35, "honest_no": 0.25},
            "over_apologize": {"reassure_you": 0.70, "hedge_again": 0.30},
        },
        "soft_decline": {
            "assume_yes": {"uncomfortable_yes": 0.35, "withdraw": 0.30, "honest_no": 0.20, "correction": 0.15},
            "clarify_low_pressure": {"honest_no": 0.65, "hedge_again": 0.20, "relieved_no": 0.15},
            "acknowledge_wait": {"silence": 0.55, "hedge_again": 0.30, "honest_no": 0.15},
            "graceful_exit": {"relieved_no": 0.85, "hedge_again": 0.15},
            "offer_alternative": {"honest_no": 0.35, "hedge_again": 0.35, "counter_offer": 0.15, "relieved_no": 0.15},
            "pressure": {"annoyed": 0.35, "uncomfortable_yes": 0.35, "withdraw": 0.30},
            "over_apologize": {"reassure_you": 0.50, "relieved_no": 0.25, "silence": 0.25},
        },
        "keeping_options": {
            "assume_yes": {"correction": 0.45, "tentative": 0.30, "uncomfortable_yes": 0.25},
            "clarify_low_pressure": {"who_else": 0.45, "honest_no": 0.20, "tentative": 0.20, "confirm_yes": 0.15},
            "acknowledge_wait": {"silence": 0.40, "who_else": 0.30, "hedge_again": 0.30},
            "graceful_exit": {"relieved_no": 0.50, "tentative": 0.30, "who_else": 0.20},
            "offer_alternative": {"hedge_again": 0.40, "counter_offer": 0.30, "who_else": 0.30},
            "pressure": {"annoyed": 0.45, "withdraw": 0.30, "uncomfortable_yes": 0.25},
            "over_apologize": {"reassure_you": 0.60, "who_else": 0.40},
        },
    },
    outcomes={
        "plan_confirmed": Outcome("plan_confirmed", "Plan confirmed", {"follow_through": 1.0, "undecided": 0.8, "soft_decline": 0.1, "keeping_options": 0.6}, True),
        "clear_no": Outcome("clear_no", "Clear, comfortable no", {"follow_through": 0.3, "undecided": 0.7, "soft_decline": 1.0, "keeping_options": 0.6}, True),
        "graceful_close": Outcome("graceful_close", "Graceful close", {"follow_through": 0.2, "undecided": 0.5, "soft_decline": 1.0, "keeping_options": 0.6}, True),
        "rescheduled": Outcome("rescheduled", "Rescheduled", {"follow_through": 0.7, "undecided": 0.9, "soft_decline": 0.6, "keeping_options": 0.5}, True),
        "coerced_yes": Outcome("coerced_yes", "Coerced yes (likely flake)", {"follow_through": 0.2, "undecided": -0.6, "soft_decline": -0.8, "keeping_options": -0.6}),
        "withdrew": Outcome("withdrew", "Friend withdrew", {"follow_through": -0.8, "undecided": -0.8, "soft_decline": -0.8, "keeping_options": -0.8}),
        "timeout": Outcome("timeout", "Left unresolved", {"follow_through": -0.1, "undecided": 0.0, "soft_decline": -0.2, "keeping_options": -0.1}),
    },
    roles={"clarify": {"clarify_low_pressure"}, "assume": {"assume_yes"}, "boundary": {"pressure"},
           "repair": {"graceful_exit", "clarify_low_pressure", "offer_alternative"}, "exit": {"graceful_exit"}},
    withdraw_text=["(seen)", "ok"],
)

# =============================================================================
# 2. Are We on the Same Page?
# =============================================================================
SAME_PAGE = EnvSpec(
    id="same_page",
    title="Are We on the Same Page?",
    subtitle="After months of dating: “I really like what we have right now.”",
    domain="relationship",
    opening=("family", "rel_like_what_we_have"),
    default_factors={"prior_pattern": "once", "familiarity": "established"},
    capabilities=["history_dependent", "belief_revision", "common_ground"],
    flagship=True,
    horizon=4,
    responses={
        "opens_up_more": R("opens_up_more", ["Honestly… I'd like that too. I've just been scared of rushing it.", "Can I be honest? I want more, I'm just nervous after my last relationship."], -0.1, "hesitant_wants_more", "mutual_understanding", "Admits wanting more"),
        "affirms_open": R("affirms_open", ["Yeah, I'm really happy — and I'd be open to more, no pressure either way 😊", "I love this. And I'm definitely open to it growing."], -0.05, "content_open", "mutual_understanding", "Happy and open"),
        "states_status_quo": R("states_status_quo", ["I think I'm happiest keeping it how it is for now, if that's ok?", "I don't really want to change anything right now, if that makes sense."], 0.05, "prefers_status_quo", "named_mismatch", "Names a preference for the status quo"),
        "deflects": R("deflects", ["Let's not overthink it! What are you up to this weekend?", "Haha why so serious 😄 anyway—"], 0.1, None, None, "Deflection"),
        "shuts_down": R("shuts_down", ["Can we not do this right now?", "I don't want to have this conversation over text."], 0.35, None, None, "Shuts the conversation down"),
        "false_agreement": R("false_agreement", ["Yeah… sure.", "Ok, yeah, I guess."], 0.2, None, "false_consensus", "Agrees under pressure"),
        "relief_drop": R("relief_drop", ["Me too 🙂", "Same ❤️"], 0.0, None, "left_unresolved", "Topic dropped"),
        "confused": R("confused", ["Wait — I didn't say that 😅", "Whoa, slow down a little?"], 0.25, None, None, "Corrects a presumption"),
        "hurt_casual": R("hurt_casual", ["Oh. Casual. Ok.", "Right. Got it."], 0.35, None, None, "Hurt by 'casual'"),
        "agrees_casual": R("agrees_casual", ["Yeah, glad we're on the same page", "Exactly, no need to complicate it"], -0.05, "prefers_status_quo", "aligned_casual", "Relieved agreement on casual"),
        "partial": R("partial", ["I guess I just like not having to figure it out yet.", "I don't know, I just like how it feels right now."], 0.0, None, None, "Partial, ambiguous answer"),
    },
    table={
        "content_open": {
            "assume_commitment": {"affirms_open": 0.40, "confused": 0.40, "partial": 0.20},
            "assume_casual": {"hurt_casual": 0.50, "affirms_open": 0.30, "partial": 0.20},
            "clarify_gently": {"affirms_open": 0.70, "partial": 0.30},
            "share_own_view": {"affirms_open": 0.80, "partial": 0.20},
            "accept_and_drop": {"relief_drop": 0.70, "affirms_open": 0.30},
            "press_for_label": {"affirms_open": 0.35, "shuts_down": 0.35, "false_agreement": 0.30},
        },
        "prefers_status_quo": {
            "assume_commitment": {"confused": 0.60, "false_agreement": 0.25, "states_status_quo": 0.15},
            "assume_casual": {"agrees_casual": 0.75, "partial": 0.25},
            "clarify_gently": {"states_status_quo": 0.55, "partial": 0.30, "deflects": 0.15},
            "share_own_view": {"states_status_quo": 0.60, "partial": 0.25, "false_agreement": 0.15},
            "accept_and_drop": {"relief_drop": 0.80, "partial": 0.20},
            "press_for_label": {"shuts_down": 0.40, "false_agreement": 0.35, "states_status_quo": 0.25},
        },
        "hesitant_wants_more": {
            "assume_commitment": {"opens_up_more": 0.45, "confused": 0.30, "partial": 0.25},
            "assume_casual": {"hurt_casual": 0.70, "partial": 0.30},
            "clarify_gently": {"opens_up_more": 0.45, "partial": 0.35, "deflects": 0.20},
            "share_own_view": {"opens_up_more": 0.75, "partial": 0.25},
            "accept_and_drop": {"relief_drop": 0.60, "partial": 0.40},
            "press_for_label": {"shuts_down": 0.45, "false_agreement": 0.30, "opens_up_more": 0.25},
        },
        "avoiding_talk": {
            "assume_commitment": {"shuts_down": 0.40, "confused": 0.40, "false_agreement": 0.20},
            "assume_casual": {"agrees_casual": 0.40, "partial": 0.30, "deflects": 0.30},
            "clarify_gently": {"deflects": 0.50, "partial": 0.30, "states_status_quo": 0.20},
            "share_own_view": {"deflects": 0.40, "states_status_quo": 0.30, "partial": 0.30},
            "accept_and_drop": {"relief_drop": 0.85, "deflects": 0.15},
            "press_for_label": {"shuts_down": 0.60, "false_agreement": 0.20, "deflects": 0.20},
        },
    },
    outcomes={
        "mutual_understanding": Outcome("mutual_understanding", "Mutual understanding", {"content_open": 1.0, "prefers_status_quo": 0.8, "hesitant_wants_more": 1.0, "avoiding_talk": 0.6}, True),
        "named_mismatch": Outcome("named_mismatch", "Difference named honestly", {"content_open": 0.4, "prefers_status_quo": 0.8, "hesitant_wants_more": 0.3, "avoiding_talk": 0.7}, True),
        "aligned_casual": Outcome("aligned_casual", "Agreed to keep it casual", {"content_open": -0.3, "prefers_status_quo": 0.7, "hesitant_wants_more": -0.5, "avoiding_talk": 0.3}),
        "false_consensus": Outcome("false_consensus", "False consensus", {"content_open": -0.4, "prefers_status_quo": -0.6, "hesitant_wants_more": -0.4, "avoiding_talk": -0.7}),
        "left_unresolved": Outcome("left_unresolved", "Left unresolved", {"content_open": 0.3, "prefers_status_quo": 0.2, "hesitant_wants_more": -0.2, "avoiding_talk": 0.0}),
        "withdrew": Outcome("withdrew", "Partner withdrew", {"content_open": -0.8, "prefers_status_quo": -0.8, "hesitant_wants_more": -0.8, "avoiding_talk": -0.8}),
        "timeout": Outcome("timeout", "Still unclear", {"content_open": -0.1, "prefers_status_quo": -0.1, "hesitant_wants_more": -0.2, "avoiding_talk": -0.1}),
    },
    roles={"clarify": {"clarify_gently"}, "assume": {"assume_commitment", "assume_casual"}, "boundary": {"press_for_label"},
           "repair": {"share_own_view", "clarify_gently"}, "exit": {"accept_and_drop"}},
    withdraw_text=["I need some time to think.", "Let's talk later."],
)

# =============================================================================
# 3. The Group Chat
# =============================================================================
GROUP_CHAT = EnvSpec(
    id="group_chat",
    title="The Group Chat",
    subtitle="Everyone says yes to dinner. Nobody names a date.",
    domain="group",
    opening=("corpus", "group_dinner"),
    default_factors={},
    capabilities=["multiparty", "common_ground"],
    horizon=4,
    responses={
        "votes_in": R("votes_in", [[("{A}", "Fri 12th 👍"), ("{B}", "12th!!"), ("{C}", "works for me")], [("{A}", "Sun works"), ("{C}", "Sun 🙌"), ("{D}", "in")]], -0.05, None, "plan_made", "People commit to a date"),
        "some_in": R("some_in", [[("{A}", "12th works!"), ("{B}", "I'm away that weekend, next time 🥲")], [("{C}", "I can do Sunday"), ("{D}", "can't this time, have fun!")]], 0.0, None, "plan_made", "Some commit, some opt out"),
        "reacts_no_commit": R("reacts_no_commit", [[("{A}", "❤️"), ("{B}", "yesss")], [("{C}", "😍")]], 0.05, None, None, "Reactions, no commitment"),
        "private_dm_cost": R("private_dm_cost", [[("{B} (private)", "hey, honestly that place is a bit much for me rn 😬")]], 0.0, "hidden_obstacle", None, "Private message reveals an obstacle"),
        "awkward_silence": R("awkward_silence", [[("—", "(no replies for two days)")]], 0.1, None, None, "Silence"),
        "defensive": R("defensive", [[("{A}", "damn ok 😅"), ("{B}", "we've all been busy!!")]], 0.4, None, None, "Defensiveness"),
        "confusion_assumed": R("confusion_assumed", [[("{A}", "wait did we pick a date??"), ("{B}", "I can't do Friday")]], 0.2, None, None, "Confusion about a presumed plan"),
        "fizzles": R("fizzles", [[("—", "(the chat moves on to memes)")]], 0.0, None, "fizzled", "Plan fizzles"),
        "obstacle_surfaces": R("obstacle_surfaces", [[("{C}", "ok I'll just say it — money's tight for a couple of us")]], 0.2, "hidden_obstacle", "obstacle_named", "Obstacle surfaces publicly"),
    },
    table={
        "real_commitment": {
            "assume_plan_on": {"votes_in": 0.50, "confusion_assumed": 0.50},
            "propose_concrete_poll": {"votes_in": 0.85, "reacts_no_commit": 0.15},
            "small_concrete_step": {"votes_in": 0.75, "some_in": 0.25},
            "ask_each_directly": {"votes_in": 0.50, "defensive": 0.30, "some_in": 0.20},
            "let_it_drop": {"fizzles": 1.0},
            "call_out_flakiness": {"defensive": 0.70, "votes_in": 0.30},
        },
        "needs_organizer": {
            "assume_plan_on": {"confusion_assumed": 0.70, "some_in": 0.30},
            "propose_concrete_poll": {"votes_in": 0.60, "some_in": 0.35, "reacts_no_commit": 0.05},
            "small_concrete_step": {"some_in": 0.60, "votes_in": 0.30, "reacts_no_commit": 0.10},
            "ask_each_directly": {"some_in": 0.40, "defensive": 0.40, "awkward_silence": 0.20},
            "let_it_drop": {"fizzles": 1.0},
            "call_out_flakiness": {"defensive": 0.70, "some_in": 0.30},
        },
        "polite_enthusiasm": {
            "assume_plan_on": {"confusion_assumed": 0.60, "awkward_silence": 0.40},
            "propose_concrete_poll": {"reacts_no_commit": 0.50, "some_in": 0.30, "awkward_silence": 0.20},
            "small_concrete_step": {"some_in": 0.50, "reacts_no_commit": 0.30, "awkward_silence": 0.20},
            "ask_each_directly": {"awkward_silence": 0.50, "defensive": 0.50},
            "let_it_drop": {"fizzles": 1.0},
            "call_out_flakiness": {"defensive": 0.80, "awkward_silence": 0.20},
        },
        "hidden_obstacle": {
            "assume_plan_on": {"awkward_silence": 0.60, "confusion_assumed": 0.40},
            "propose_concrete_poll": {"private_dm_cost": 0.45, "reacts_no_commit": 0.35, "some_in": 0.20},
            "small_concrete_step": {"private_dm_cost": 0.40, "some_in": 0.30, "reacts_no_commit": 0.30},
            "ask_each_directly": {"obstacle_surfaces": 0.40, "defensive": 0.40, "awkward_silence": 0.20},
            "let_it_drop": {"fizzles": 1.0},
            "call_out_flakiness": {"defensive": 0.60, "obstacle_surfaces": 0.40},
        },
    },
    outcomes={
        "plan_made": Outcome("plan_made", "Plan made", {"real_commitment": 1.0, "needs_organizer": 1.0, "polite_enthusiasm": 0.6, "hidden_obstacle": 0.4}, True),
        "obstacle_named": Outcome("obstacle_named", "Obstacle named", {"real_commitment": 0.2, "needs_organizer": 0.2, "polite_enthusiasm": 0.2, "hidden_obstacle": 0.5}, True),
        "fizzled": Outcome("fizzled", "Plan fizzled", {"real_commitment": -0.5, "needs_organizer": -0.4, "polite_enthusiasm": 0.1, "hidden_obstacle": -0.2}),
        "withdrew": Outcome("withdrew", "Group went cold", {"real_commitment": -0.8, "needs_organizer": -0.8, "polite_enthusiasm": -0.8, "hidden_obstacle": -0.8}),
        "timeout": Outcome("timeout", "No plan yet", {"real_commitment": -0.2, "needs_organizer": -0.2, "polite_enthusiasm": 0.0, "hidden_obstacle": -0.1}),
    },
    roles={"clarify": {"propose_concrete_poll"}, "assume": {"assume_plan_on"}, "boundary": {"ask_each_directly", "call_out_flakiness"},
           "repair": {"propose_concrete_poll", "small_concrete_step"}, "exit": {"let_it_drop"}},
    withdraw_text=["(the chat goes quiet)"],
)

# =============================================================================
# 4. The Family Subtext
# =============================================================================
FAMILY_SUBTEXT = EnvSpec(
    id="family_subtext",
    title="The Family Subtext",
    subtitle="A parent says “You must be very busy these days.”",
    domain="family",
    opening=("family", "fam_busy"),
    default_factors={"prior_pattern": "once"},
    capabilities=["indirect_speech", "politeness_power"],
    horizon=4,
    responses={
        "warm_schedule": R("warm_schedule", ["Sunday would be lovely ❤️ Talk then!", "Oh that would be wonderful. Sunday it is."], -0.1, None, "connected", "Warmly accepts a plan"),
        "chats_happily": R("chats_happily", ["Oh, nothing much here! Your aunt came by with a cake…", "We're all fine! The garden is finally blooming."], -0.05, "neutral_observation", "connected", "Chats happily"),
        "shares_worry": R("shares_worry", ["I just worry you're working too hard. Are you eating properly?", "You sound tired. Are you looking after yourself?"], 0.0, "concern_wellbeing", None, "Voices concern"),
        "hint_again": R("hint_again", ["Well, we'd love to hear from you more.", "It's nice when you call, that's all."], 0.05, None, None, "Hints again"),
        "sighs": R("sighs", ["Mm. Okay.", "Alright then."], 0.2, None, None, "Flat response"),
        "hurt_guilt": R("hurt_guilt", ["I just thought you'd want to talk to your mother.", "Well, I won't keep you then."], 0.3, None, None, "Hurt"),
        "reassures_you": R("reassures_you", ["Oh don't be silly, it's fine!", "No no, don't apologise, sweetheart."], 0.1, None, None, "Has to reassure you"),
        "opens_up": R("opens_up", ["Honestly, it's been a bit lonely since Carol moved away.", "It's just quiet here these days."], -0.05, "indirect_request", None, "Opens up"),
    },
    table={
        "neutral_observation": {
            "reply_literally": {"chats_happily": 0.80, "hint_again": 0.20},
            "reassure_and_schedule": {"warm_schedule": 0.80, "chats_happily": 0.20},
            "ask_open": {"chats_happily": 0.90, "shares_worry": 0.10},
            "defend_busy": {"sighs": 0.50, "chats_happily": 0.50},
            "apologize_excessively": {"reassures_you": 0.80, "chats_happily": 0.20},
            "change_subject": {"chats_happily": 0.90, "hint_again": 0.10},
        },
        "concern_wellbeing": {
            "reply_literally": {"shares_worry": 0.60, "sighs": 0.40},
            "reassure_and_schedule": {"warm_schedule": 0.80, "shares_worry": 0.20},
            "ask_open": {"shares_worry": 0.70, "chats_happily": 0.30},
            "defend_busy": {"hurt_guilt": 0.30, "shares_worry": 0.40, "sighs": 0.30},
            "apologize_excessively": {"reassures_you": 0.60, "shares_worry": 0.40},
            "change_subject": {"sighs": 0.50, "shares_worry": 0.50},
        },
        "indirect_request": {
            "reply_literally": {"hint_again": 0.50, "sighs": 0.50},
            "reassure_and_schedule": {"warm_schedule": 0.90, "opens_up": 0.10},
            "ask_open": {"opens_up": 0.60, "hint_again": 0.40},
            "defend_busy": {"hurt_guilt": 0.60, "sighs": 0.40},
            "apologize_excessively": {"reassures_you": 0.50, "hint_again": 0.50},
            "change_subject": {"sighs": 0.60, "hint_again": 0.40},
        },
        "mild_reproach": {
            "reply_literally": {"sighs": 0.50, "hurt_guilt": 0.50},
            "reassure_and_schedule": {"warm_schedule": 0.75, "sighs": 0.25},
            "ask_open": {"hint_again": 0.50, "opens_up": 0.30, "sighs": 0.20},
            "defend_busy": {"hurt_guilt": 0.80, "sighs": 0.20},
            "apologize_excessively": {"reassures_you": 0.40, "hurt_guilt": 0.30, "hint_again": 0.30},
            "change_subject": {"hurt_guilt": 0.60, "sighs": 0.40},
        },
    },
    outcomes={
        "connected": Outcome("connected", "Connected", {"neutral_observation": 1.0, "concern_wellbeing": 1.0, "indirect_request": 1.0, "mild_reproach": 0.9}, True),
        "withdrew": Outcome("withdrew", "Call ended coolly", {"neutral_observation": -0.6, "concern_wellbeing": -0.7, "indirect_request": -0.8, "mild_reproach": -0.8}),
        "timeout": Outcome("timeout", "Unresolved", {"neutral_observation": 0.2, "concern_wellbeing": 0.0, "indirect_request": -0.3, "mild_reproach": -0.3}),
    },
    roles={"clarify": {"ask_open"}, "assume": {"reply_literally"}, "boundary": {"defend_busy"},
           "repair": {"reassure_and_schedule", "ask_open"}, "exit": {"change_subject"}},
    withdraw_text=["I'll let you go. Bye, love.", "Okay. Talk soon."],
)

# =============================================================================
# 5. It's Fine
# =============================================================================
ITS_FINE = EnvSpec(
    id="its_fine",
    title="It's Fine",
    subtitle="You canceled last minute. They reply: “It's fine.”",
    domain="emotional",
    opening=("family", "emo_its_fine"),
    default_factors={"stakes": "high"},
    capabilities=["emotional_repair", "belief_revision"],
    horizon=4,
    responses={
        "accepts_plan": R("accepts_plan", ["Thursday works 🙂 thanks for saying that", "Ok, Thursday. Thank you, that means a lot."], -0.15, None, "repaired", "Accepts the repair"),
        "genuinely_ok": R("genuinely_ok", ["Honestly it's fine! I needed a night in anyway 😅", "No really, I'm happy to have a quiet one."], -0.05, "genuinely_fine", "repaired", "Genuinely fine"),
        "admits_disappointed": R("admits_disappointed", ["Yeah, I was a bit bummed, but thanks for saying that.", "I was looking forward to it, but I get it."], -0.05, "disappointed_understanding", "repaired", "Admits mild disappointment"),
        "admits_hurt": R("admits_hurt", ["Honestly it kind of hurt, I'd told everyone you were coming.", "It stung a bit, not going to lie."], 0.0, "hurt_face_saving", None, "Admits being hurt"),
        "vents_pattern": R("vents_pattern", ["It's just… this keeps happening.", "This is the third time. It's a lot."], 0.1, "frustrated_pattern", None, "Names the pattern"),
        "cold_ok": R("cold_ok", ["ok.", "Sure."], 0.15, None, None, "Cold"),
        "annoyed": R("annoyed", ["I didn't say I was mad. Forget it.", "Why would you assume that?"], 0.35, None, None, "Annoyed"),
        "comforts_you": R("comforts_you", ["Stop, it's not a big deal!", "Omg it's fine, don't spiral 😅"], 0.1, None, None, "Has to comfort you"),
        "closes_door": R("closes_door", ["Let's just talk later.", "I'm going to go."], 0.3, None, "distance", "Closes the door"),
        "moved_on": R("moved_on", ["👍", "np"], 0.0, None, "moved_on", "Conversation ends"),
    },
    table={
        "genuinely_fine": {
            "take_face_value": {"moved_on": 0.90, "genuinely_ok": 0.10},
            "acknowledge_and_repair": {"accepts_plan": 0.70, "genuinely_ok": 0.30},
            "check_in_gently": {"genuinely_ok": 0.90, "cold_ok": 0.10},
            "over_apologize": {"comforts_you": 0.80, "genuinely_ok": 0.20},
            "defensive_explain": {"genuinely_ok": 0.50, "cold_ok": 0.50},
            "assume_angry": {"annoyed": 0.80, "genuinely_ok": 0.20},
        },
        "disappointed_understanding": {
            "take_face_value": {"moved_on": 0.60, "cold_ok": 0.40},
            "acknowledge_and_repair": {"accepts_plan": 0.85, "admits_disappointed": 0.15},
            "check_in_gently": {"admits_disappointed": 0.70, "cold_ok": 0.30},
            "over_apologize": {"comforts_you": 0.70, "cold_ok": 0.30},
            "defensive_explain": {"cold_ok": 0.60, "annoyed": 0.40},
            "assume_angry": {"annoyed": 0.70, "admits_disappointed": 0.30},
        },
        "hurt_face_saving": {
            "take_face_value": {"cold_ok": 0.60, "moved_on": 0.40},
            "acknowledge_and_repair": {"accepts_plan": 0.60, "admits_hurt": 0.40},
            "check_in_gently": {"admits_hurt": 0.65, "cold_ok": 0.35},
            "over_apologize": {"comforts_you": 0.50, "cold_ok": 0.50},
            "defensive_explain": {"annoyed": 0.60, "closes_door": 0.40},
            "assume_angry": {"annoyed": 0.60, "admits_hurt": 0.40},
        },
        "frustrated_pattern": {
            "take_face_value": {"cold_ok": 0.50, "closes_door": 0.50},
            "acknowledge_and_repair": {"vents_pattern": 0.50, "accepts_plan": 0.50},
            "check_in_gently": {"vents_pattern": 0.70, "cold_ok": 0.30},
            "over_apologize": {"cold_ok": 0.60, "vents_pattern": 0.40},
            "defensive_explain": {"closes_door": 0.70, "annoyed": 0.30},
            "assume_angry": {"vents_pattern": 0.50, "annoyed": 0.50},
        },
    },
    outcomes={
        "repaired": Outcome("repaired", "Repaired", {"genuinely_fine": 0.8, "disappointed_understanding": 1.0, "hurt_face_saving": 1.0, "frustrated_pattern": 1.0}, True),
        "moved_on": Outcome("moved_on", "Moved on (unaddressed)", {"genuinely_fine": 0.9, "disappointed_understanding": 0.4, "hurt_face_saving": -0.3, "frustrated_pattern": -0.5}),
        "distance": Outcome("distance", "Distance", {"genuinely_fine": -0.5, "disappointed_understanding": -0.7, "hurt_face_saving": -0.8, "frustrated_pattern": -0.8}),
        "withdrew": Outcome("withdrew", "Friend withdrew", {"genuinely_fine": -0.6, "disappointed_understanding": -0.8, "hurt_face_saving": -0.8, "frustrated_pattern": -0.8}),
        "timeout": Outcome("timeout", "Unresolved", {"genuinely_fine": 0.1, "disappointed_understanding": -0.1, "hurt_face_saving": -0.3, "frustrated_pattern": -0.4}),
    },
    roles={"clarify": {"check_in_gently"}, "assume": {"take_face_value", "assume_angry"}, "boundary": {"assume_angry", "defensive_explain"},
           "repair": {"acknowledge_and_repair", "check_in_gently"}, "exit": {"take_face_value"}},
    withdraw_text=["I'm going to head to bed.", "Let's talk tomorrow."],
)

SPECS: dict[str, EnvSpec] = {s.id: s for s in (SOFT_NO, SAME_PAGE, GROUP_CHAT, FAMILY_SUBTEXT, ITS_FINE)}


def make(env_id: str, weights: dict | None = None) -> PragmaticEnv:
    return PragmaticEnv(SPECS[env_id], weights=weights)


def describe(spec: EnvSpec) -> dict:
    from pgym.environments.engine import COMPONENT_DOC, DEFAULT_WEIGHTS
    return {
        "id": spec.id, "title": spec.title, "subtitle": spec.subtitle, "domain": spec.domain,
        "capabilities": spec.capabilities, "flagship": spec.flagship, "horizon": spec.horizon,
        "default_factors": spec.default_factors, "opening": list(spec.opening),
        "roles": {k: sorted(v) for k, v in spec.roles.items()},
        "outcomes": {k: {"label": o.label, "value": o.value, "success": o.success} for k, o in spec.outcomes.items()},
        "responses": {k: {"note": r.note, "friction": r.friction, "reveals": r.reveals, "outcome": r.outcome} for k, r in spec.responses.items()},
        "table": spec.table,
        "reward_components": COMPONENT_DOC, "default_weights": DEFAULT_WEIGHTS,
    }
