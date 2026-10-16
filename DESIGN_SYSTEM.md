# Pragmatic Gym — Design System

A Swiss/International-style visual language for an interactive research tool: white paper, one ink, one hot accent, tight grotesk type, square corners, and dots as the data language.

## 1. Tokens

Defined in `apps/web/src/app/globals.css` (Tailwind v4 `@theme` + shadcn CSS variables):

```css
--pg-paper:  #FFFFFF;   /* page */
--pg-ink:    #111111;   /* primary text, rules, primary buttons */
--pg-ink-2:  #4A4A4A;   /* secondary text, links */
--pg-ink-3:  #8A8A8A;   /* tertiary / captions */
--pg-line:   #E7E7E5;   /* hairlines */
--pg-wash:   #F4F4F2;   /* incoming message bubbles, inset panels */
--pg-accent: #FF3B00;   /* the one accent: model output, live state, focus */
--pg-accent-wash: #FFE7DE; /* 12% accent for fills */
--radius: 0;            /* every shadcn component is square */
```

**Type scale.** Inter Tight; see the decisions below.

| Token | Size / line-height / tracking | Use |
|---|---|---|
| `display` | 88 / 0.88 / −0.035em, 500 | landing headline |
| `title` | 56 / 0.95 / −0.03em, 500 | page titles |
| `lead` | 24 / 1.3 / −0.02em, 400 | page standfirst |
| `h3` | 20 / 1.2 / −0.015em, 700 | section heads |
| `body` | 15 / 1.45 / −0.005em, 400 | UI text |
| `small` | 13 / 1.35, 500 | labels, metadata |
| `num` | tabular figures | all numbers |

**Spacing.** There's a 4px base. Page gutters are `clamp(20px, 7.8vw, 112px)`. Sections are separated by space and hairlines, not by boxes.

## 2. Implementation decisions

1. **Typeface.** **Inter Tight** (OFL, Google Fonts), a freely licensed neo-grotesk with tight default spacing. Negative tracking is applied on large sizes: −0.03em on display sizes, −0.02em on leads.
2. **The dot matrix is the data language.** Dots carry information, not decoration: every probability distribution is drawn as a row of dots.
   - Filled `#FF3B00` dots are the model.
   - Hollow ink rings are the reference.
   - This is the only place the accent is used heavily, keeping a "one hot accent" discipline.
   - The landing figure is a dot drawing of two speech bubbles.
3. **The heavy rule.** A 3px ink rule sits under the header and above each page title. Hairlines (`--pg-line`) separate everything else.
4. **No cards, no radius, no shadows.** shadcn defaults (rounded, shadowed, gray borders) are overridden globally: `--radius: 0`, and inputs and buttons get 1.5px ink borders. Primary buttons are solid ink with white text. The accent is reserved for live and model state, never for decoration.
5. **Conversation first.** Messages are square-cornered blocks. Incoming messages sit on `--pg-wash`; "You" messages are ink with white text. The research panel is a separate column with smaller, denser type, so the dialogue stays the most legible thing on screen.
6. **Charts (Recharts).** Ink axes without gridlines, or hairline gridlines at most. Series are ink for the reference or hypothesis and accent for the model. Uncertainty is shown as thin whiskers. There are no gradients, fills or legend boxes, and every value is labeled with tabular figures.
7. **Provenance badges** sit next to every model number:
   - a filled accent dot for real inference;
   - a hollow ring for a replay of a recorded inference;
   - a striped gray square for a fixture or heuristic, with the word "FIXTURE".
8. **Motion.** Motion is minimal: 150ms opacity and color transitions, and dot fills that animate to new values when results arrive.

## 3. Screenshots

App screenshots at 1440 × 900 are in `docs/design/app/` (`make screenshots` regenerates them).
