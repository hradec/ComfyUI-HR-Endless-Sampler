# Production Reasoning Plan V3

Status: design proposal for future implementation. This document does not
describe active runtime behavior.

## Problem

The current Gemma preproduction path makes one global production-bible request
containing every source shot, followed by one detailed timing request per
source shot. The global response can become large because it repeats shot
intent, environment, camera design, participating-character states, identity
bindings, and voice information for the entire production.

Simply reducing the global pass to identity extraction and leaving every shot
independent would lose useful whole-video reasoning. Gemma must see a compact
representation of all shots at least once to coordinate narrative pacing and
cross-shot state. At the same time, the existing frame-level beat reasoning is
valuable and must not be removed.

## Goals

- Bound each model response so long productions are less likely to truncate.
- Preserve one explicit whole-production pacing decision.
- Preserve the existing detailed per-shot `visual_beats`, `overlays`, dialogue
  segmentation, `continuity_slices`, and `light_change` reasoning.
- Keep sampler-owned facts immutable: source shot numbers and boundaries,
  frame ownership intervals, character-to-subject bindings, coverage IDs, and
  exact source dialogue.
- Make a validation failure regenerate only the smallest affected stage.
- Continue using Python-built JSON fill forms rather than asking Gemma to
  reconstruct schemas.

## Non-goals

- Do not let Gemma move source-shot cuts or change the total video duration.
- Do not replace detailed beat planning with a vague global outline.
- Do not split individual fields into separate model calls.
- Do not ask Gemma to invent additional shots, actions, dialogue, camera moves,
  characters, props, sounds, or locations.
- Do not change chunk directing, MTP containment, worker isolation, or the
  operation-local non-MTP retry as part of this redesign.

## Proposed hierarchy

```text
Complete source prompt and sampler-owned timeline
    |
    v
Phase 1: bounded fact extraction, one source shot at a time
    |
    v
Phase 2: one compact whole-production pacing and continuity pass
    |
    v
Phase 3: detailed frame-level planning, one source shot at a time
    |
    v
Validated immutable preproduction plan used by chunk directing
```

### Phase 1: per-shot fact extraction

Request one source shot at a time. Supply the complete authoritative prose for
that shot, the global character/subject bindings detected by Python, and any
global reference definitions needed to interpret its labels.

The response should extract facts without assigning frames:

```json
{
  "confidence": null,
  "analysis": null,
  "source_shot": 1,
  "shot_intent": null,
  "environment": null,
  "camera_and_cut": null,
  "successive_actions": [],
  "concurrent_events": [],
  "characters": [
    {
      "character_name": "Heman",
      "subject": "<Subject 1>",
      "opening_state": null,
      "closing_state": null
    }
  ]
}
```

Python must pre-populate `source_shot`, known character names, and subject
bindings. Gemma fills nulls and open event arrays. Dialogue and sound content
must remain byte-for-byte reconstructible from the source prompt.

Identity and speaker voice profiles may be established in a small global
identity form before these requests, or deterministically pre-populated when
the prompt already declares them. Do not combine every shot's prose back into
that identity response.

### Phase 2: compact global pacing pass

Give Gemma all validated Phase 1 summaries, exact source-shot durations, shot
order, and fixed cut boundaries. Do not resend every complete source-shot prose
block unless validation shows that a compact summary is insufficient.

This is the only stage responsible for reasoning jointly about the complete
video. It should decide broad pacing, cross-shot progression, and compatible
boundary states without allocating detailed frame ranges.

Suggested form:

```json
{
  "confidence": null,
  "analysis": null,
  "production_arc": null,
  "shots": [
    {
      "source_shot": 1,
      "duration_frames": 144,
      "pacing_role": null,
      "major_progression": [],
      "required_opening_state": null,
      "required_closing_state": null,
      "transition_to_next_shot": null
    }
  ]
}
```

Python must pre-populate shot numbers, durations, and order. The response may
describe when a development should occur in broad terms such as opening,
middle, or ending, but it must not change shot boundaries or fabricate precise
frame numbers.

The global validator must confirm:

- every source shot appears exactly once and in order;
- shot numbers and durations are unchanged;
- opening and closing states do not contradict validated source facts;
- adjacent transitions are compatible unless the source explicitly changes
  place, time, state, or subject;
- every progression item is traceable to a Phase 1 source fact;
- no new cut, action, dialogue, camera movement, character, prop, or sound was
  introduced.

### Phase 3: detailed per-shot beat planning

Retain the current one-request-per-source-shot detailed timing stage. Each
request receives:

- the full authoritative prose for that source shot;
- its validated Phase 1 facts;
- the compact global pacing map;
- its fixed duration and exact retained chunk-ownership intervals;
- the previous and next global boundary states when applicable;
- immutable identity bindings and speaker voice profiles;
- a Python-built JSON fill form.

The form must continue to include the existing detailed reasoning fields:

```json
{
  "confidence": null,
  "analysis": null,
  "source_shot": 1,
  "light_change": null,
  "visual_beats": [],
  "overlays": [],
  "continuity_slices": []
}
```

This phase still performs all current beat work:

- build a contiguous `visual_beats` timeline covering the complete shot;
- distinguish genuinely sequential actions from concurrent overlays;
- assign exact source-relative half-open frame intervals;
- preserve exact dialogue and split it naturally across retained ownership
  slices;
- preserve voice profiles and uninterrupted-speech requirements;
- produce character entry and expected-exit state for every continuity slice;
- set `light_change` from source intent;
- respect the global pacing role and broad progression without replacing
  evidence from the authoritative source shot.

The global pacing map guides beat placement. It does not become H3 prose and
does not replace `visual_beats` or `overlays`.

## Validation and correction ownership

Validate after every stage. A correction should receive the rejected response,
the precise validation errors, the original Python-built form, and only the
context needed for that stage.

- Phase 1 failure: retry only that source-shot extraction.
- Phase 2 failure: retry only the compact global pacing response.
- Phase 3 failure: retry only that source-shot beat plan.
- Chunk response failure: retain the existing chunk-local correction behavior.

Never silently repair model-authored creative facts in Python. Python may
calculate and enforce structural facts such as durations, intersections,
ordering, exact identifiers, dialogue reconstruction, and frame coverage.

## Runtime and cache considerations

The final validated artifact must include all three layers so replay can prove
that it matches the current prompt and physical plan. Increment the
preproduction/replay format when implementing V3.

A useful fingerprint should cover:

- original prompt;
- source-shot records and fixed boundaries;
- complete physical chunk plan, not a `debug_stop_chunk` subset;
- prompt mode and prompt-template contents;
- Phase 1 fact records;
- Phase 2 pacing map;
- Phase 3 detailed shot plans;
- character table and speaker voice profiles.

`debug_stop_chunk` must continue to limit only rendering. Every preproduction
stage must see the complete production horizon.

If the optional clean Gemma KV cache is retained, materialize it only after all
V3 stages validate. Do not weaken the disposable-worker boundary or the
operation-local non-MTP fallback.

## Implementation outline

1. Add explicit dataclasses and validators for per-shot fact records and the
   compact global pacing map.
2. Add Python fill-form builders and editable prompt sections for Phases 1 and
   2.
3. Move shot intent/environment/camera facts out of the present large global
   response and into Phase 1.
4. Extend the existing detailed shot request with the validated Phase 1 record
   and relevant Phase 2 constraints.
5. Assemble the final `GemmaShotTimingPlan` only after every stage validates.
6. Update transcript output so each request, raw response, correction, and
   validation warning remains inspectable.
7. Increment replay/cache formats and reject incompatible older V3-incomplete
   plans while preserving compatible completed H3 chunks where safe.

## Smallest meaningful checks

Leave focused runnable checks proving that:

- a long multi-shot prompt produces one Phase 1 request per shot, one global
  pacing request, and one Phase 3 request per shot;
- the global pacing request contains every shot summary and duration but not
  the complete verbose source prose;
- detailed shot plans still contain contiguous `visual_beats`, overlays,
  dialogue segments, continuity slices, and `light_change`;
- a Phase 2 validation failure does not regenerate Phase 1;
- a Phase 3 validation failure regenerates only its source shot;
- exact dialogue reconstructs once and in order across all segments;
- `debug_stop_chunk` does not shorten any planning stage's horizon;
- cache fingerprints change when any V3 layer or template changes.

## Decision summary

V3 should make requests smaller through hierarchy, not by removing reasoning.
Gemma first extracts bounded shot facts, then sees all compact shot summaries in
one global pacing pass, and finally performs the existing detailed beat
reasoning independently for each source shot. At least one compact pass must
see the entire production; otherwise the system cannot claim whole-video
pacing coordination.
