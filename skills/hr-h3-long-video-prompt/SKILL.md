# HR H3 Long Video Prompt Skill

Runtime guidance for the `HR H3 Prompt Skill Compiler` node.

## Contract

Transform an ordinary story into a structured MiniMax H3 long-video plan before physical chunk sampling.

Every shot must define:

- a distinct camera setup;
- a visible start state inherited from the prior shot;
- atomic events with stable IDs and one owner shot;
- a visible end state usable by the next shot;
- completed actions and compositions that must not repeat;
- synchronized audio.

Compound sequences must be split into separate shots or atomic events. A later shot must never restart an event that already started or completed. Character-to-subject and picture mappings must remain unique and stable.

The model returns structured JSON. Program code validates it and deterministically compiles the six H3 fields:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

The skill uses the backend selected by the connected `HR Qwen Director Config`; it does not select or load another model.
