# ADR 0001: Versioned chunk retake cache

- Status: Accepted
- Date: 2026-09-07

## Context

The sampler already writes a bounded `last_run_replay` cache so an interrupted serial render can resume without repeating completed chunks. Users also need to inspect and selectively retake short chunks, edit the effective H3 prompt, preserve original audio by default, and restore the original result.

The current tensor checkpoints contain most execution state, but prompt and observation-image artifacts are not exposed through a stable per-chunk contract. Existing replay truncation also deletes the selected chunk and every later checkpoint, which is unsuitable for non-destructive isolated retakes.

## Decision

Evolve the existing replay cache into the source of truth for retakes rather than creating a second sampler cache.

1. Keep tensor payloads in `chunks/chunk_NNNN.pt` and JSON/media outside tensor files.
2. Version the manifest schema and publish per-chunk JSON sidecars containing physical range, source prompt hash, original/final H3 prompts, director state, observation-image paths, and active revision metadata.
3. Store observation JPEGs as the exact bytes sent to the selected visual director.
4. Treat original chunk artifacts as immutable. Retakes are append-only revisions selected by manifest pointers.
5. Keep UI selection and prompt overrides in a separate retake-plan node. The sampler consumes that plan through an optional input added only when execution support is implemented.
6. Default isolated retakes to preserving the original audio exactly. AV retakes and continuous suffix retakes are explicit modes.
7. Validate every cache path beneath the fixed replay root and use atomic writes.
8. Preserve existing node interfaces and ordinary sampling behavior during the archival phases.

## Consequences

- Future retake UI and execution code depend on a versioned manifest contract.
- The cache becomes an authoring artifact, not only a transient resume optimization.
- Disk usage increases because original tensors and retake revisions coexist.
- Independent visual retakes can preserve later chunks and original audio; independent AV retakes may still have audible boundaries.
- Continuous AV retakes must regenerate the suffix to maintain latent continuity.
- Old replay formats remain readable only through explicit format migration; missing metadata is reported rather than fabricated.

## Initial delivery stages

1. Archive Qwen observation JPEGs and per-chunk prompt sidecars.
2. Add a read-only visual retake director.
3. Add prompt overrides and retake-plan output.
4. Implement isolated video-only retakes with original audio preservation.
5. Add isolated AV and continuous AV modes.
6. Add revision activation and assembly.
