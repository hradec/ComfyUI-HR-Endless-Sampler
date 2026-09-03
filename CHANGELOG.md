# Changelog - mickeylan fork

## [Unreleased] - 2026-09-03

### Added
- **Multi-director support**: Added Qwen3.5, Qwen3.6, and Qwen3.8 as alternative director backends
- **12GB VRAM support**: Qwen3.6/3.8 27B MoE models with UD-IQ2-mtp quantization work on 12GB GPUs
- **Qwen3.6/3.8 features**:
  - Embedded MTP (speculative decoding) support
  - MoE CPU offload (`cpu_moe` / `n_cpu_moe` options)
  - Qwen3.8 reasoning effort control (`xhigh` / `medium` / `low`)
  - Native chat template support
- **Chinese language support**: Native Chinese prompt understanding with English H3 output
- **Local model discovery**: Recursive GGUF scanning in `models/llama_cpp/` and `models/LLM/GGUF/`
- **Path safety**: Strict validation of local model paths
- **Replay fingerprinting**: Director/model changes trigger replay cache invalidation

### Changed
- Updated README with fork-specific documentation
- Updated dependency tracking in `dependency.md`

### Technical Details

#### Qwen Model Specifications

| Model | Architecture | Context | MTP | VRAM (Q4) |
|-------|-------------|---------|-----|-----------|
| Qwen3.5 | 9B Dense | 65,536 | ❌ | ~4-5GB |
| Qwen3.6 | 27B MoE | 32,768 | ✅ | ~5-6GB |
| Qwen3.8 | 27B MoE | 32,768 | ✅ | ~5-6GB |

#### Why Qwen3.6/3.8 over Gemma 4 on 12GB?

- Gemma 4 12B Q4 requires ~6-8GB alone, plus llama.cpp CUDA context overhead
- On a 12GB card, Gemma simply **cannot be loaded** even with strict serialization
- Qwen3.6/3.8 27B MoE uses only ~3-4B active parameters per forward pass
- UD-IQ2-mtp quantization compresses the 27B model to ~8-9GB
- Speed is comparable to 9B models thanks to MoE architecture
