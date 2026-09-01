---
base_model:
- google/embeddinggemma-300M
base_model_relation: quantized
quantized_by: Atomic-Germ
---
# test-embed-model

**FastFlowLM NPU2 Embedding model** converted from [`google/embeddinggemma-300M`](https://huggingface.co/google/embeddinggemma-300M) for AMD XDNA NPU inference.

This repository contains an **NPU2-optimized embedding model** for FastFlowLM. It uses the open embedding engine with BF16→FP32 matmul kernels on the NPU2, validated against the official oracle (cosine ≥ 0.999).

| Item | Value |
|------|-------|
| Source model | [`google/embeddinggemma-300M`](https://huggingface.co/google/embeddinggemma-300M) |
| Modality | embeddings |
| Precision | BF16 inputs, FP32 output (bit-exact with FP32 CPU) |
| NPU topology | 4 rows × 4 cols (16 cores) |
| Base model | [`google/embeddinggemma-300M`](https://huggingface.co/google/embeddinggemma-300M) |

## Install and run

```bash
uv tool install flm-add
flm-add <your-hf-repo> --family embed-gemma --xclbin-from <official-tag>
FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json" FLM_XCLBIN_PATH="$HOME/.config/flm" flm serve embed-gemma:300m
```

## Files

| File | Description |
|------|-------------|
| `config.json` | FLM runtime configuration |
| `model.safetensors` | FP32 embedding weights |
| `weights_manifest.json` | Tensor offsets for streaming |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special token mapping |
| `tokenizer.model` | SentencePiece model (if present) |
| `chat_template.jinja` | Chat template (if applicable) |
| `weights_manifest.json` | Tensor offsets for streaming DMA |
| `npu_matmul_f32/` | NPU2 BF16→FP32 matmul kernels (xclbin + insts) |
