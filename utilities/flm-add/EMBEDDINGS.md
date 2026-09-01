# Summary
The flm-add tool is now adapted for embedding models with full NPU offload support. Here's what was implemented:

## Embedding detection	
 - Reads README.md frontmatter (pipeline_tag: sentence-similarity, tags: [sentence-similarity, text-embeddings-inference], base_model)
 - Base model resolution	Fetches weights/tokenizer from base_model (e.g., google/embeddinggemma-300M)
 - Manifest generation	Builds weights_manifest.json with relative paths + tensor offsets from .safetensors
NPU artifacts	Installs 12 xclbin/insts pairs (6 shapes × M=512/2048) to npu_matmul_f32/
 - Tag format	embed-<family>:<size> → embed-gemma:300m

## Verified Results
 - E1 Response Structure: PASS
 - E2 Repeatability: PASS (cosine 1.000000)
 - E3 Batch & Index Integrity: PASS
 - E4 Dimensionality: PASS
 - E5 Semantic Ordering: PASS
 - E6 Cross-Path Consistency: PASS (cosine 1.000000)
 - E7 Batch Reference Consistency: PASS (cosine 1.000000)
 - E8 Reference Agreement: PASS (cosine 0.999993, threshold 0.999)

## Hugging Face Upload Checklist
 - config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json, tokenizer.model
 - model.safetensors (FP32 weights)
 - weights_manifest.json (generated at install with relative paths + offsets)
 - npu_matmul_f32/ (12 xclbin/insts pairs for 6 shapes × 2 pads)
 - config.json with proper details.family: embed-gemma

