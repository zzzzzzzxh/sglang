# EAGLE3 Standalone Model

This directory contains tools to run EAGLE3 draft models as standalone models without speculative decoding.

## Purpose

EAGLE3 is a speculative decoding technique that uses a small draft model to predict tokens,
which are then verified by a larger target model. However, for benchmarking and comparison
purposes, you may want to run the EAGLE3 draft model independently.

This standalone version allows you to:
1. Run EAGLE3 draft model without requiring a base/target model
2. Compare performance across different backends (CUDA vs MindSpore)
3. Profile and optimize the draft model independently

## Key Changes

The standalone model modifies the forward pass to:
- Generate hidden states from input embeddings (instead of relying on target model)
- Triple the hidden size and pass through the FC layer internally
- Remove dependency on speculative decoding infrastructure

## Usage

### SGLang Native (CUDA/NPU)

```bash
# Run EAGLE3 draft model as standalone
python examples/run_eagle3_standalone.py \
    --model-path /path/to/Llama-3-8B-Eagle3 \
    --device cuda \
    --tp-size 1 \
    --port 24678
```

### SGLang MindSpore (Ascend NPU)

```bash
# Run EAGLE3 draft model on MindSpore backend
python examples/run_eagle3_standalone.py \
    --model-path /path/to/Llama-3-8B-Eagle3 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --port 24678
```

## Benchmarking Comparison

To compare the two backends:

```bash
# Terminal 1: SGLang native
python examples/run_eagle3_standalone.py \
    --model-path /path/to/Llama-3-8B-Eagle3 \
    --device cuda \
    --port 24678

# Terminal 2: SGLang MindSpore
python examples/run_eagle3_standalone.py \
    --model-path /path/to/Llama-3-8B-Eagle3 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --port 24679
```

Then use your benchmarking tools (e.g., `sgl_bench`) to send requests to both ports.

## Technical Details

### Model Architecture

The EAGLE3 draft model:
- Has only 1 transformer layer
- Takes concatenated [embed, hidden_states] as input
- Uses special FC layer for hidden state transformation
- Has modified QKV projection (2x hidden size)

In standalone mode:
- Input embeddings are tripled and passed through FC layer
- This generates the hidden_states that would normally come from target model
- The model then processes [embed, hidden_states] as usual

### Model Registration

The standalone models are registered as:
- `LForCausalLMEagle3Standalone` for SGLang native
- `LForCausalLMEagle3Standalone` for SGLang MindSpore

These can be loaded automatically by SGLang when you provide an EAGLE3 model path.

## Limitations

- This is a draft model, so output quality is lower than the full model
- Hidden state transformation is simplified (may not match exact speculative decoding behavior)
- For production use, enable speculative decoding with the full target model
