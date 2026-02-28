#!/usr/bin/env python3
"""
Run EAGLE3 draft model as standalone model without speculative decoding.

This allows direct comparison of EAGLE3 draft model across different
backends (SGLang native vs MindSpore) without overhead of speculative decoding.
"""

import os
import sglang as sgl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to EAGLE3 draft model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or npu)")
    parser.add_argument("--model-impl", type=str, default=None, help="Model implementation (cuda or mindspore)")
    parser.add_argument("--attention-backend", type=str, default=None, help="Attention backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=24678, help="Server port")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable CUDA graph (use for standalone mode)")
    args = parser.parse_args()

    engine_args = {
        "model_path": args.model_path,
        "tp_size": args.tp_size,
        "port": args.port,
        "mem_fraction_static": 0.8,
    }

    if args.device:
        engine_args["device"] = args.device
    if args.model_impl:
        engine_args["model_impl"] = args.model_impl
    if args.attention_backend:
        engine_args["attention_backend"] = args.attention_backend
    if args.disable_cuda_graph:
        engine_args["disable_cuda_graph"] = True

    # Set environment variable to enable standalone mode
    os.environ["SGLANG_EAGLE3_STANDALONE"] = "1"

    print("Starting EAGLE3 standalone model server...")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Model implementation: {args.model_impl}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Standalone mode: ENABLED")
    print(f"Disable CUDA graph: {args.disable_cuda_graph}")

    llm = sgl.Engine(**engine_args)

    prompts = [
        "what is speculative decoding?",
    ]

    # Use max_new_tokens instead of max_tokens for SGLang
    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 100}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
