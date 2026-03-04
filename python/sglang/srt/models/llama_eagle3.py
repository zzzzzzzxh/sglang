"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import copy
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # For tensor saving
        self.save_tensors = os.environ.get("SGLANG_SAVE_TENSORS", "0") == "1"
        self.save_dir = os.environ.get("SGLANG_SAVE_DIR", "/tmp/sglang_tensors")
        self.forward_count = 0

    def _save_input_tensors(
        self, input_ids, positions, forward_batch, embeds, hidden_states_before_fc
    ):
        """Save input tensors for comparison"""
        inputs_dir = os.path.join(
            self.save_dir, f"forward_{self.forward_count}", "inputs"
        )
        os.makedirs(inputs_dir, exist_ok=True)

        # Save input_ids
        if input_ids is not None:
            torch.save(input_ids.cpu(), os.path.join(inputs_dir, "input_ids.pt"))
            print(
                f"[SGLANG] Saved input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}"
            )

        # Save positions
        if positions is not None:
            torch.save(positions.cpu(), os.path.join(inputs_dir, "positions.pt"))
            print(
                f"[SGLANG] Saved positions: shape={positions.shape}, dtype={positions.dtype}"
            )

        # Save embeds (after embedding lookup)
        if embeds is not None:
            torch.save(embeds.cpu(), os.path.join(inputs_dir, "embeds.pt"))
            print(f"[SGLANG] Saved embeds: shape={embeds.shape}, dtype={embeds.dtype}")

        # Save hidden_states from spec_info (before fc projection)
        if hidden_states_before_fc is not None:
            torch.save(
                hidden_states_before_fc.cpu(),
                os.path.join(inputs_dir, "hidden_states_before_fc.pt"),
            )
            print(
                f"[SGLANG] Saved hidden_states_before_fc: shape={hidden_states_before_fc.shape}, dtype={hidden_states_before_fc.dtype}"
            )

        # Save forward_batch info
        if forward_batch is not None:
            batch_info = {}
            if (
                hasattr(forward_batch, "seq_lens")
                and forward_batch.seq_lens is not None
            ):
                batch_info["seq_lens"] = forward_batch.seq_lens.cpu()
            if (
                hasattr(forward_batch, "block_tables")
                and forward_batch.block_tables is not None
            ):
                batch_info["block_tables"] = forward_batch.block_tables.cpu()

            torch.save(batch_info, os.path.join(inputs_dir, "forward_batch_info.pt"))
            print(f"[SGLANG] Saved forward_batch info")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states

        # Save input tensors before any computation (only first forward pass)
        if self.save_tensors and self.forward_count == 0:
            hidden_states_before_fc = hidden_states
            if hidden_states.shape[-1] != embeds.shape[-1]:
                hidden_states = self.fc(hidden_states)
            self._save_input_tensors(
                input_ids, positions, forward_batch, embeds, hidden_states_before_fc
            )

            # Save hidden_states after fc projection
            inputs_dir = os.path.join(
                self.save_dir, f"forward_{self.forward_count}", "inputs"
            )
            torch.save(
                hidden_states.cpu(),
                os.path.join(inputs_dir, "hidden_states_after_fc.pt"),
            )
            print(
                f"[SGLANG] Saved hidden_states_after_fc: shape={hidden_states.shape}, dtype={hidden_states.dtype}"
            )
        else:
            if hidden_states.shape[-1] != embeds.shape[-1]:
                hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # Save output tensors (only first forward pass)
        if self.save_tensors and self.forward_count == 0:
            outputs_dir = os.path.join(
                self.save_dir, f"forward_{self.forward_count}", "outputs"
            )
            os.makedirs(outputs_dir, exist_ok=True)

            torch.save(
                hidden_states_to_logits.cpu(),
                os.path.join(outputs_dir, "hidden_states_to_logits.pt"),
            )
            print(
                f"[SGLANG] Saved hidden_states_to_logits: shape={hidden_states_to_logits.shape}, dtype={hidden_states_to_logits.dtype}"
            )

            if isinstance(hidden_states_to_aux, list):
                aux_to_save = torch.cat(hidden_states_to_aux, dim=-1)
            else:
                aux_to_save = hidden_states_to_aux
            torch.save(
                aux_to_save.cpu(), os.path.join(outputs_dir, "aux_hidden_states.pt")
            )
            print(
                f"[SGLANG] Saved aux_hidden_states: shape={aux_to_save.shape}, dtype={aux_to_save.dtype}"
            )

            self.forward_count += 1
            print(
                f"[SGLANG] Completed tensor saving for forward pass {self.forward_count}"
            )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

        # For tensor saving/comparison experiment
        self.save_tensors = os.environ.get("SGLANG_SAVE_TENSORS", "0") == "1"
        self.save_dir = os.environ.get("SGLANG_SAVE_DIR", "/tmp/sglang_tensors")
        self.forward_count = 0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        # Save model weights if enabled
        if self.save_tensors:
            self._save_model_weights()

    def _save_model_weights(self):
        """Save all model weights to disk for comparison"""
        os.makedirs(self.save_dir, exist_ok=True)
        weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Save all model parameters
        for name, param in self.named_parameters():
            if param is not None:
                save_path = os.path.join(weights_dir, f"{name.replace('.', '_')}.pt")
                torch.save(param.data.cpu(), save_path)
                print(f"[SGLANG] Saved weight: {name} -> {save_path}")

        # Save config
        config_path = os.path.join(self.save_dir, "config.pt")
        torch.save(
            {
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "num_hidden_layers": self.config.num_hidden_layers,
                "draft_vocab_size": getattr(self.config, "draft_vocab_size", None),
            },
            config_path,
        )
        print(f"[SGLANG] Saved config to {config_path}")

    def _save_input_tensors(self, input_ids, positions, forward_batch):
        """Save input tensors for a single forward pass"""
        inputs_dir = os.path.join(
            self.save_dir, f"forward_{self.forward_count}", "inputs"
        )
        os.makedirs(inputs_dir, exist_ok=True)

        # Save input_ids
        if input_ids is not None:
            torch.save(input_ids.cpu(), os.path.join(inputs_dir, "input_ids.pt"))
            print(f"[SGLANG] Saved input_ids: shape={input_ids.shape}")

        # Save positions
        if positions is not None:
            torch.save(positions.cpu(), os.path.join(inputs_dir, "positions.pt"))
            print(f"[SGLANG] Saved positions: shape={positions.shape}")

        # Save hidden_states from spec_info
        if (
            forward_batch is not None
            and hasattr(forward_batch, "spec_info")
            and forward_batch.spec_info is not None
        ):
            hidden_states = forward_batch.spec_info.hidden_states
            if hidden_states is not None:
                torch.save(
                    hidden_states.cpu(), os.path.join(inputs_dir, "hidden_states.pt")
                )
                print(f"[SGLANG] Saved hidden_states: shape={hidden_states.shape}")

        # Save other forward_batch info
        if forward_batch is not None:
            batch_info = {}
            if hasattr(forward_batch, "seq_lens"):
                batch_info["seq_lens"] = (
                    forward_batch.seq_lens.cpu()
                    if forward_batch.seq_lens is not None
                    else None
                )
            if hasattr(forward_batch, "block_tables"):
                batch_info["block_tables"] = (
                    forward_batch.block_tables.cpu()
                    if forward_batch.block_tables is not None
                    else None
                )
            if hasattr(forward_batch, "positions"):
                batch_info["positions"] = (
                    forward_batch.positions.cpu()
                    if forward_batch.positions is not None
                    else None
                )

            torch.save(batch_info, os.path.join(inputs_dir, "forward_batch_info.pt"))
            print(f"[SGLANG] Saved forward_batch info")

    def _save_output_tensors(self, hidden_states, aux_hidden_states):
        """Save output tensors from forward pass"""
        outputs_dir = os.path.join(
            self.save_dir, f"forward_{self.forward_count}", "outputs"
        )
        os.makedirs(outputs_dir, exist_ok=True)

        # Save hidden_states (logits input)
        if hidden_states is not None:
            torch.save(
                hidden_states.cpu(), os.path.join(outputs_dir, "hidden_states.pt")
            )
            print(f"[SGLANG] Saved output hidden_states: shape={hidden_states.shape}")

        # Save aux_hidden_states
        if aux_hidden_states is not None:
            if isinstance(aux_hidden_states, list):
                aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
            torch.save(
                aux_hidden_states.cpu(),
                os.path.join(outputs_dir, "aux_hidden_states.pt"),
            )
            print(f"[SGLANG] Saved aux_hidden_states: shape={aux_hidden_states.shape}")

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
