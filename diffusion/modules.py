from copy import deepcopy
from typing import Optional

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    GemmaMLP,
    GemmaRMSNorm,
    repeat_kv,
)

from .configs import DiTConfig


class AdaLayerNormZero(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size, bias=True)
        self.norm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    
class AdaLayerNormOut(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.dit_hidden_size, config.dit_hidden_size * 2, bias=True)
        self.norm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class DiTSelfAttention(nn.Module):
    """
    Diffusion Transformer Self-Attention.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        if config.qk_norm:
            self.q_norm = GemmaRMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)
            self.k_norm = GemmaRMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)

        self.q_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_attention_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_key_value_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_key_value_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )

        if config.model_type == "DiT" and config.attention == "self":
            self.text_k_proj = nn.Linear(
                config.dit_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )
            self.text_v_proj = nn.Linear(
                config.dit_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )

        self.o_proj = nn.Linear(config.base_config.num_attention_heads * config.base_config.head_dim, config.dit_hidden_size, bias=config.base_config.attention_bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_embed: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.config.attention == "self":
            text_key_states = self.text_k_proj(text_hidden_states)
            text_value_states = self.text_v_proj(text_hidden_states)

        query_states = rearrange(query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        key_states = rearrange(key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
        value_states = rearrange(value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if self.config.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if not self.config.pos_embed == "ape":
            cos, sin = pos_embed
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.attention == "self":
            text_key_states = rearrange(text_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
            text_value_states = rearrange(text_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

            if self.config.qk_norm:
                text_key_states = self.k_norm(text_key_states)

            key_states = torch.cat([text_key_states, key_states], dim=2)
            value_states = torch.cat([text_value_states, value_states], dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        attn_output = self.o_proj(attn_output)
        return attn_output


class DiTCrossAttention(nn.Module):
    """
    Diffusion Transformer Cross-Attention.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        if config.qk_norm:
            self.q_norm = GemmaRMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)
            if config.model_type == "DiT":
                self.k_norm = GemmaRMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)

        self.q_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_attention_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.base_config.num_attention_heads * config.base_config.head_dim,
            config.dit_hidden_size,
            bias=config.base_config.attention_bias,
        )

        if config.model_type == "DiT":
            self.text_k_proj = nn.Linear(
                config.text_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )
            self.text_v_proj = nn.Linear(
                config.text_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )

    def forward(
        self,
        dit_hidden_states: torch.FloatTensor,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        key_states: Optional[torch.FloatTensor] = None,
        value_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        query_states = self.q_proj(dit_hidden_states)
        if self.config.model_type == "DiT":
            key_states = self.text_k_proj(text_hidden_states)
            value_states = self.text_v_proj(text_hidden_states)

        query_states = rearrange(query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        if self.config.model_type == "DiT":
            key_states = rearrange(key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
            value_states = rearrange(value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if self.config.qk_norm:
            query_states = self.q_norm(query_states)
            if self.config.model_type == "DiT":
                key_states = self.k_norm(key_states)
                
        if self.config.model_type == "DiT":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        attn_output = self.o_proj(attn_output)

        return attn_output


class DiTLayer(nn.Module):
    """
    Diffusion Transformer Block.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        if config.timestep_conditioning == "adaln-zero":
            self.input_layernorm = AdaLayerNormZero(config)
        else:
            if config.timestep_conditioning == "adaln-single":
                self.scale_shift_table = nn.Parameter(torch.randn(6 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)
            self.input_layernorm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

        self.self_attn = DiTSelfAttention(config)
        if config.attention == "cross":
            self.cross_attn = DiTCrossAttention(config)

        self.post_attention_layernorm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        if config.sandwich_norm:
            self.pre_feedforward_layernorm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
            self.post_feedforward_layernorm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
            if config.attention == "cross":
                self.post_cross_attention_layernorm = GemmaRMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        mlp_config = deepcopy(config.base_config)
        mlp_config.hidden_size = config.dit_hidden_size
        self.mlp = GemmaMLP(mlp_config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        pos_embed: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        if self.config.timestep_conditioning == "adaln-zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.input_layernorm(hidden_states, emb=temb)
            shift_msa = scale_msa = torch.zeros_like(temb)
        else:
            norm_hidden_states = self.input_layernorm(hidden_states)
            if self.config.timestep_conditioning == "adaln-single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table + temb).chunk(6, dim=1)
            else:
                shift_msa = scale_msa = shift_mlp = scale_mlp = torch.zeros_like(temb)
                gate_msa = gate_mlp = torch.ones_like(temb)

        # Attention.
        norm_hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]

        if self.config.attention == "self":
            attn_output = self.self_attn(norm_hidden_states, pos_embed, text_hidden_states, attention_mask)

            if self.config.sandwich_norm:
                attn_output = self.post_attention_layernorm(attn_output)

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        elif self.config.attention == "cross":
            attn_output = self.self_attn(norm_hidden_states, pos_embed)

            if self.config.sandwich_norm:
                attn_output = self.post_attention_layernorm(attn_output)

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

            cross_attn_output = self.cross_attn(hidden_states, text_hidden_states=text_hidden_states, attention_mask=attention_mask)

            if self.config.sandwich_norm:
                cross_attn_output = self.post_cross_attention_layernorm(cross_attn_output)

            hidden_states = hidden_states + cross_attn_output

        if self.config.sandwich_norm:
            norm_hidden_states = self.pre_feedforward_layernorm(hidden_states)
        else:
            norm_hidden_states = self.post_attention_layernorm(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        mlp_output = self.mlp(norm_hidden_states)

        if self.config.sandwich_norm:
            mlp_output = self.post_feedforward_layernorm(mlp_output)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * mlp_output

        return hidden_states