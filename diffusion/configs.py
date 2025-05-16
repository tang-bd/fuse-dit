from typing import List, Optional, Union

from transformers import (
    GemmaConfig,
    Gemma2Config,
    PretrainedConfig
)


class DiTConfig(PretrainedConfig):
    model_type = "DiT"

    def __init__(
        self,
        attention: str = "self",
        base_config: Optional[PretrainedConfig] = None,
        dit_hidden_size: int = None,
        dit_num_hidden_layers: int = 18,
        text_hidden_states_index: Optional[int] = -1,
        in_channels: int = 16,
        initial_layers: int = 0,
        out_channels: int = 16,
        patch_size: int = 2,
        pos_embed: str = "1d-rope",
        pos_embed_max_size: Optional[int] = 64,
        qk_norm: bool = False,
        sample_size: int = 32,
        sandwich_norm: bool = False,
        shared_attention_layers: Union[str, List[int]] = "all",
        text_hidden_size: Optional[int] = None,
        text_modulation_embeds_dim: Optional[int] = None,
        timestep_conditioning: str = "adaln-zero",
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(base_config, dict):
            if base_config["model_type"] == "gemma":
                base_config = GemmaConfig.from_dict(base_config)
            elif base_config["model_type"] == "gemma2":
                base_config = Gemma2Config.from_dict(base_config)
            else:
                raise ValueError(f"Model type {base_config['model_type']} not supported.")

        self.attention = attention
        self.base_config = base_config
        self.dit_hidden_size = dit_hidden_size if dit_hidden_size is not None else getattr(base_config, "hidden_size", 2048)
        self.dit_num_hidden_layers = dit_num_hidden_layers
        self.text_hidden_states_index = text_hidden_states_index
        self.in_channels = in_channels
        self.initial_layers = initial_layers
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.pos_embed_max_size = pos_embed_max_size
        self.qk_norm = qk_norm
        self.sample_size = sample_size
        self.sandwich_norm = sandwich_norm
        if isinstance(shared_attention_layers, str):
            self.shared_attention_layers = shared_attention_layers
        else:
            self.shared_attention_layers = list(shared_attention_layers)
        self.text_hidden_size = text_hidden_size
        self.text_modulation_embeds_dim = text_modulation_embeds_dim
        self.timestep_conditioning = timestep_conditioning


class FuseDiTConfig(DiTConfig):
    model_type = "FuseDiT"

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)