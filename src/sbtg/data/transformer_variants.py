import math
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class ZeroPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, position_ids):
        return torch.zeros((1,), device=position_ids.device)

def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_alibi_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

class CustomCAttnRoPE(nn.Module):
    def __init__(self, old_c_attn, config, rope_base: float = 10000.0):
        super().__init__()
        self.old_c_attn = old_c_attn
        self.enabled = True
        self.num_heads = config.num_attention_heads
        self.split_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.rope_base = float(rope_base)

        dim = self.head_dim
        # θ_k = base^(-2k/d_head) for k = 0, 1, ..., d_head/2 - 1.  Matches
        # Su et al. 2024 (RoFormer).  θ_0 = base^0 = 1 regardless of base;
        # changing base rescales the *lower* frequencies only.
        self.register_buffer(
            "inv_freq",
            1.0 / (self.rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
        )

    def forward(self, x, *args, **kwargs):
        qkv = self.old_c_attn(x, *args, **kwargs)
        
        if not getattr(self, "enabled", True):
            return qkv

        batch_size, seq_len, _ = qkv.shape
        q, k, v = qkv.split(self.split_size, dim=2)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        sin = emb.sin()[None, :,  None, :]
        cos = emb.cos()[None, :,  None, :]

        def rotate_half(x_):
            x1 = x_[..., : x_.shape[-1] // 2]
            x2 = x_[..., x_.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        q_rot = q_rot.view(batch_size, seq_len, self.split_size)
        k_rot = k_rot.view(batch_size, seq_len, self.split_size)
        
        return torch.cat([q_rot, k_rot, v], dim=2)

class ALiBiAttentionWrapper(nn.Module):
    def __init__(self, old_attn, config):
        super().__init__()
        self.old_attn = old_attn
        self.enabled = True
        self.num_heads = config.num_attention_heads
        self.register_buffer("slopes", torch.tensor(get_alibi_slopes(self.num_heads), dtype=torch.float32))

    def forward(self, hidden_states, *args, **kwargs):
        if not getattr(self, "enabled", True):
            return self.old_attn(hidden_states, *args, **kwargs)
            
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        attention_mask = kwargs.get("attention_mask", None)

        # ALiBi: bias[h, i, j] = -m_h * (i - j).  For causal i >= j this is <= 0,
        # making nearby (small i-j) keys less penalised — the recency prior.
        i_idx = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
        j_idx = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)  # (1, seq_len)
        distances = (i_idx - j_idx).unsqueeze(0).unsqueeze(0)          # (1, 1, seq_len, seq_len)
        alibi_bias = -distances * self.slopes.view(1, self.num_heads, 1, 1)  # (1, n_heads, seq_len, seq_len)

        if attention_mask is not None:
            attention_mask = attention_mask + alibi_bias
        else:
            attention_mask = alibi_bias

        kwargs["attention_mask"] = attention_mask
        return self.old_attn(hidden_states, *args, **kwargs)

def create_transformer_variant(
    config: GPT2Config,
    pe_type: str,
    rope_base: float = 10000.0,
) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(config)

    if pe_type == "absolute":
        pass  # Standard configuration uses absolute positional embeddings
    elif pe_type == "alibi":
        model.transformer.wpe = ZeroPositionalEmbedding()
        for block in model.transformer.h:
            block.attn = ALiBiAttentionWrapper(block.attn, config)

    elif pe_type == "rope":
        model.transformer.wpe = ZeroPositionalEmbedding()
        for block in model.transformer.h:
            block.attn.c_attn = CustomCAttnRoPE(
                block.attn.c_attn, config, rope_base=rope_base,
            )

    else:
        raise ValueError(f"Unknown pe_type: {pe_type}")

    return model
