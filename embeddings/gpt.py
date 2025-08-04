from pydantic import BaseModel, PositiveInt
from typing import Literal
from torch import nn, Tensor

import torch
import torch.nn.functional as F
import math


class ModelConfig(BaseModel):
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    ctx_length: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    bias: bool = False
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    rms_eps: float = 1e-9
    device: str


class MHA(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.QKV = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.bias)
        self.O = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)

        mask = torch.triu(
            torch.ones(size=[1, 1, cfg.ctx_length, cfg.ctx_length]) * float("-inf"),
            diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.cfg.n_heads
        qkv: Tensor = self.QKV(x) # bsz, seq_len, embed_dim * 3
        q, k, v = qkv.split(embed_dim, -1) # (bsz, seq_len, embed_dim) * 3
        q = q.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2)
        k = k.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2)
        v = v.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2) 
        # attn
        attn_wts = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # bsz, n_heads, seq_len, seq_len
        attn_wts = attn_wts + self.mask[:, :, :seq_len, :seq_len]
        attn_wts = F.softmax(attn_wts, dim=-1)
        op = attn_wts @ v # bsz, n_heads, seq_len, head_dim
        op = op.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return self.O(op)
    
    
class FFN(nn.Module):
    
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self._ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self._ffn(x)


class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mha = MHA(cfg)
        self.ffn = FFN(cfg)
        self.ln_1 = nn.LayerNorm(cfg.embed_dim)
        self.ln_2 = nn.LayerNorm(cfg.embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.ctx_length, cfg.embed_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[-1], device=x.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.lm_head(self.ln(x))
        return x
    
    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)[:, -1]
            token = torch.argmax(logits, -1).unsqueeze(1)
            x = torch.cat([x, token], dim=1)
        return x