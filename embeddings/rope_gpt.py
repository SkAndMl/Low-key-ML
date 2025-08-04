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
    theta_base: PositiveInt
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    rms_eps: float = 1e-9
    hidden_dim: int


def get_freqs(cfg: ModelConfig):
    head_dim = cfg.embed_dim // cfg.n_heads
    _thetas = torch.tensor([1 / (cfg.theta_base * (2*i/head_dim)) for i in range(head_dim//2)]) # dim/2
    _pos = torch.tensor([i for i in range(cfg.ctx_length)])
    thetas = torch.outer(_pos, _thetas) # ctx_length, dim / 2
    real = torch.cos(thetas)
    img = torch.sin(thetas)
    freqs = torch.complex(real, img) # ctx_length, dim / 2
    freqs = freqs.unsqueeze(0).unsqueeze(0).to(cfg.device)
    return freqs

class MHA(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.QKV = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=cfg.bias)
        self.O = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias)
        self.freqs = get_freqs(cfg)

        mask = torch.triu(
            torch.ones(size=[1, 1, cfg.ctx_length, cfg.ctx_length]) * float("-inf"),
            diagonal=0
        )
        self.register_buffer("mask", mask)

    def _rot_emb(self, tensor: Tensor) -> Tensor:
        # bsz, n_heads, seq_len, head_dim
        bsz, n_heads, seq_len, head_dim = tensor.shape
        _tensor = torch.view_as_complex(tensor.view(size=(bsz, n_heads, seq_len, head_dim // 2, 2))) # bsz, n_heads, seq_len, head_dim / 2
        # rotates the 'complex' tensor by their corresponding angles
        _tensor = _tensor * self.freqs[:, :, :seq_len, :] # bsz, n_heads, seq_len, head_dim / 2
        _tensor = torch.view_as_real(_tensor) # bsz, n_heads, seq_len, head_dim, 2
        return _tensor.view(bsz, n_heads, seq_len, head_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.cfg.n_heads
        qkv: Tensor = self.QKV(x) # bsz, seq_len, embed_dim * 3
        q, k, v = qkv.split(embed_dim, -1) # (bsz, seq_len, embed_dim) * 3
        q = q.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2)
        k = k.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2)
        v = v.view(size=(bsz, seq_len, self.cfg.n_heads, head_dim)).transpose(1, 2) 
        # apply rope
        q, k = self._rot_emb(q), self._rot_emb(k)
        # attn
        attn_wts = F.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1
        ) # bsz, n_heads, seq_len, seq_len
        attn_wts += self.mask[:, :, :seq_len, :seq_len]
        op = attn_wts @ v # bsz, n_heads, seq_len, head_dim
        op = op.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return self.O(op)
    

class RMSNorm(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.eps = cfg.rms_eps
        self.weight = nn.Parameter(torch.ones(1, 1, cfg.embed_dim))
    
    def forward(self, x: Tensor) -> Tensor:
        denom = torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        denom = denom.sqrt()
        return x * self.weight / denom
    

class FFN(nn.Module):
    
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_1 = nn.Linear(cfg.embed_dim, 2 * cfg.hidden_dim, bias=cfg.bias)
        self.layer_2 = nn.Linear(cfg.hidden_dim, cfg.embed_dim, bias=cfg.bias)
    
    def _swish(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.layer_1(x).split(self.cfg.hidden_dim, -1)
        x = self._swish(x1) * x2
        return self.layer_2(x)



class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mha = MHA(cfg)
        self.ffn = FFN(cfg)
        self.ln_1 = RMSNorm(cfg)
        self.ln_2 = RMSNorm(cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=cfg.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len
        x = self.tok_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.lm_head(self.ln(x))
        return x