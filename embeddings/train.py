import torch
import random
import os
import time
import math
import tiktoken

from torch.nn import functional as F
from .gpt import GPT, ModelConfig
from .llama import LLaMA, LlamaModelConfig
from typing import Tuple, Union
from torch.amp import autocast
from argparse import ArgumentParser

class DataLoader:

    def __init__(self, filepath, tokenizer, batch_size, ctx_size):
    
        with open(filepath, "r") as f:
            self.data = f.read().split("<|endoftext|>")
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.i = 0

    def _convert_data_to_tensors(self, data) -> None:
        tokens = []
        for i in range(len(data)):
            _tokens = self.tokenizer.encode(data[i])
            _tokens += self.tokenizer.encode("<|endoftext|>", allowed_special="all") * (self.ctx_size + 1 - len(_tokens))
            tokens.append(torch.tensor(_tokens[:self.ctx_size+1]))
        return tokens

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __iter__(self):
        self.i = 0
        random.shuffle(self.data)
        return self
            
    def __next__(self) -> Tuple[torch.Tensor]:
        if self.i + self.batch_size > len(self.data):
            raise StopIteration
        
        tokens = torch.stack(
            self._convert_data_to_tensors(self.data[self.i:self.i+self.batch_size]),
            dim=0
        )
        x, y = tokens[:, :-1], tokens[:, 1:]
        self.i += self.batch_size
        return x, y
    


def train(cfg, lm, tokenizer, bsz: int, model_type: str):
    lm.to(cfg.device)
    train_dl = DataLoader("data/tinystories/train.txt", tokenizer, bsz, cfg.ctx_length)
    val_dl = DataLoader("data/tinystories/validation.txt", tokenizer, bsz, cfg.ctx_length)
    
    optimizer = torch.optim.AdamW(lm.parameters(), lr=3e-4)
    use_amp = cfg.device == "mps"

    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]

    @torch.inference_mode()
    def evaluate() -> Tuple[float]:
        total_loss, total_tokens = 0, 0
        for x, y in val_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            with autocast(device_type=cfg.device, dtype=torch.float16 if use_amp else torch.float32):
                logits: torch.Tensor = lm(x) # bsz, seq_len, vocab_size
                logits = logits.view(-1, logits.size(-1))
                loss: torch.Tensor = F.cross_entropy(logits, y.view(-1), ignore_index=pad_token_id, reduction="none").reshape_as(y) # bsz, seq_len

            mask: torch.Tensor = (y != pad_token_id)
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()
        
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    log_file = open(f"log_{model_type}.txt", "w")
    log_step = 50
    evaluate_step = 500
    stop_step = 1000
    
    start_time = time.time()
    for step, (x, y) in enumerate(train_dl):
        if step >= stop_step:
            to_log = f"training done for vocab size: {cfg.vocab_size}; took: {time.time() - start_time:.4f} seconds"
            print(to_log)
            break
        x, y = x.to(cfg.device), y.to(cfg.device) # bsz, seq_len
        with autocast(device_type=cfg.device, dtype=torch.float16 if use_amp else torch.float32):
            logits: torch.Tensor = lm(x) # bsz, seq_len, vocab_size
            logits = logits.view(-1, logits.size(-1))
            loss: torch.Tensor = F.cross_entropy(logits, y.view(-1), ignore_index=pad_token_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_step == 0:
            to_log = f"Step {step+1} | loss = {loss.item():.4f}"
            log_file.write(to_log + "\n")
            log_file.flush()
            print(to_log)

        if (step + 1) % evaluate_step == 0 or step == len(train_dl) - 1 or step == stop_step - 1:
            val_loss, ppl = evaluate()
            to_log = f"Step {step+1} | val loss = {val_loss:.4f} | ppl: {ppl:.4f}"
            log_file.write(to_log + "\n")
            log_file.flush()
            print(to_log)

            with torch.no_grad():
                x = torch.tensor([tokenizer.encode("I am going to")]).to(cfg.device)
                out = lm.generate(x, max_new_tokens=30)
                decoded_str = tokenizer.decode(out.detach().tolist()[0])
                to_log = f"Generation at step {step+1}: {decoded_str}"
                log_file.write(to_log + "\n")
                log_file.flush()
                print(to_log)

    log_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str)
    
    args = parser.parse_args()
    model_type = args.model_type
    tokenizer = tiktoken.get_encoding("gpt2")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bsz = 16
    
    assert model_type in ["gpt", "llama"]
    
    if model_type == "gpt":
        cfg = ModelConfig(
            vocab_size=tokenizer.n_vocab,
            embed_dim=256,
            n_heads=4,
            n_layers=4,
            ctx_length=256,
            device=device
        )
    else:
        cfg = LlamaModelConfig(
            vocab_size=tokenizer.n_vocab,
            embed_dim=256,
            n_heads=4,
            n_layers=4,
            ctx_length=256,
            theta_base=10000,
            device=device,
            hidden_dim=384
        )

    lm = LLaMA(cfg) if model_type=="llama" else GPT(cfg)

    print(f"running on {device}")
    train(cfg, lm, tokenizer, bsz, model_type)     