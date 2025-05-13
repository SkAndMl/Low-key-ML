from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
from copy import deepcopy
from typing import Tuple
import random
import time

from rich.table import Table
from rich import print as rprint

def print_results_table(results: dict):
    table = Table(title="Speculative Decoding Results", show_lines=True)

    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if isinstance(value, list):
            value = f"{value[:5]}..." if len(value) > 5 else str(value)
        elif isinstance(value, float):
            value = f"{value:.4f}"
        else:
            value = str(value)
        table.add_row(key, value)

    rprint(table)


@dataclass
class SpeculativeDecoder:

    target_id: str
    draft_id: str
    gamma: str
    device: str
    dtype = torch.float16

    def __post_init__(self) -> None:
        self.accepted_counts: list[int] = []
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_id, torch_dtype=self.dtype
        ).to(self.device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_id, torch_dtype=self.dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.draft_id)
    
    def _itertext(self, prompt: str, n: int):
      if hasattr(self.tokenizer, "apply_chat_template"):
          print(f"{self.draft_id}'s tokenizer has a chat template. Using it...")
          messages = [
              {"role": "system", "content": "Try to complete or answer the user's prompt"},
              {"role": "user", "content": prompt}
          ]
          prompt = self.tokenizer.apply_chat_template(
              messages,
              tokenize=False,
              add_generation_prompt=True,
              enable_thinking=False
          )
      input_tokens: torch.Tensor = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
      output_tokens: torch.Tensor = deepcopy(input_tokens).to(self.device)

      while output_tokens.shape[1] - input_tokens.shape[1] < n:
          step_tokens = self.step(output_tokens)
          chunk = self.tokenizer.decode(step_tokens[:, output_tokens.shape[1]:].detach().cpu().tolist()[0])
          yield chunk
          output_tokens = step_tokens

    def __call__(self, prompt: str, n: int, stream: bool=False) -> dict:
        self.accepted_counts = []
        return_dict = {}
        generated_text = ""
        start = time.time()
        for chunk in self._itertext(prompt, n):
            generated_text += chunk
            if stream:
              print(chunk, end="")
        end = time.time()
        print("")
        return_dict["speculative_decoding_time_taken"] = end - start
        return_dict["generated_text"] = generated_text
        return_dict["accepted_token_counts"] = self.accepted_counts
        return_dict["forward_passes_saved"] = sum(self.accepted_counts)
        return return_dict

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        draft_token_logits, draft_tokens = self._draft_step(tokens)
        target_token_logits: torch.Tensor = self.target_model(
            input_ids=torch.cat([tokens, draft_tokens], dim=-1).to(self.device),
            attention_mask=torch.ones(size=[1, tokens.shape[1]+self.gamma]).to(self.device)
        ).logits
        target_token_logits = target_token_logits[:, tokens.shape[1]-1:tokens.shape[1]+self.gamma-1]
        assert draft_token_logits.shape == target_token_logits.shape, f"{draft_token_logits.shape}, {target_token_logits.shape}"

        accepted_till_i = self.gamma
        for i in range(self.gamma):
            token = draft_tokens[0, i]
            q_prob = torch.softmax(draft_token_logits[0, i, :], dim=0)[token].item()
            p_prob = torch.softmax(target_token_logits[0, i, :], dim=0)[token].item()
            acc_prob = min(1.0, p_prob / q_prob)
            if random.random() > acc_prob:
                accepted_till_i = i
                break
        
        if accepted_till_i < self.gamma:
            accepted_tokens = draft_tokens[:, :accepted_till_i]
            p_probs = torch.softmax(target_token_logits[0, accepted_till_i, :], dim=0)
            q_probs = torch.softmax(draft_token_logits[0, accepted_till_i, :], dim=0)        
            correction = torch.clamp(p_probs - q_probs, min=0)
            if correction.sum() > 0:
                correction = correction / correction.sum()
                new_token = torch.multinomial(correction, num_samples=1).unsqueeze(0)
            else:
                new_token = torch.multinomial(p_probs, num_samples=1).unsqueeze(0)
        else:
            accepted_tokens = draft_tokens
            new_token_logits = self.target_model(
                input_ids=torch.cat([tokens, accepted_tokens], dim=1),
                attention_mask=torch.ones(size=[1, tokens.shape[1]+self.gamma])
            ).logits[0, -1, :]
            new_token_probs = torch.softmax(new_token_logits, dim=0)
            new_token = torch.multinomial(new_token_probs, num_samples=1).unsqueeze(0)

        # print(f"{accepted_tokens.shape[1]} out of {self.gamma}")
        self.accepted_counts.append(accepted_tokens.shape[1])
        return torch.cat([tokens, accepted_tokens, new_token], dim=1).to(self.device)
    
    def _draft_step(self, tokens: torch.Tensor) -> Tuple[torch.Tensor]:
        draft_token_logits = []
        _tokens = deepcopy(tokens).to(self.device)
        for _ in range(self.gamma):
            logits: torch.Tensor = self.draft_model(
                input_ids=_tokens,
                attention_mask=torch.ones_like(_tokens, device=self.device)
            ).logits
            draft_token_logits.append(logits[0, -1, :])
            _tokens = torch.cat([
                _tokens, torch.tensor([[logits[0, -1, :].argmax().item()]]).to(self.device)
            ], dim=-1).to(self.device)
        
        return torch.stack(draft_token_logits, dim=0).unsqueeze(0).to(self.device), _tokens[:, -self.gamma:]
    
    def greedy_decode(self, prompt: str, n: int) -> dict:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        start = time.time()
        output_ids = self.target_model.generate(
            input_ids,
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        end = time.time()
        results_dict = {
            "greedy_decoding_time_taken": end-start,
            "generated_text": self.tokenizer.decode(output_ids[0])
        }
        return results_dict
    

sd = SpeculativeDecoder(
    target_id="Qwen/Qwen3-1.7B",
    draft_id="Qwen/Qwen3-0.6B",
    # target_id="EleutherAI/pythia-410m",
    # draft_id="EleutherAI/pythia-70m",
    gamma=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)