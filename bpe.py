from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
import regex as re
import os

class BPE:

    reg_pat = re.compile(
        r"""'s|'ve|'d|'ll|'t|'re|'m| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        flags=re.IGNORECASE
    )  

    def __init__(self, tokenizer_dir: str) -> None:
        self.tokenizer_dir = tokenizer_dir
        merges_path = os.path.join(tokenizer_dir, "merges.json")
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")

        if os.path.exists(merges_path) and os.path.exists(vocab_path):
            with open(merges_path, "r") as f:
                merges = json.load(f)
            self.merges = {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in merges.items()}

            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            self.vocab = {int(k): bytes(v) for k, v in vocab.items()}
        else:
            self.merges = {}
            self.vocab = {}


    def encode(self, text: str) -> List[int]:
        words = re.findall(self.reg_pat, text)
        token_seqs = [list(w.encode("utf-8")) for w in words]
        while True:
            freq = BPE.get_stats(token_seqs)
            if not freq:
                break
            pair = min(freq, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            token_seqs = BPE.merge(token_seqs, pair, self.merges[pair])
        
        return [tok for seq in token_seqs for tok in seq]

    def decode(self, tokens: List[int]) -> str:
        _bytes = b"".join(self.vocab[token_id] for token_id in tokens)
        return _bytes.decode(encoding="utf-8", errors="replace")

    @staticmethod
    def get_stats(tokens: List[List[int]]) -> Dict[Tuple, int]:
            freq = defaultdict(int)
            for seq in tokens:
                for i in range(1, len(seq)):
                    freq[seq[i-1], seq[i]] += 1
            return freq

    @staticmethod
    def merge(tokens: List[List[int]], pair: Tuple[int], new_id: int) -> List[List[int]]:
        new_tokens = []
        for byte_seq in tokens:
            new_byte_seq = []
            i = 0
            while i < len(byte_seq) - 1:
                if byte_seq[i] == pair[0] and byte_seq[i+1] == pair[1]:
                    new_byte_seq.append(new_id)
                    i += 2
                else:
                    new_byte_seq.append(byte_seq[i])
                    i += 1
            
            if i==len(byte_seq) - 1:
                new_byte_seq.append(byte_seq[-1])
            new_tokens.append(new_byte_seq)
        
        return new_tokens

    @staticmethod
    def build_vocab(merges: Dict[Tuple[int], int]) -> Dict[int, List]:
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for (p0, p1), token_id in sorted(merges.items(), key=lambda merge: merge[1]):
            vocab[token_id] = vocab[p0] + vocab[p1]
        return vocab
 
    @classmethod
    def train(cls, text: str, num_merges: int, tokenizer_dir: str) -> "BPE":
        merges: Dict[Tuple[int], int] = {}
        words = re.findall(pattern=BPE.reg_pat, string=text)
        tokens = [list(w.encode("utf-8")) for w in words]
        for i in tqdm(range(num_merges)):
            freq = cls.get_stats(tokens)
            if len(freq) == 0:
                break
            pair = max(freq, key=freq.get)
            new_id = 256 + i
            merges[pair] = new_id
            tokens = cls.merge(tokens, pair, new_id)
        
        vocab = cls.build_vocab(merges)

        ## save
        os.makedirs(tokenizer_dir, exist_ok=True)

        merges_path = os.path.join(tokenizer_dir, "merges.json")
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")

        with open(merges_path, "w") as f:
            json.dump({f"{pair[0]},{pair[1]}": idx for pair, idx in merges.items()}, f, indent=2)

        with open(vocab_path, "w") as f:
            json.dump({str(idx): list(byte_seq) for idx, byte_seq in vocab.items()}, f, indent=2)
        
        return BPE(tokenizer_dir)

        

if __name__ == "__main__":
    with open("data/tinystories/train.txt", "r") as f:
        text = f.read()
    
    tokenizer = BPE.train(
        text=text,
        num_merges=3000,
        tokenizer_dir="tokenizer"
    )
    string = "Hi how are you?"
    print(f"encoded: {tokenizer.encode(string)}")
    print(f"decoded: {tokenizer.decode(tokenizer.encode(string))}")