from collections import defaultdict
import regex as re
from tqdm import tqdm
from typing import List, Dict, Tuple
import json


class BPE:

    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path, "r") as f:
            vocab = json.loads(f.read())

        self.merges: List[List[int]] = vocab["merges"]
        self.id2pair: Dict[int, List[int]] = {int(k): tuple(v) for k, v in vocab["id2pair"].items()}
        self.pair2id = {v: k for k, v in self.id2pair.items()}
        self.reg_pat = re.compile(
            r"""'s|'ve|'d|'ll|'t|'re|'m| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.IGNORECASE
        )  

    def encode(self, text: str) -> List[int]:
        words = re.findall(self.reg_pat, text)
        token_seqs = [list(w.encode("utf-8")) for w in words]
        for merge in self.merges:
            new_token_seqs = []
            for byte_seq in token_seqs:
                new_seq = []
                i = 0
                while i < len(byte_seq) - 1:
                    if byte_seq[i] == merge[0] and byte_seq[i+1] == merge[1]:
                        new_seq.append(self.pair2id[tuple(merge)])
                        i += 2
                    else:
                        new_seq.append(byte_seq[i])
                        i += 1
                if i == len(byte_seq) - 1:
                    new_seq.append(byte_seq[i])
                new_token_seqs.append(new_seq)
            token_seqs = new_token_seqs[:]

        return [tok for seq in token_seqs for tok in seq]


    def decode(self, tokens: List[int]) -> str:
        decoded_tokens = tokens[:]
        while True:
            if all(dec_tok <= 255 for dec_tok in decoded_tokens):
                break
            _decoded_tokens = []
            for tok in decoded_tokens:
                if tok <= 255:
                    _decoded_tokens.append(tok)
                else:
                    _decoded_tokens.extend(self.id2pair[tok])
            decoded_tokens = _decoded_tokens[:]

        return bytes(decoded_tokens).decode("utf-8", errors="replace")

    @staticmethod
    def train(text: str, num_merges: int, save_path: str):
        reg_pat = re.compile(
            r"""'s|'ve|'d|'ll|'t|'re|'m| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.IGNORECASE
        ) 

        def _get_stats(tokens: List[List[int]]) -> Dict[Tuple, int]:
            freq = defaultdict(int)
            for byte_seq in tokens:
                for i in range(1, len(byte_seq)):
                    freq[byte_seq[i-1], byte_seq[i]] += 1
            return freq

        def _merge(tokens: List[List[int]], pair: Tuple[int], new_id: int) -> List[List[int]]:
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

        words = re.findall(pattern=reg_pat, string=text)
        tokens = [list(w.encode("utf-8")) for w in words]
        merges = []
        for i in tqdm(range(num_merges)):
            freq = _get_stats(tokens)
            if len(freq) == 0:
                break
            pair = max(freq, key=freq.get)
            new_id = 256 + i
            merges.append(pair)
            tokens = _merge(tokens, pair, new_id)
        
        with open(save_path, "w") as f:
            json_to_dump = {
                "merges": merges,
                "id2pair": {str(256+i): list(pair) for i, pair in enumerate(merges)}
            }
            f.write(json.dumps(json_to_dump))
    


if __name__ == "__main__":
    with open("data/tinystories/train.txt", "r") as f:
        text = f.read()
    
    BPE.train(
        text=text, 
        num_merges=3000,
        save_path="vocab.json"
    )