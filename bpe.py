from collections import defaultdict


def get_stats(tokens: list[int]) -> dict:
    freq = defaultdict(int)
    for i in range(1, len(tokens)):
        freq[tokens[i-1], tokens[i]] += 1
    return freq


def merge(tokens: list[int], pair: tuple, new_id: int) -> list[int]:
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    
    if i == len(tokens) - 1:
        new_tokens.append(tokens[i])

    return new_tokens

def bpe(text: str, num_merges: int) -> tuple:
    tokens = list(text.encode("utf-8"))
    vocab = {}
    for i in range(num_merges):
        freq = get_stats(tokens)
        pair = max(freq, key=freq.get)
        new_id = 256 + i
        vocab[new_id] = pair
        tokens = merge(tokens, pair, new_id)
    return tokens, vocab


if __name__ == "__main__":
    import json

    with open("sample_txt.txt", "r") as f:
        text = f.read()
    
    tokens, vocab = bpe(text, 200)
    with open("vocab.json", "w") as f:
        f.write(json.dumps(vocab))