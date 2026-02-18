"""
Slow BPE-style tokenizer loop - target for rustify-ml acceleration.
Expected gain: 10-20x via Rust loop + HashMap.
"""


def tokenize(text, vocab):
    """Split text into subword tokens using a simple greedy BPE-like scan."""
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for length in range(min(10, len(text) - i), 0, -1):
            substr = text[i : i + length]
            if substr in vocab:
                tokens.append(vocab[substr])
                i += length
                matched = True
                break
        if not matched:
            tokens.append(0)  # unknown token
            i += 1
    return tokens


if __name__ == "__main__":
    import time

    vocab = {chr(c): c for c in range(32, 127)}
    text = "hello world " * 1000
    start = time.time()
    for _ in range(100):
        tokenize(text, vocab)
    elapsed = time.time() - start
    print(f"tokenize: {elapsed:.3f}s for 100 iterations")
