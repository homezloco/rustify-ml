"""
BPE (Byte-Pair Encoding) tokenizer — rustify-ml acceleration target.

This is a simplified but realistic BPE encode loop similar to what
tiktoken / HuggingFace tokenizers use internally. The inner while-loop
merge pass is O(n²) in the number of tokens and is the primary hotspot.

Expected Rust speedup: 10–50x via Vec<usize> + HashMap lookups.
"""

from __future__ import annotations


def bpe_encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Encode text using BPE merge rules.

    Args:
        text:   Input string to encode.
        merges: Ordered list of (a, b) merge pairs. Earlier = higher priority.

    Returns:
        List of token IDs after all merges are applied.
    """
    # Start with UTF-8 byte values as initial tokens
    tokens = list(text.encode("utf-8"))

    # Build a merge priority map: (a, b) -> rank (lower = higher priority)
    merge_rank: dict[tuple[int, int], int] = {}
    for rank, pair in enumerate(merges):
        merge_rank[pair] = rank

    # Apply merges greedily (lowest rank first)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_rank:
                # Merge: replace pair with a new token ID
                new_id = 256 + merge_rank[pair]
                tokens[i] = new_id
                tokens.pop(i + 1)
                changed = True
            else:
                i += 1

    return tokens


def count_pairs(tokens: list[int]) -> dict[tuple[int, int], int]:
    """Count all adjacent pairs in a token list (used during BPE training)."""
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def build_vocab(text: str, num_merges: int) -> list[tuple[int, int]]:
    """Train BPE: greedily merge the most frequent pair num_merges times."""
    tokens = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []

    for _ in range(num_merges):
        counts = count_pairs(tokens)
        if not counts:
            break
        best = max(counts, key=lambda p: counts[p])
        merges.append(best)
        new_id = 256 + len(merges) - 1
        # Replace all occurrences of best pair
        i = 0
        new_tokens: list[int] = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return merges


if __name__ == "__main__":
    import time

    # Simulate a realistic ML workload: encode a batch of short texts
    sample_text = (
        "the quick brown fox jumps over the lazy dog. " * 20
    )

    # Train a tiny vocab on the sample
    print("Training BPE vocab (50 merges)...")
    merges = build_vocab(sample_text, num_merges=50)
    print(f"  Learned {len(merges)} merge rules")

    # Benchmark: encode 10,000 times
    n_iters = 10_000
    print(f"\nBenchmarking bpe_encode x{n_iters}...")
    start = time.perf_counter()
    for _ in range(n_iters):
        result = bpe_encode(sample_text[:50], merges)
    elapsed = time.perf_counter() - start
    print(f"  bpe_encode (50 chars): {elapsed:.3f}s for {n_iters} iters")
    print(f"  avg: {elapsed / n_iters * 1000:.4f}ms per call")
    print(f"  output tokens: {len(result)}")

    # Benchmark count_pairs
    tokens = list(sample_text.encode("utf-8"))
    print(f"\nBenchmarking count_pairs x{n_iters}...")
    start = time.perf_counter()
    for _ in range(n_iters):
        pairs = count_pairs(tokens)
    elapsed = time.perf_counter() - start
    print(f"  count_pairs ({len(tokens)} tokens): {elapsed:.3f}s for {n_iters} iters")
    print(f"  unique pairs: {len(pairs)}")
