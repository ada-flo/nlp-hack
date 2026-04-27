"""FastText pretrained embedding loader for the BiLSTM encoder.

FastText vectors are word-level, our tokenizer is BPE — direct lookup misses.
We approximate by averaging the vectors of FastText words that *contain* the
BPE subword as a substring. Crude but consistently better than random init
on small data, and only used as initialization (still trained end-to-end).

Vector files (~5 GB each) — download once:

    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
    https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz

Drop them at:
    data/raw_manual/fasttext/cc.en.300.vec
    data/raw_manual/fasttext/cc.ko.300.vec
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
import torch
from tqdm import tqdm


def _iter_vec_lines(path: Path) -> Iterable[tuple[str, np.ndarray]]:
    with path.open(encoding="utf-8") as f:
        first = f.readline().split()
        try:
            int(first[0])  # vocab size header
        except ValueError:
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 50:
                continue
            word = parts[0]
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            yield word, vec


def load_fasttext_for_spm(
    sp: spm.SentencePieceProcessor,
    vec_paths: list[Path],
    target_dim: int,
    max_words: int | None = 500_000,
) -> torch.Tensor:
    """Build an embedding matrix [vocab_size, target_dim] aligned to the SP vocab.

    For each piece, average FastText vectors of all loaded words containing
    the piece's surface form (after stripping ▁). Random Xavier init for misses.
    """
    vocab_size = sp.get_piece_size()
    pieces = [sp.id_to_piece(i).lstrip("▁") for i in range(vocab_size)]

    # Collect FastText vectors (word → vec).
    ft: dict[str, np.ndarray] = {}
    for path in vec_paths:
        if not path.exists():
            continue
        for i, (word, vec) in enumerate(tqdm(_iter_vec_lines(path),
                                             desc=f"reading {path.name}")):
            if max_words and i >= max_words:
                break
            ft[word] = vec

    if not ft:
        print("[embeddings] no FastText vectors loaded; using random init")
        return _xavier_init(vocab_size, target_dim)

    src_dim = next(iter(ft.values())).shape[0]
    matrix = _xavier_init(vocab_size, target_dim).numpy()

    # Pre-bucket FastText words by 3-char prefix for crude lookup speed.
    prefix_index: dict[str, list[str]] = {}
    for word in ft:
        prefix_index.setdefault(word[:3], []).append(word)

    hits = 0
    for idx, piece in enumerate(pieces):
        if not piece:
            continue
        candidates = prefix_index.get(piece[:3], []) if len(piece) >= 3 else []
        # Direct word match first.
        if piece in ft:
            vec = ft[piece]
        else:
            substr_matches = [w for w in candidates if piece in w][:32]
            if not substr_matches:
                continue
            vec = np.mean([ft[w] for w in substr_matches], axis=0)
        if src_dim != target_dim:
            # Project / pad.
            if src_dim >= target_dim:
                vec = vec[:target_dim]
            else:
                pad = np.zeros(target_dim - src_dim, dtype=np.float32)
                vec = np.concatenate([vec, pad])
        matrix[idx] = vec
        hits += 1

    coverage = hits / vocab_size
    print(f"[embeddings] FastText coverage: {hits}/{vocab_size} ({coverage:.1%})")
    return torch.from_numpy(matrix).float()


def _xavier_init(vocab_size: int, dim: int) -> torch.Tensor:
    bound = (6.0 / (vocab_size + dim)) ** 0.5
    return torch.empty(vocab_size, dim).uniform_(-bound, bound)
