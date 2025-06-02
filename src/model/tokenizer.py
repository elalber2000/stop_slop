import random as rand
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from typing import Callable

import numpy as np


class AbstractTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list:
        """Encode a single string into tokens or token IDs."""
        ...

    @abstractmethod
    def batch_encode(self, texts: Iterable[str]) -> list:
        """Encode a list of strings."""
        ...


def _hash_impl(
    text: str,
    bucket_size: int,
    seed: int = 0xCBF29CE484222325,
) -> int:
    prime = 0x100000001B3
    h = seed & 0xFFFFFFFFFFFFFFFF
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h % bucket_size


def hash(text: str, bucket_size: int, seed: int = 0xCBF29CE484222325) -> int:
    return _hash_impl(text, bucket_size=bucket_size, seed=seed)


class FastTextTokenizer(AbstractTokenizer):
    def __init__(
        self,
        bucket_size: int = int(5e5),
        ngram_size: int | tuple[int, int] = 3,
        min_subword_freq: int = 2,
        max_token: int = 200,
        tok_pattern: str = r"\w+|[^\w\s]+",
        preprocess_fn: Callable[[str], str] | None = None,
    ):
        if isinstance(ngram_size, tuple):
            if len(ngram_size) != 2 or ngram_size[0] >= ngram_size[1]:
                raise ValueError(
                    "ngram_size tuple must be (min_ngram_size, max_ngram_size)"
                )
            self.min_ngram, self.max_ngram = ngram_size
        else:
            self.min_ngram = self.max_ngram = ngram_size

        self.bucket_size = bucket_size
        self.min_subword_freq = min_subword_freq
        self.max_token = max_token
        self.re_tok = re.compile(tok_pattern)
        self.preprocess_fn = preprocess_fn or (lambda s: s)

        self._pattern_full = re.compile(r"^__\w+__$")
        self._ngram_cache = lru_cache(maxsize=None)(self._ngrams_impl)

        self.subword_counts: defaultdict[str, int] = defaultdict(int)
        self.subword_mask: np.ndarray = np.ones(bucket_size, dtype=bool)

    def _split(self, text: str) -> list[str]:
        return self.re_tok.findall(self.preprocess_fn(text.lower()))

    def _ngrams_impl(self, word: str) -> list[str]:
        if self._pattern_full.match(word):
            return [word]
        ext = word  # f"<{word}>"
        seen = set()
        out: list[str] = []
        for n in range(self.min_ngram, self.max_ngram + 1):
            if len(word) > n:
                for i in range(len(ext) - n + 1):
                    ng = ext[i : i + n]
                    if ng not in seen:
                        seen.add(ng)
                        out.append(ng)
            if len(word) != n + 1 and word not in seen:
                seen.add(word)
                out.append(word)
        return out

    def _ngrams(self, word: str) -> list[str]:
        return self._ngram_cache(word)

    def encode(
        self, text: str, *, return_hashes: bool = True
    ) -> list[list[int]] | list[list[str]]:
        words = self._split(text)[: self.max_token]
        toks = [self._ngrams(w) for w in words]
        if not return_hashes:
            return toks

        out: list[list[int]] = []
        for w in toks:
            ids = [
                h
                for ng in w
                if (h := hash(ng, bucket_size=self.bucket_size))
                and self.subword_mask[h]
            ]
            out.append(ids)
        return out

    def batch_encode(
        self, texts: Iterable[object], *, return_hashes: bool = True
    ) -> list[list[list[int]] | list[list[str]]]:
        return [self.encode(str(t), return_hashes=return_hashes) for t in texts]

    def build_subword_mask(self, texts: Iterable[str], max_tok: int = int(2e5)) -> None:
        counts: Counter = Counter()
        for txt in texts:
            tokens = self._split(txt)
            sampled = rand.sample(tokens, k=min(max_tok, len(tokens)))
            for w in sampled:
                for ng in self._ngrams(w):
                    counts[ng] += 1

        mask = np.zeros(self.bucket_size, dtype=bool)
        for ng, cnt in counts.items():
            if cnt >= self.min_subword_freq:
                mask[hash(ng, bucket_size=self.bucket_size)] = True
        self.subword_mask = mask


class WhitelistFastTextTokenizer(AbstractTokenizer):
    def __init__(
        self,
        allowed_tokens: list[str],
        bucket_size: int = int(5e5),
        tok_pattern: str = r"\w+|[^\w\s]+",
        preprocess_fn: Callable[[str], str] | None = None,
    ):
        self.allowed = set(allowed_tokens)
        self.bucket_size = bucket_size
        lengths = {len(tok) for tok in allowed_tokens}
        self.allowed_lengths = lengths
        self.re_tok = re.compile(tok_pattern)
        self.preprocess_fn = preprocess_fn or (lambda s: s)

    def _split(self, text: str) -> list[str]:
        return self.re_tok.findall(self.preprocess_fn(text.lower()))

    def _ngrams(self, word: str) -> list[str]:
        out: list[str] = []
        for L in self.allowed_lengths:
            if len(word) < L:
                continue
            for i in range(len(word) - L + 1):
                sub = word[i : i + L]
                if sub in self.allowed:
                    out.append(sub)
        return out

    def encode(self, text: str, *, return_hashes: bool = True) -> list[list[str]]:
        toks = [self._ngrams(w) for w in self._split(text)]
        if not return_hashes:
            return toks

        out: list[list[int]] = []
        for w in toks:
            ids = [h for ng in w if (h := hash(ng, bucket_size=self.bucket_size))]
            out.append(ids)
        return out

    def batch_encode(self, texts: Iterable[str]) -> list[list[list[str]]]:
        return [self.encode(txt) for txt in texts]
