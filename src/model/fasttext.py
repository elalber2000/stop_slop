import logging
import random as rand
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("train.log"),  # optional: save to file
    ],
)
log = logging.getLogger("FastTextTrainer")

TOKENIZER_RE = re.compile(r"\w+|[^\w\s]+")


class FastTextTokenizer:
    def __init__(
        self,
        bucket_size: int = int(5e5),
        ngram_size: int = 3,
        min_subword_freq: int = 2,
        max_token: int = 200,
        tok_pattern: str = r"\w+|[^\w\s]+",
        preprocess_fn: Callable[[str], str] | None = None,
    ):
        self.bucket_size = bucket_size
        self.ngram_size = ngram_size
        self.min_subword_freq = min_subword_freq
        self.max_token = max_token
        self.re_tok = re.compile(tok_pattern)
        self.preprocess_fn = preprocess_fn or (lambda s: s)
        self.subword_counts: defaultdict[str, int] = defaultdict(int)
        self.subword_mask: np.ndarray = np.ones(bucket_size, dtype=bool)

    def hash(self, text: str, seed: int = 0xCBF29CE484222325) -> int:
        prime = 0x100000001B3
        h = seed & 0xFFFFFFFFFFFFFFFF
        for b in text.encode("utf-8"):
            h ^= b
            h = (h * prime) & 0xFFFFFFFFFFFFFFFF
        return h % self.bucket_size

    def _split(self, text: str) -> list[str]:
        return self.re_tok.findall(self.preprocess_fn(text.lower()))

    def _ngrams(self, word: str) -> list[str]:
        ext = f"<{word}>"
        n = self.ngram_size
        ngrams = (
            [ext[i : i + n] for i in range(len(ext) - n + 1)] if len(word) > n else []
        )
        if len(word) != n + 1:
            ngrams.append(word)
        return ngrams

    def encode(
        self, text: str, *, return_hashes: bool = True
    ) -> list[list[int] | list[str]]:
        words = self._split(text)[: self.max_token]
        toks = [self._ngrams(w) for w in words]
        if not return_hashes:
            return toks
        out = []
        for w in toks:
            ids = []
            for ng in w:
                h = self.hash(ng)
                if self.subword_mask[h]:
                    ids.append(h)
            out.append(ids)
        return out

    def batch_encode(
        self, texts: Sequence[str], *, return_hashes: bool = True
    ) -> list[list[list[int]]]:
        return [self.encode(t, return_hashes=return_hashes) for t in texts]

    def build_subword_mask(self, texts: Iterable[str], max_tok: int = int(2e5)) -> None:
        for txt in texts:
            tokens = self._split(txt)
            k = min(max_tok, len(tokens))
            for w in rand.sample(tokens, k=k):
                for ng in self._ngrams(w):
                    self.subword_counts[ng] += 1
        mask = np.zeros(self.bucket_size, dtype=bool)
        for ng, cnt in self.subword_counts.items():
            if cnt >= self.min_subword_freq:
                mask[self.hash(ng)] = True
        self.subword_mask = mask


class FastText:
    def __init__(
        self,
        num_class: int = 2,
        lr: float = 0.05,
        emb_size: int = 100,
        batch_size: int = 16,
        bucket_size: int = int(5e5),
        dropout_rate: float = 0.1,
    ):
        log.info("Initializing config")
        self.num_class = num_class
        self.lr = lr
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.dropout_rate = dropout_rate

        log.info("Initializing parameters")
        self.vocab = 0.1 * np.random.randn(bucket_size, emb_size).astype(np.float32)
        self.weights = 0.1 * np.random.randn(emb_size, num_class).astype(np.float32)
        self.bias = np.zeros(num_class, dtype=np.float32)

        log.info("Initializing vals")
        self._loss_vals: list[float] = []
        self._loss_test_vals: list[float] = []

    def embed(self, ids: list[list[int]]):
        log.debug("Embedding text")
        word_embs = []
        for w in ids:
            embs = [self.vocab[i] for i in w]
            if embs:
                word_embs.append(np.mean(embs, axis=0))
            else:
                word_embs.append(np.zeros(self.emb_size, dtype=np.float32))
        return (
            np.mean(word_embs, axis=0)
            if word_embs
            else np.zeros(self.emb_size, dtype=np.float32)
        )

    def apply_dropout(self, x):
        if self.dropout_rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) >= self.dropout_rate).astype(x.dtype)
        return x * mask / (1 - self.dropout_rate)

    def forward(self, embed_df):
        log.debug("Forward pass")
        x = embed_df
        if x.ndim == 1:
            x = x[np.newaxis, :]
        log.debug("- Linear layer")
        linear = x @ self.weights + self.bias
        log.debug("- Softmax")
        exp_shift = np.exp(linear - np.max(linear, axis=1, keepdims=True))
        logits = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return logits

    def loss(self, y_true, y_pred):
        log.debug("Calculating loss")
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.eye(y_pred.shape[1])[y_true]
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, batch: pd.DataFrame):
        log.debug("Backward pass")
        log.debug("- Tokenizing and embedding")
        text_df, y_true_df = batch["text"], batch["label"]
        embed_df = text_df.apply(self.embed)
        embed_mat = np.stack(embed_df.values, axis=0)
        embed_mat = self.apply_dropout(embed_mat)

        y_pred = self.forward(embed_mat)
        loss = self.loss(y_true_df, y_pred)
        self._loss_vals.append(loss)

        log.debug("- Logits dz")
        y_true_vec = np.eye(y_pred.shape[1])[y_true_df]
        dl_dz = y_pred - y_true_vec

        log.debug("- Linear layer")
        dl_dw = embed_mat.T @ dl_dz
        self.weights -= self.lr * dl_dw

        log.debug("- Bias")
        dl_db = np.sum(dl_dz, axis=0)
        self.bias -= self.lr * dl_db

        log.debug("- Embeddings")
        dl_dx = dl_dz @ self.weights.T

        for i, text_token in enumerate(text_df):
            grad = dl_dx[i]
            total_len = sum(len(word) for word in text_token)
            if total_len == 0:
                continue  # no subwords → skip embedding update
            for word in text_token:
                grad_contrib = (self.lr * grad) / total_len
                for ngram in word:
                    self.vocab[ngram] -= grad_contrib

    def eval(self, val_data: pd.DataFrame):
        text_df, y_true_df = val_data["text"], val_data["label"]
        embed_df = text_df.apply(self.embed)
        embed_mat = np.stack(embed_df.values, axis=0)
        y_pred = self.forward(embed_mat)
        loss_val = self.loss(y_true_df, y_pred)
        self._loss_test_vals.append(loss_val)
        return loss_val

    def predict(self, tokens):
        probs = self.forward(self.embed(tokens))
        return probs.flatten()


if __name__ == "__main__":
    train_df = pd.DataFrame(
        [
            # Positive
            ("i love this", 1),
            ("this is amazing", 1),
            ("really great work", 1),
            ("fantastic effort", 1),
            ("absolutely wonderful", 1),
            ("i really liked it", 1),
            ("superb execution", 1),
            ("brilliant result", 1),
            ("top notch job", 1),
            ("impressive work", 1),
            ("everything is perfect", 1),
            ("nailed it", 1),
            ("great outcome", 1),
            ("this is excellent", 1),
            ("outstanding performance", 1),
            # Negative
            ("i hate this", 0),
            ("this is terrible", 0),
            ("really bad job", 0),
            ("horrible result", 0),
            ("absolutely awful", 0),
            ("i really disliked it", 0),
            ("poor execution", 0),
            ("disappointing work", 0),
            ("low quality", 0),
            ("very sloppy", 0),
            ("this sucks", 0),
            ("worst ever", 0),
            ("not good", 0),
            ("this is garbage", 0),
            ("total failure", 0),
        ],
        columns=["text", "label"],
    )

    val_df = pd.DataFrame(
        [
            # Positive
            ("i love it", 1),
            ("this is fantastic", 1),
            ("very well done", 1),
            ("high quality work", 1),
            ("i'm impressed", 1),
            # Negative
            ("hate this thing", 0),
            ("this is trash", 0),
            ("completely useless", 0),
            ("very bad job", 0),
            ("i'm disappointed", 0),
        ],
        columns=["text", "label"],
    )

    tok = FastTextTokenizer(bucket_size=200_000, ngram_size=3)
    tok.build_subword_mask(train_df["text"])

    model = FastText(
        num_class=2,
        bucket_size=tok.bucket_size,
        dropout_rate=0.2,
    )

    train_tokens = tok.batch_encode(train_df["text"])
    val_tokens = tok.batch_encode(val_df["text"])

    EPOCHS = 30
    for ep in range(1, EPOCHS + 1):
        model._loss_vals.clear()

        idx = np.random.permutation(len(train_df))
        for start in range(0, len(idx), model.batch_size):
            batch_idx = idx[start : start + model.batch_size]
            batch_toks = [train_tokens[i] for i in batch_idx]
            batch_lbls = train_df["label"].values[batch_idx]
            batch_df = pd.DataFrame({"text": batch_toks, "label": batch_lbls})
            model.backward(batch_df)

        train_loss = float(np.mean(model._loss_vals))

        val_batch = pd.DataFrame({"text": val_tokens, "label": val_df["label"].values})
        vl = model.eval(val_batch)
        log.info(f"Epoch {ep} train_loss={train_loss:.4f} val_loss={vl:.4f}")
