import logging
import random as rand
import re
from collections import defaultdict

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


class FastText:
    def __init__(
        self,
        num_class: int = 2,
        lr: float = 0.005,
        ngram_size: int = 3,
        emb_size: int = 100,
        batch_size: int = 16,
        bucket_size: int = int(5e5),
        dropout_rate: float = 0.1,
        min_subword_freq=2,
        max_token=200,
    ):
        log.info("Initializing config")
        self.num_class = num_class
        self.lr = lr
        self.ngram_size = ngram_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.dropout_rate = dropout_rate
        self.min_subword_freq = min_subword_freq
        self.max_token = max_token

        log.info("Initializing parameters")
        self.vocab = 0.1 * np.random.randn(bucket_size, emb_size).astype(np.float32)
        self.weights = 0.1 * np.random.randn(emb_size, num_class).astype(np.float32)
        self.bias = np.zeros(num_class, dtype=np.float32)

        log.info("Initializing aux stuff")
        self.subword_counts: defaultdict[str, int] = defaultdict(int)
        self.subword_mask: np.ndarray = np.ones(bucket_size, dtype=bool)
        self._loss_vals: list[float] = []
        self._loss_test_vals: list[float] = []

    def tokenize(self, text: str):
        text = text.lower()
        return [
            self.tokenize_ngrams(i)
            for i in TOKENIZER_RE.findall(text)[: self.max_token]
        ]

    def tokenize_ngrams(self, word: str):
        ext = f"<{word}>"
        ngrams = []
        if len(word) > self.ngram_size:
            ngrams += [
                ext[i : i + self.ngram_size]
                for i in range(len(ext) - self.ngram_size + 1)
            ]
        if len(word) != self.ngram_size + 1:
            ngrams.append(word)
        return ngrams

    def embed(self, tokens: list[list[str]]):
        log.debug("Embedding text")
        sent_emb = []
        for word in tokens:
            ngram_vec = []
            for ngram in word:
                idx = self.hash(ngram)
                if not self.subword_mask[idx]:
                    continue
                emb = self.vocab[idx]
                ngram_vec.append(emb)
            if ngram_vec:
                word_emb = sum(ngram_vec) / len(ngram_vec)
            else:
                word_emb = np.zeros(self.emb_size)
            sent_emb.append(word_emb)
        return np.mean(sent_emb, axis=0) if sent_emb else np.zeros(self.emb_size)

    def build_subword_mask(self, data, max_tok=int(2e5)):
        log.info("Building subword mask")
        for text in data["text"]:
            tokens = TOKENIZER_RE.findall(text.lower())
            rand.shuffle(tokens)
            for word in tokens[:max_tok]:
                ext = f"<{word}>"
                ngrams = [
                    ext[i : i + self.ngram_size]
                    for i in range(len(ext) - self.ngram_size + 1)
                ]
                for ng in ngrams:
                    self.subword_counts[ng] += 1

        self.subword_mask = np.zeros(self.bucket_size, dtype=bool)
        for ng, count in self.subword_counts.items():
            if count >= self.min_subword_freq:
                idx = self.hash(ng)
                self.subword_mask[idx] = True

    def hash(self, text: str, seed: int = 0xCBF29CE484222325):
        prime = 0x100000001B3
        h = seed & 0xFFFFFFFFFFFFFFFF
        for b in text.encode("utf-8"):
            h ^= b
            h = (h * prime) & 0xFFFFFFFFFFFFFFFF
        return h % self.bucket_size

    @staticmethod
    def apply_dropout(x, rate):
        if rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) >= rate).astype(x.dtype)
        return x * mask / (1 - rate)

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
        tok_df = text_df.apply(self.tokenize)
        embed_df = tok_df.apply(self.embed)
        embed_mat = np.stack(embed_df.values, axis=0)
        embed_mat = self.apply_dropout(embed_mat, rate=self.dropout_rate)

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

        for i in range(len(tok_df)):
            text_token = tok_df.iloc[i]
            grad = dl_dx[i]
            total_len = sum(len(word) for word in text_token)
            for word in text_token:
                grad_contrib = (self.lr * grad) / total_len
                for ngram in word:
                    idx = self.hash(ngram)
                    self.vocab[idx] -= grad_contrib

    def eval(self, val_data: pd.DataFrame):
        text_df, y_true_df = val_data["text"], val_data["label"]
        tok_df = text_df.apply(self.tokenize)
        embed_df = tok_df.apply(self.embed)
        embed_mat = np.stack(embed_df.values, axis=0)
        y_pred = self.forward(embed_mat)
        loss_val = self.loss(y_true_df, y_pred)
        self._loss_test_vals.append(loss_val)
        return loss_val


if __name__ == "__main__":
    train_data = pd.DataFrame(
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

    val_data = pd.DataFrame(
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

    ft = FastText()
