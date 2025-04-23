import random as rand
import re
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOKENIZER_RE = re.compile(r"\w+|[^\w\s]+")


class FastText:
    def __init__(
        self,
        num_class: int = 2,
        lr: float = 0.005,
        ngram_size: int = 3,
        emb_size: int = 100,
        batch_size: int = 16,
        bucket_size: int = int(2e6),
        dropout_rate: float = 0.1,
        min_subword_freq=2,
        max_token=200,
    ):
        self.num_class = num_class
        self.lr = lr
        self.ngram_size = ngram_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.dropout_rate = dropout_rate
        self.min_subword_freq = min_subword_freq
        self.max_token = max_token

        self.vocab = np.random.normal(0, 0.1, size=(bucket_size, emb_size))
        self.weights = np.random.normal(0, 0.1, size=(emb_size, num_class))
        self.bias = np.zeros(num_class)

        self.subword_counts: defaultdict[str, int] = defaultdict(int)
        self.subword_mask: np.ndarray = np.ones(bucket_size, dtype=bool)
        self._loss_vals: list[float] = []
        self._loss_test_vals: list[float] = []

    def tokenize(self, text: str):
        # print(f"Processing '{text}'")
        text = text.lower()
        return [
            self.tokenize_ngrams(i)
            for i in TOKENIZER_RE.findall(text.lower())[: self.max_token]
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
        for text in data["text"]:
            tokens = re.findall(r"\w+|[^\w\s]+", text.lower())
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

    def forward(
        self,
        embed_df,
    ):
        #  (BxD)
        x = embed_df

        if x.ndim == 1:
            x = x[np.newaxis, :]

        #  (DxC)(BxD)+(C)
        # print("Wx+B")
        linear = x @ self.weights + self.bias
        # print(linear)

        # print("Logits")
        exp_shift = np.exp(linear - np.max(linear, axis=1, keepdims=True))
        logits = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        # print(logits)

        return logits

    def loss(
        self,
        y_true,
        y_pred,
    ):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.eye(y_pred.shape[1])[y_true]
        # print(f"y_true = {y_true}, y_pred = {y_pred}")
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, batch: pd.DataFrame):
        text_df, y_true_df = batch["text"], batch["label"]
        tok_df = text_df.apply(self.tokenize)
        embed_df = tok_df.apply(self.embed)
        embed_mat = np.stack(embed_df.values, axis=0)
        embed_mat = self.apply_dropout(embed_mat, rate=self.dropout_rate)

        y_pred = self.forward(embed_mat)
        loss = self.loss(y_true_df, y_pred)
        print(f"Loss: {loss}")
        self._loss_vals.append(loss)

        y_true_vec = np.eye(y_pred.shape[1])[y_true_df]
        dl_dz = y_pred - y_true_vec
        # print(f"dl_dz: {dl_dz}")

        # print(dl_dz)
        dl_dw = embed_mat.T @ dl_dz
        # print(f"dl_dw: {dl_dw}")
        self.weights -= self.lr * dl_dw

        dl_db = np.sum(dl_dz, axis=0)
        # print(f"dl_db: {dl_db}")
        self.bias -= self.lr * dl_db

        dl_dx = dl_dz @ self.weights.T
        # print(f"dl_dx: {dl_dx}")

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

    def plot_loss(self, epoch_num, row_num):
        [i / epoch_num for i in range(epoch_num * row_num)]
        plt.plot(self._loss_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def plot_epoch_loss(self, epoch_num, row_num):
        epoch_avg = [
            np.mean(self._loss_vals[i * row_num : (i + 1) * row_num])
            for i in range(epoch_num)
        ]
        plt.plot(epoch_avg)
        if self._loss_test_vals != []:
            plt.plot(self._loss_test_vals)
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.title("Epoch-Level Loss")
        plt.show()

    def word_contributions(self, text: str, target_class: Optional[int] = None):
        tokens = self.tokenize(text)
        embeddings = []
        ngram_indices = []

        for word in tokens:
            word_emb_list, word_ids = [], []
            for ngram in word:
                idx = self.hash(ngram)
                if self.subword_mask[idx]:
                    word_emb_list.append(self.vocab[idx])
                    word_ids.append(idx)
            if word_emb_list:
                word_emb = np.mean(np.array(word_emb_list), axis=0)
            else:
                word_emb = np.zeros(self.emb_size)

            embeddings.append(word_emb)
            ngram_indices.append(word_ids)

        sentence_emb = np.mean(embeddings, axis=0)
        logits = sentence_emb @ self.weights + self.bias
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        if target_class is None:
            target_class = int(np.argmax(probs))

        grad = self.weights[:, target_class]

        contributions = [float(np.dot(grad, w)) for w in embeddings]

        return list(zip(["".join(word) for word in tokens], contributions))

    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        epoch_num: int = 15,
    ):
        if self.min_subword_freq > 0:
            self.build_subword_mask(train_data)

        for epoch in range(epoch_num):
            train_data = train_data.sample(frac=1).reset_index(drop=True)

            for batch_i in range(0, len(train_data), self.batch_size):
                print(f"Epoch: {epoch}, Batch: {batch_i/self.batch_size}")
                self.backward(train_data.iloc[batch_i : batch_i + self.batch_size])
                print("-" * 100)

                for plot_func in [self.plot_loss, self.plot_epoch_loss]:
                    plot_func(epoch_num, len(train_data))

            if eval_data is not None:
                self.eval(eval_data)


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
    ft.train(train_data, val_data)
