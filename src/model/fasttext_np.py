import logging

import numpy as np
import pandas as pd
from model.tokenizer import FastTextTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"),
    ],
)
log = logging.getLogger("FastTextTrainer")


class FastTextModel:
    def __init__(
        self,
        num_class: int = 2,
        lr: float = 0.05,
        emb_size: int = 100,
        batch_size: int = 16,
        bucket_size: int = int(5e5),
        dropout_rate: float = 0,
        max_grad_norm: float = 5.0,
        reg_lambda: float = 1e-4,
    ):
        log.info("Initializing config")
        self.num_class = num_class
        self.lr = lr
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.dropout_rate = dropout_rate
        self.max_grad_norm = max_grad_norm
        self.reg_lambda = reg_lambda

        log.info("Initializing parameters")
        self.vocab = 0.1 * np.random.randn(bucket_size, emb_size).astype(np.float32)
        self.weights = None
        self.bias: np.ndarray | None = None
        self.exp_feats: int | None = None
        self.num_columns: list[str] | None = None
        self.mu = None
        self.sigma: np.ndarray | None = None

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

    def forward(self, x):
        log.debug("Forward pass")
        if x.ndim == 1:
            x = x[np.newaxis, :]
        linear = x @ self.weights + self.bias
        exp_shift = np.exp(linear - np.max(linear, axis=1, keepdims=True))
        logits = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return logits

    def loss(self, y_true, y_pred):
        log.debug("Calculating loss")
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.eye(y_pred.shape[1])[y_true]
        ce = np.mean(-np.sum(y_true * np.log(y_pred), axis=1))
        reg = self.reg_lambda * np.sum(np.abs(self.weights))
        return ce + reg

    def fit_scaler(self, df: pd.DataFrame) -> None:
        num_df = df.drop(columns=["text", "label"])
        self.num_columns = sorted(num_df.columns)
        num = num_df.reindex(self.num_columns, axis=1).to_numpy(dtype=np.float32)
        self.mu = num.mean(axis=0, keepdims=True)
        self.sigma = np.clip(num.std(axis=0, keepdims=True), 1e-6, None)

    def _standardize(self, mat: np.ndarray):
        assert self.sigma is not None
        return (mat - self.mu) / np.maximum(self.sigma, 1e-6)

    def backward(self, batch: pd.DataFrame):
        log.debug("Backward pass")

        # tokenize & embed text
        text_df = batch["text"]
        embed_mat = np.stack(
            [self.embed(toks) for toks in text_df], axis=0
        )  # (B, emb_size)

        # extract numeric features
        num_df = batch.drop(columns=["text", "label"])
        assert num_df.dtypes.apply(
            lambda dt: np.issubdtype(dt, np.number)
        ).all(), f"Non-numeric features found: { num_df.columns[~num_df.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))].to_list()}"

        if self.num_columns is None:
            self.num_columns = sorted(num_df.columns)
        num_mat = num_df.reindex(self.num_columns, axis=1, fill_value=0).to_numpy(
            dtype=np.float32
        )
        num_mat = self._standardize(num_mat)

        # init weights & bias if needed
        if self.weights is None:
            input_size = self.emb_size + num_mat.shape[1]
            self.exp_feats = num_mat.shape[1]
            assert self.exp_feats is not None
            scale = np.sqrt(2.0 / (input_size + self.num_class))
            self.weights = (
                np.random.randn(input_size, self.num_class).astype(np.float32) * scale
            )
            self.bias = np.zeros(self.num_class, dtype=np.float32)
            assert self.bias is not None
        else:
            assert (
                num_mat.shape[1] == self.exp_feats
            ), f"Expected {self.exp_feats} numeric columns, got {num_mat.shape[1]}"

        # fuse features
        x = np.concatenate([embed_mat, num_mat], axis=1)
        x = self.apply_dropout(x)

        # forward & loss
        y_pred = self.forward(x)
        loss = self.loss(batch["label"].values, y_pred)
        self._loss_vals.append(loss)

        # backward pass
        b = x.shape[0]
        labels = batch["label"].to_numpy(dtype=int)
        y_true_vec = np.eye(y_pred.shape[1])[labels]
        dl_dz = (y_pred - y_true_vec) / b

        # gradient w.r.t. weights and bias
        assert self.weights is not None
        dl_dw = x.T @ dl_dz + self.reg_lambda * np.sign(self.weights)

        gnorm = np.linalg.norm(dl_dw)
        if gnorm > self.max_grad_norm:
            dl_dw *= self.max_grad_norm / gnorm
            log.debug(f"grad clipped to {self.max_grad_norm}")

        self.weights -= self.lr * dl_dw
        self.bias -= self.lr * np.sum(dl_dz, axis=0)

        # gradient w.r.t. embeddings (fixed update)
        dl_dx = dl_dz @ self.weights.T
        for token_ids, grad in zip(text_df, dl_dx[:, : self.emb_size]):
            ngrams = [ng for word in token_ids for ng in word]
            if not ngrams:
                continue
            share = self.lr * grad / len(ngrams)
            self.vocab[ngrams] -= share

    def eval(self, val_data: pd.DataFrame):
        embed_mat = np.stack(val_data["text"].apply(self.embed).values, axis=0)

        # numeric features
        num_mat = (
            val_data.drop(columns=["text", "label"])
            .reindex(self.num_columns, axis=1, fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        num_mat = self._standardize(num_mat)

        # fuse
        x = np.concatenate([embed_mat, num_mat], axis=1)
        y_pred = self.forward(x)
        loss_val = self.loss(val_data["label"].values, y_pred)
        self._loss_test_vals.append(loss_val)
        return loss_val

    def predict(self, tokens):
        probs = self.forward(self.embed(tokens))
        return probs.flatten()


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)  # reproducibility

    # define texts
    positive_texts = [
        "i love this",
        "this is amazing",
        "really great work",
        "fantastic effort",
        "absolutely wonderful",
        "i really liked it",
        "superb execution",
        "brilliant result",
        "top notch job",
        "impressive work",
        "everything is perfect",
        "nailed it",
        "great outcome",
        "this is excellent",
        "outstanding performance",
    ]
    negative_texts = [
        "i hate this",
        "this is terrible",
        "really bad job",
        "horrible result",
        "absolutely awful",
        "i really disliked it",
        "poor execution",
        "disappointing work",
        "low quality",
        "very sloppy",
        "this sucks",
        "worst ever",
        "not good",
        "this is garbage",
        "total failure",
    ]

    # auto-generate labels
    train_texts = positive_texts + negative_texts
    train_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    # generate numeric features correlated with label
    feat1 = rng.normal(loc=np.array(train_labels) * 2 - 1, scale=0.5)
    feat2 = rng.normal(loc=np.array(train_labels) * -2 + 1, scale=0.5)

    train_df = pd.DataFrame(
        {
            "text": train_texts,
            "label": train_labels,
            "feat1": feat1,
            "feat2": feat2,
        }
    )

    # validation set
    val_positive = [
        "i love it",
        "this is fantastic",
        "very well done",
        "high quality work",
        "i'm impressed",
    ]
    val_negative = [
        "hate this thing",
        "this is trash",
        "completely useless",
        "very bad job",
        "i'm disappointed",
    ]

    val_texts = val_positive + val_negative
    val_labels = [1] * len(val_positive) + [0] * len(val_negative)
    vf1 = rng.normal(loc=np.array(val_labels) * 2 - 1, scale=0.5)
    vf2 = rng.normal(loc=np.array(val_labels) * -2 + 1, scale=0.5)

    val_df = pd.DataFrame(
        {
            "text": val_texts,
            "label": val_labels,
            "feat1": vf1,
            "feat2": vf2,
        }
    )

    tok = FastTextTokenizer(bucket_size=200_000, ngram_size=3)
    tok.build_subword_mask(train_df["text"])

    model = FastTextModel(
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
            batch_df = pd.DataFrame(
                {
                    "text": [train_tokens[i] for i in batch_idx],
                    "label": train_df["label"].values[batch_idx],
                    "feat1": train_df["feat1"].values[batch_idx],
                    "feat2": train_df["feat2"].values[batch_idx],
                }
            )
            model.backward(batch_df)

        train_loss = float(np.mean(model._loss_vals))

        val_batch = pd.DataFrame(
            {
                "text": val_tokens,
                "label": val_df["label"].values,
                "feat1": val_df["feat1"].values,
                "feat2": val_df["feat2"].values,
            }
        )
        vl = model.eval(val_batch)
        log.info(f"Epoch {ep} train_loss={train_loss:.4f} val_loss={vl:.4f}")
