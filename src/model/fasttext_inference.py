import json
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from config import ROOT_DIR


def inference(html: str) -> np.ndarray:
    """
    Given raw HTML html, compute text + numeric features internally,
    then return class probability vector.
    """

    re_tok = re.compile(r"\w+|[^\w\s]+")
    allowed_lengths = {4, 5, 6, 7, 8, 9, 10}
    allowed_tokens = [
        "onee",
        "rdle",
        "reduction",
        "efits",
        "ssic",
        "citizens",
        "ideas",
        "unlike",
        "ueak",
        "aked",
        "bark",
        "loak",
        "udic",
        "myste",
        "eekl",
        "oten",
        "obal",
        "cerem",
        "eeds",
        "arli",
        "auty",
        "research",
        "bann",
        "governor",
        "ikel",
        "regis",
        "sparked",
        "generous",
        "ered",
        "etal",
        "efor",
        "ghes",
        "epit",
        "ility",
        "dynam",
        "vente",
        "oache",
        "nuin",
        "democratic",
        "payw",
        "cono",
        "passi",
    ]
    num_columns = [
        "as_i_x_i_will_y",
        "i_x_that_is_not_y_but_z",
        "iframe_count",
        "inline_css_ratio",
        "links_per_kb",
        "markup_to_text_ratio",
        "prp_ratio",
        "sentences_per_paragraph",
        "stopword_ratio",
        "straight_apostrophe",
        "type_token_ratio",
        "vbg",
    ]
    with open(f"{ROOT_DIR}/src/model/weights.json", encoding="utf-8") as f:
        weights = json.load(f)
        weight_names = ["W_num", "bias", "U", "mu", "sigma"]
        W_num, bias, U_lst, mu, sigma = (weights[elem] for elem in weight_names)
        W_num, bias, mu, sigma = (
            np.array(weights[w]) for w in weight_names if w != "U"
        )
        U = {k: np.array(v) for k, v in U_lst.items()}

    tokens = re_tok.findall(html.lower())
    ngrams_per_word: list[list[int]] = []
    embs = []
    for word in tokens:
        for L in allowed_lengths:
            if len(word) < L:
                continue
            for i in range(len(word) - L + 1):
                sub = word[i : i + L]
                if sub in allowed_tokens:
                    embs.append(U[sub])
        ngrams_per_word.append(embs)

    word_scores: list[np.ndarray] = []
    for embs in ngrams_per_word:
        if embs:
            word_scores.append(np.mean(embs, axis=0))
        else:
            word_scores.append(np.zeros(embs[0].shape[1], dtype=np.float32))
    if word_scores:
        text_score = np.mean(np.stack(word_scores, axis=0), axis=0)
    else:
        text_score = np.zeros(embs[0].shape[1], dtype=np.float32)

    feats = _feature_dict(html)
    num_vec = np.array([feats.get(col, 0.0) for col in num_columns], dtype=np.float32)
    num_std = (num_vec - mu.reshape(-1)) / sigma.reshape(-1)

    numeric_score = num_std @ W_num

    logits = text_score + numeric_score + bias

    exp_shift = np.exp(logits - np.max(logits))
    probs = exp_shift / np.sum(exp_shift)
    return probs


def interpretability(html: str) -> str:
    re_tok = re.compile(r"\w+|[^\w\s]+")
    allowed_lengths = {4, 5, 6, 7, 8, 9, 10}
    allowed_tokens = [
        "onee",
        "rdle",
        "reduction",
        "efits",
        "ssic",
        "citizens",
        "ideas",
        "unlike",
        "ueak",
        "aked",
        "bark",
        "loak",
        "udic",
        "myste",
        "eekl",
        "oten",
        "obal",
        "cerem",
        "eeds",
        "arli",
        "auty",
        "research",
        "bann",
        "governor",
        "ikel",
        "regis",
        "sparked",
        "generous",
        "ered",
        "etal",
        "efor",
        "ghes",
        "epit",
        "ility",
        "dynam",
        "vente",
        "oache",
        "nuin",
        "democratic",
        "payw",
        "cono",
        "passi",
    ]
    num_columns = [
        "as_i_x_i_will_y",
        "i_x_that_is_not_y_but_z",
        "iframe_count",
        "inline_css_ratio",
        "links_per_kb",
        "markup_to_text_ratio",
        "prp_ratio",
        "sentences_per_paragraph",
        "stopword_ratio",
        "straight_apostrophe",
        "type_token_ratio",
        "vbg",
    ]
    with open(f"{ROOT_DIR}/src/model/weights.json", encoding="utf-8") as f:
        weights = json.load(f)
        weight_names = ["W_num", "bias", "U", "mu", "sigma"]
        W_num, bias, U_lst, mu, sigma = (weights[elem] for elem in weight_names)
        W_num, bias, mu, sigma = (
            np.array(weights[w]) for w in weight_names if w != "U"
        )
        U = {k: np.array(v) for k, v in U_lst.items()}

    tokens = re_tok.findall(html.lower())
    matched_subs = []
    word_scores = []
    for word in tokens:
        embs = []
        subs_for_word = []
        for L in allowed_lengths:
            if len(word) < L:
                continue
            for i in range(len(word) - L + 1):
                sub = word[i : i + L]
                if sub in allowed_tokens:
                    embs.append(U[sub])
                    subs_for_word.append(sub)
        if subs_for_word:
            matched_subs.extend(set(subs_for_word))
            word_scores.append(np.mean(embs, axis=0))
        else:
            word_scores.append(np.zeros(2, dtype=np.float32))
    text_score = (
        np.mean(np.stack(word_scores, axis=0), axis=0)
        if word_scores
        else np.zeros(2, dtype=np.float32)
    )
    feats = _feature_dict(html)
    num_vec = np.array([feats.get(col, 0.0) for col in num_columns], dtype=np.float32)
    num_std = (num_vec - mu.reshape(-1)) / sigma.reshape(-1)
    numeric_score = num_std @ W_num
    contribs = {
        col: abs(num_std[i] * (W_num[i, 1] - W_num[i, 0]))
        for i, col in enumerate(num_columns)
    }
    top_numeric = sorted(contribs, key=lambda k: contribs[k], reverse=True)[:5]
    logits = text_score + numeric_score + bias
    exp_shift = np.exp(logits - np.max(logits))
    probs = exp_shift / np.sum(exp_shift)
    cleaned = _RX_SCRIPT_STYLE.sub("", html)
    text_only = _RX_TAG.sub(" ", cleaned)
    feature_map = {
        "as_i_x_i_will_y": "Phrases with the structure 'As I …, I will …'",
        "i_x_that_is_not_y_but_z": "Phrases with the structure 'I … that is not …, but …'",
        "iframe_count": "Contains <iframe> elements",
        "inline_css_ratio": "Uses lots of inline CSS styling",
        "links_per_kb": "Has many hyperlinks",
        "markup_to_text_ratio": "High markup-to-text proportion",
        "prp_ratio": "Uses personal pronouns",
        "sentences_per_paragraph": "Multiple sentences per paragraph",
        "stopword_ratio": "High use of common words",
        "straight_apostrophe": "Contains straight apostrophes",
        "type_token_ratio": "Diverse vocabulary",
        "vbg": "Contains words ending in -ing",
    }
    pattern_matches = {
        "as_i_x_i_will_y": "('"
        + "', '".join(EXPRS["as_i_x_i_will_y"].findall(text_only)[:3])
        + "')",
        "i_x_that_is_not_y_but_z": "('"
        + "', '".join(EXPRS["i_x_that_is_not_y_but_z"].findall(text_only)[:3])
        + "')",
    }

    verdict = "slop" if probs[1] > probs[0] else "not slop"
    lines = f"I think this is more likely {verdict} (P(not-slop)={probs[0]:.2f}, P(slop)={probs[1]:.2f}).\n"
    active_numeric = [
        feature_map[f] + pattern_matches.get(f, "")
        for f in top_numeric
        if feats.get(f, 0) != 0
    ]

    if active_numeric:
        lines += "\nReasoning: " + "".join([f"\n- {an}" for an in active_numeric])
    if matched_subs:
        unique_subs = sorted(set(matched_subs))
        lines += "\n- Contains n-grams like " + ", ".join(
            [f"'{s}'" for s in unique_subs]
        )
    return lines


STOPWORDS = {
    "the",
    "and",
    "is",
    "in",
    "it",
    "of",
    "to",
    "a",
    "with",
    "that",
    "for",
    "on",
    "as",
    "are",
    "this",
    "but",
    "be",
    "at",
    "or",
    "by",
    "an",
    "if",
    "from",
    "about",
    "into",
    "over",
    "after",
    "under",
}

_RX_SCRIPT_STYLE = re.compile(
    r"<(?:script|style)[^>]*>.*?</(?:script|style)>", re.S | re.I
)
_RX_TAG = re.compile(r"<[^>]+>")
_RX_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_RX_PARAGRAPH = re.compile(r"\n{2,}")
_RX_TOKENS = re.compile(r"\w+")
_RX_TAG_NAME = re.compile(r"<\s*(\w+)", re.I)
_RX_IFRAME = re.compile(r"<\s*iframe\b", re.I)
_RX_LINK = re.compile(r'href=["\']([^"\']+)["\']', re.I)

EXPRS = {
    "i_x_that_is_not_y_but_z": re.compile(
        r"\bI\s+\w+\s+that\s+is\s+not\s+\w+,\s*but\s+\w+", re.I
    ),
    "as_i_x_i_will_y": re.compile(r"\bAs\s+I\s+\w+,\s*I\s+will\s+\w+", re.I),
}


@lru_cache(maxsize=2**14)
def _feature_dict(html: str) -> dict[str, float]:
    cleaned = _RX_SCRIPT_STYLE.sub("", html)
    text = _RX_TAG.sub(" ", cleaned)
    tokens = _RX_TOKENS.findall(text.lower())
    [s for s in _RX_SENTENCE_SPLIT.split(text) if s.strip()]
    paragraphs = [p for p in _RX_PARAGRAPH.split(text) if p.strip()]
    total_bytes, text_bytes = len(html), len(text)
    tags = _RX_TAG_NAME.findall(html.lower())
    n_tags = len(tags) or 1
    iframe_count = len(_RX_IFRAME.findall(html))
    hrefs = _RX_LINK.findall(html)
    total_links = len(hrefs)
    links_per_kb = total_links / (total_bytes / 1024) if total_bytes else 0
    sw_count = sum(1 for t in tokens if t in STOPWORDS)
    stopword_ratio = sw_count / len(tokens) if tokens else 0
    spp_list = [len(_RX_SENTENCE_SPLIT.split(p)) for p in paragraphs]
    sentences_per_paragraph = sum(spp_list) / len(spp_list) if spp_list else 0
    freq = Counter(tokens)
    type_token_ratio = len(freq) / len(tokens) if tokens else 0
    prp_count = len(
        re.findall(r"\b(?:I|me|you|he|she|it|we|they|him|her|us|them)\b", text, re.I)
    )
    prp_ratio = prp_count / len(tokens) if tokens else 0
    vbg_count = len(re.findall(r"\b\w+ing\b", text))
    straight_apostrophe = text.count("'")
    markup_to_text_ratio = (
        (total_bytes - text_bytes) / total_bytes if total_bytes else 0
    )
    inline_css_ratio = html.lower().count("style=") / n_tags
    ix_not = len(EXPRS["i_x_that_is_not_y_but_z"].findall(text))
    as_i = len(EXPRS["as_i_x_i_will_y"].findall(text))
    return {
        "stopword_ratio": stopword_ratio,
        "links_per_kb": links_per_kb,
        "type_token_ratio": type_token_ratio,
        "i_x_that_is_not_y_but_z": ix_not,
        "prp_ratio": prp_ratio,
        "sentences_per_paragraph": sentences_per_paragraph,
        "markup_to_text_ratio": markup_to_text_ratio,
        "inline_css_ratio": inline_css_ratio,
        "iframe_count": iframe_count,
        "as_i_x_i_will_y": as_i,
        "vbg": vbg_count,
        "straight_apostrophe": straight_apostrophe,
    }


if __name__ == "__main__":
    tests = [
        (
            "Minimal HTML (no tokens)",
            "<html><body><p>Just plain text.</p></body></html>",
        ),
        (
            "Contains allowed token 'research'",
            "<html><body><p>Our research initiative begins now.</p></body></html>",
        ),
        (
            "Triggers EXPRS 'i_x_that_is_not_y_but_z'",
            "<html><body><p>I data that is not flawed, but exemplary.</p></body></html>",
        ),
        (
            "Triggers EXPRS 'as_i_x_i_will_y'",
            "<html><body><p>As I write, I will explore new ideas.</p></body></html>",
        ),
        (
            "Heavy numeric features: iframe, inline CSS, links, apostrophes, -ing words, multiple paragraphs",
            """
            <html>
              <head><style>body {background: #fff;}</style></head>
              <body>
                <div style="color: blue;">Testing inline CSS and an apostrophe here: It's working.</div>
                <iframe src="https://example.com"></iframe>
                <p>First paragraph has two sentences. Second sentence here.</p>

                <p>Second paragraph with a <a href="http://example.com">link</a> and another link.</p>
                <p>Ending with a VBG word: testing.</p>
              </body>
            </html>
            """,
        ),
    ]

    for desc, html in tests:
        interpr = interpretability(html)
        print(f"[{desc}]\n\n\n{interpr}\n\n{'-'*50}\n")
