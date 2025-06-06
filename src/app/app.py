import json
import os
import re
from collections import Counter
from typing import Any

import gradio as gr
import numpy as np
import requests

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


def _feature_dict(html: str) -> dict:
    cleaned = _RX_SCRIPT_STYLE.sub("", html)
    text = _RX_TAG.sub(" ", cleaned)
    tokens = _RX_TOKENS.findall(text.lower())
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


def load_weights():
    with open(
        os.path.join(os.path.dirname(__file__), "weights.json"), encoding="utf-8"
    ) as f:
        weights = json.load(f)
        weight_names = ["W_num", "bias", "U", "mu", "sigma"]
        w_num, bias, u_lst, mu, sigma = (weights[elem] for elem in weight_names)
        w_num, bias, mu, sigma = (
            np.array(weights[w]) for w in weight_names if w != "U"
        )
        u = {k: np.array(v) for k, v in u_lst.items()}
    return w_num, bias, u, mu, sigma


def interpretability_viz(html: str):
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
    w_num, bias, u, mu, sigma = load_weights()
    tokens = re_tok.findall(html.lower())
    matched_subs: list[str] = []

    word_scores = []
    emb_dim = next(iter(u.values())).shape[-1] if u else 2
    for word in tokens:
        embs = []
        subs_for_word = []
        for length in allowed_lengths:
            if len(word) < length:
                continue
            for i in range(len(word) - length + 1):
                sub = word[i : i + length]
                if sub in allowed_tokens:
                    embs.append(u[sub])
                    subs_for_word.append(sub)
        if subs_for_word:
            matched_subs.extend(set(subs_for_word))
            word_scores.append(np.mean(embs, axis=0))
        else:
            word_scores.append(np.zeros(emb_dim, dtype=np.float32))
    text_score = (
        np.mean(np.stack(word_scores, axis=0), axis=0)
        if word_scores
        else np.zeros(emb_dim, dtype=np.float32)
    )
    feats = _feature_dict(html)
    num_vec = np.array([feats.get(col, 0.0) for col in num_columns], dtype=np.float32)
    num_std = (num_vec - mu.reshape(-1)) / sigma.reshape(-1)
    numeric_score = num_std @ w_num
    logits = text_score + numeric_score + bias
    exp_shift = np.exp(logits - np.max(logits))
    probs = exp_shift / np.sum(exp_shift)

    feature_info = []
    for i, col in enumerate(num_columns):
        delta = w_num[i, 1] - w_num[i, 0]
        cval = num_std[i] * delta
        abs_cval = abs(cval)
        direction = cval > 0  # True = slop, False = not-slop
        feature_info.append(
            {
                "col": col,
                "value": feats.get(col, 0),
                "abs_cval": abs_cval,
                "direction": direction,
                "cval": cval,
            }
        )

    verdict = "slop" if probs[1] > probs[0] else "not slop"
    for f in feature_info:
        f["signed"] = (
            f["abs_cval"] if f["direction"] == (verdict == "slop") else -f["abs_cval"]
        )
    feature_info.sort(key=lambda x: x["signed"], reverse=True)
    feature_info = feature_info[:5]

    feature_map = {
        "as_i_x_i_will_y": "Phrases: <b>'As I …, I will …'</b>",
        "i_x_that_is_not_y_but_z": "Phrases: <b>'I … that is not …, but …'</b>",
        "iframe_count": "Contains &lt;iframe&gt; elements",
        "inline_css_ratio": "Uses lots of inline CSS styling",
        "links_per_kb": "Has many hyperlinks",
        "markup_to_text_ratio": "High markup-to-text proportion",
        "prp_ratio": "Uses personal pronouns",
        "sentences_per_paragraph": "Multiple sentences per paragraph",
        "stopword_ratio": "High use of common words",
        "straight_apostrophe": "Contains straight apostrophes",
        "type_token_ratio": "Diverse vocabulary",
        "vbg": "Contains words ending in <b>-ing</b>",
    }
    cleaned = _RX_SCRIPT_STYLE.sub("", html)
    text_only = _RX_TAG.sub(" ", cleaned)
    pattern_matches = {
        "as_i_x_i_will_y": "('"
        + "', '".join(EXPRS["as_i_x_i_will_y"].findall(text_only)[:3])
        + "')",
        "i_x_that_is_not_y_but_z": "('"
        + "', '".join(EXPRS["i_x_that_is_not_y_but_z"].findall(text_only)[:3])
        + "')",
    }

    def feat_color(strength, direction, max_strength):
        if max_strength <= 0:
            return "background:#fffde7;color:#333;"
        norm = min(strength / max_strength, 1.0)
        yellow, red, green = (227, 213, 123), (196, 70, 67), (92, 173, 95)
        if direction:
            r, g, b = (y + (norm * (r - y)) for y, r in zip(yellow, red))
        else:
            r, g, b = (y + (norm * (g - y)) for y, g in zip(yellow, green))
        return f"background:rgb({r},{g},{b});color:#111;"

    top_feats_table = (
        "<table style='border-collapse:collapse;width:100%;margin-bottom:12px;'>"
    )
    top_feats_table += "<tr><th style='padding:4px 8px;text-align:center;'>Top Features</th><th style='padding:4px 8px;text-align:center;'>Value</th></tr>"

    tot_abs = sum(f["abs_cval"] for f in feature_info) or 1.0
    for f in feature_info:
        f["norm01"] = f["abs_cval"] / tot_abs

    for feat in feature_info:
        feat_col = feat["col"]
        human = feature_map[feat_col]
        extra = pattern_matches.get(feat_col, "") if "Phrases" in human else ""
        color = feat_color(
            feat["abs_cval"],
            feat["direction"],
            max(f["abs_cval"] for f in feature_info),
        )
        sign = "+" if feat["signed"] > 0 else "-"
        cell = f"{sign}{abs(feat['norm01']):.2f}"
        if cell[1:] != "0.00":
            top_feats_table += (
                f"<tr style='{color}'>"
                f"<td style='padding:4px 8px;'>{human}{extra}</td>"
                f"<td style='padding:4px 8px;text-align:right;'>{cell}</td>"
                f"</tr>"
            )

    def verdict_button(verdict):
        if verdict == "not slop":
            return "<button style='background:#43a047;color:white;font-weight:800;font-size:1.2em;padding:16px 32px;border-radius:10px;border:none;margin-bottom:14px;box-shadow:0 2px 8px #1111;'>NOT SLOP</button>"
        else:
            return "<button style='background:#e53935;color:white;font-weight:800;font-size:1.2em;padding:16px 32px;border-radius:10px;border:none;margin-bottom:14px;box-shadow:0 2px 8px #1111;'>SLOP</button>"

    ngram_html = ""
    if matched_subs:
        unique_subs = sorted(set(matched_subs))
        subs_info: list[dict[str, Any]] = []
        for s in unique_subs:
            emb = u.get(s, np.zeros(emb_dim, dtype=np.float32))
            delta_sub = float(emb[1] - emb[0])
            abs_delta = abs(delta_sub)
            direction_sub = delta_sub > 0
            subs_info.append(
                {
                    "sub": s,
                    "score": delta_sub,
                    "abs_score": abs_delta,
                    "direction": direction_sub,
                }
            )

        subs_info.sort(key=lambda x: x["abs_score"], reverse=True)
        subs_info = subs_info[:5]

        for s_i in subs_info:
            s_i["signed"] = (
                s_i["abs_score"]
                if s_i["direction"] == (verdict == "slop")
                else -s_i["abs_score"]
            )
        subs_info.sort(key=lambda x: x["signed"], reverse=True)

        max_abs_sub = max(s["abs_score"] for s in subs_info) or 1.0
        ngram_html = "<div style='margin:8px 0;'>Matched n-grams:<br>"
        for s_i in subs_info:
            color = feat_color(s_i["abs_score"], s_i["direction"], max_abs_sub)
            sign = "+" if s_i["signed"] > 0 else "-"
            ngram_html += (
                f"<span style='{color} border-radius:4px; padding:2px 5px; margin:2px; display:inline-block; font-family:monospace;'>"
                f"{sign}{s_i['sub']}"
                f"</span>"
            )
        ngram_html += "</div>"

    overall = f"""
    <div style='padding:18px; background:#fff; border-radius:16px; box-shadow:0 2px 8px #0001;'>
      <div style='text-align:center;'>{verdict_button(verdict)}</div>
      {top_feats_table}
      {ngram_html}
    </div>
    """
    return overall


def process_input_viz(url_input, html_input):
    user_input = (url_input or "").strip()
    html = (html_input or "").strip()
    if user_input:
        try:
            resp = requests.get(user_input, timeout=6)
            html = resp.text
        except Exception as e:
            return f"<span style='color:red;'>Error fetching URL: {e}</span>"
    elif html:
        pass
    else:
        return "<span style='color:red;'>Please provide a URL or HTML code.</span>"
    return interpretability_viz(html)


desc = (
    "This is a demo for Stop-Slop, an AI model that detects slop "
    "(low-quality, unoriginal, or spammy material —often AI-generated— "
    "that adds noise rather than value) websites.\n"
    "\n\n\n"
    "To start, input a <b>valid URL (top box)</b> <span style='color:#888;"
    "'>or</span> some <b>HTML code (bottom box)</b>."
)

iface = gr.Interface(
    fn=process_input_viz,
    inputs=[
        gr.Textbox(
            lines=1,
            label="URL",
            placeholder="https://nymag.com/intelligencer/article/ai-generated-content-internet-online-slop-spam.html",
        ),
        gr.Textbox(lines=10, label="HTML", placeholder="<html>...</html>"),
    ],
    outputs=gr.HTML(label="Result"),
    description=desc,
    title="🚫🧟 Stop Slop",
)

if __name__ == "__main__":
    iface.launch()
