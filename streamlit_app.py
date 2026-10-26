"""Streamlit demo for the Khmer LSTM next-word model (launched from script.ipynb, section 12).

Self-contained: loads the checkpoint saved by the notebook (weights + config + vocab) and repeats the
notebook's clean -> segment -> encode pipeline. Set KHMER_LM_CKPT to the checkpoint path.
"""
import os
import re
import unicodedata

import streamlit as st
import torch
import torch.nn as nn
from khmernltk import word_tokenize

CKPT = os.environ.get("KHMER_LM_CKPT", "artifacts/checkpoints_n300/best.pt")
PAD, UNK, EOS, NUM = "<pad>", "<unk>", "<eos>", "<num>"

SENT_SPLIT_RE = re.compile(r"[\n។៕?!]+")
INVISIBLE_RE = re.compile(r"[‌‍﻿­឴឵]")
KHMER_PUNCT_RE = re.compile(r"[៖៘-៚]")
NON_KHMER_RE = re.compile(r"[^ក-៿᧠-᧿0-9\s]+")
KHMER_CHAR_RE = re.compile(r"[ក-៿᧠-᧿]")
DIGIT_RE = re.compile(r"[0-9០-៩]+")
WS_RE = re.compile(r"\s+")


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("​", " ")
    text = INVISIBLE_RE.sub("", text)
    out = []
    for chunk in SENT_SPLIT_RE.split(text):
        chunk = KHMER_PUNCT_RE.sub(" ", NON_KHMER_RE.sub(" ", chunk))
        chunk = WS_RE.sub(" ", chunk).strip()
        if KHMER_CHAR_RE.search(chunk):
            out.append(chunk)
    return out


def segment_sentence(sentence):
    tokens = []
    for tok in word_tokenize(sentence, return_tokens=True):
        tok = tok.strip()
        if not tok:
            continue
        if DIGIT_RE.fullmatch(tok):
            if tokens and tokens[-1] == NUM:
                continue
            tok = NUM
        tokens.append(tok)
    return tokens


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, tie_weights, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim, self.proj = hidden_dim, None
        if tie_weights and hidden_dim != embed_dim:
            self.proj = nn.Linear(hidden_dim, embed_dim)
            out_dim = embed_dim
        self.decoder = nn.Linear(out_dim, vocab_size)
        if tie_weights:
            self.decoder.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.drop(self.embedding(x)), hidden)
        out = self.drop(out)
        if self.proj is not None:
            out = self.proj(out)
        return self.decoder(out), hidden


@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    c, itos = ckpt["config"], ckpt["vocab"]
    stoi = {w: i for i, w in enumerate(itos)}
    model = LSTMLanguageModel(len(itos), c["embed_dim"], c["hidden_dim"], c["num_layers"],
                              c["dropout"], c["tie_weights"], stoi[PAD])
    model.load_state_dict(ckpt["model"])
    model.eval()
    blocked = [stoi[t] for t in (PAD, UNK, EOS, NUM) if t in stoi]
    return model, itos, stoi, blocked


@torch.no_grad()
def predict(text, k=5, max_context=64):
    model, itos, stoi, blocked = load_model()
    ids = [stoi[EOS]]
    sentences = [s for s in (segment_sentence(s) for s in clean_text(text)) if s]
    ended = text.rstrip()[-1:] in ("។", "៕", "?", "!", "\n") if text.strip() else False
    for i, toks in enumerate(sentences):
        ids.extend(stoi.get(t, stoi[UNK]) for t in toks)      # unknown words -> <unk> (FR3)
        if i < len(sentences) - 1 or ended:
            ids.append(stoi[EOS])
    logits, _ = model(torch.tensor([ids[-max_context:]]))
    probs = logits[0, -1].softmax(-1)
    probs[blocked] = 0.0
    top = probs.topk(k)
    return [(itos[i], p) for i, p in zip(top.indices.tolist(), top.values.tolist())]


st.set_page_config(page_title="Khmer Next-Word Prediction", page_icon="🇰🇭")
st.title("Khmer Next-Word Prediction")
st.caption("LSTM language model trained on Khmer Wikipedia. Type Khmer text and the suggestions update as you type.")

try:                                   # per-keystroke updates if the helper package is installed
    from st_keyup import st_keyup
    text = st_keyup("Khmer text", placeholder="ខ្ញុំចង់ទៅ...", debounce=150)
except ImportError:                    # otherwise suggestions update on Enter / when focus leaves the box
    text = st.text_input("Khmer text", placeholder="ខ្ញុំចង់ទៅ...")

k = st.slider("Number of suggestions", 1, 10, 5)
try:
    preds = predict(text or "", k=k)
except Exception as e:                 # never crash the demo on odd input
    st.error(f"{type(e).__name__}: {e}")
else:
    st.subheader("Next word" if text else "Sentence-initial words")
    for w, p in preds:
        st.markdown(f"**{w}**")
        st.progress(min(max(float(p), 0.0), 1.0), text=f"{p:.1%}")
