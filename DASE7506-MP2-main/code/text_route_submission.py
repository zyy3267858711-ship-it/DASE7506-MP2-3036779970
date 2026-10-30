"""Hashed question/solution text encoder plus supplied fixed embeddings.

CPU inference uses only NumPy. No pretrained weights or network calls.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from common import parse_answer

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", re.I)
VOCAB_SIZE = 8192
EMBED_DIM = 48


def token_ids(text, answer=None):
    tokens = TOKEN_RE.findall(text.lower())
    if answer is not None:
        tokens += ["<answer>", "<num>"]
        # Keep answer shape/type without making exact answer a lookup key.
        tokens.append("<fraction>" if "/" in answer else "<integer>" if "." not in answer else "<decimal>")
    if not tokens:
        tokens = ["<empty>"]
    # Bound pathological traces while retaining both beginning and ending.
    if len(tokens) > 256:
        tokens = tokens[:128] + tokens[-128:]
    terms = ["w:" + token for token in tokens]
    terms.extend("b:" + a + "\x1f" + b for a, b in zip(tokens, tokens[1:]))
    result = []
    for term in terms:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        result.append(int.from_bytes(digest, "little") % VOCAB_SIZE)
    return result or [0]


def text_sequences(bank):
    questions, candidates = [], []
    for row in bank:
        q = token_ids(row["question"])
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        for candidate, answer in zip(row["candidates"], answers):
            questions.append(q)
            candidates.append(token_ids(candidate["text"], answer))
    return questions, candidates


def bag_mean(ids, table):
    # ids is one token-id list; table is [vocab, dim]
    return table[np.asarray(ids, dtype=np.int64)].mean(axis=0)


def auxiliary(bank, arrays):
    e = np.asarray(arrays["embeddings"], dtype=np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    answers = []
    rows = []
    for question_index, row in enumerate(bank):
        group = [parse_answer(c["text"]) for c in row["candidates"]]
        answers.append(group)
        distinct = len({a for a in group if a is not None})
        sim = e[question_index] @ e[question_index].T
        for j, c in enumerate(row["candidates"]):
            text, ans = c["text"], group[j]
            others = [sim[j, k] for k in range(4) if k != j]
            rows.append([
                float(ans is not None),
                sum(a == ans for a in group) / 4 if ans is not None else 0.0,
                np.log1p(len(text)), np.log1p(len(text.splitlines())),
                np.log1p(text.count("=")), distinct / 4,
                float(max(others)), float(min(others)),
            ])
    return e, np.asarray(rows, dtype=np.float32).reshape(len(bank), 4, 8), answers


def candidate_features(bank, arrays, table, mean, scale):
    qseq, cseq = text_sequences(bank)
    q = np.stack([bag_mean(ids, table) for ids in qseq]).reshape(len(bank), 4, EMBED_DIM)
    c = np.stack([bag_mean(ids, table) for ids in cseq]).reshape(len(bank), 4, EMBED_DIM)
    e, extra, answers = auxiliary(bank, arrays)
    fixed = np.concatenate([e, extra], axis=-1)
    fixed = np.clip((fixed-mean)/scale, -10, 10)
    # The word vectors are learned at training time and are not standardized
    # with fixed statistics. Normalize only the fixed 1024+8 dimensional block.
    z = np.concatenate([q, c, np.abs(q-c), q*c, fixed], axis=-1).astype(np.float32)
    return z, answers


def scores_from_features(z, model):
    h = np.maximum(z @ model["w1"] + model["b1"], 0)
    h = np.maximum(h @ model["w2"] + model["b2"], 0)
    return (h @ model["w3"] + model["b3"])[..., 0]


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Pass --checkpoint runs/text_route/selected")
    root = Path(checkpoint)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    with np.load(root / "model.npz", allow_pickle=False) as f:
        model = {key: f[key].copy() for key in f.files}
    z, answers = candidate_features(bank, arrays, model["table"], model["mean"], model["scale"])
    scores = scores_from_features(z, model)
    choices = []
    for row, group, value in zip(bank, answers, scores):
        valid = [j for j, answer in enumerate(group) if answer is not None]
        choices.append(int(max(valid, key=lambda j: float(value[j]))) if valid else int(np.argmax(value)))
    return choices
