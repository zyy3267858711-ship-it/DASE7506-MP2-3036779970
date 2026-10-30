"""Candidate comparison with a compact ensemble; NumPy-only inference.

AI assistance: OpenAI ChatGPT/Codex assisted with design and implementation.
No reference answers, question IDs, or cached predictions are used at inference.
Self-contained method: no V2 or experiment scripts required. The official
assignment's common.py supplies the unchanged numeric answer parser.
"""

import ast
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

# Support both placement directly in official code/ and code/v3_share/.
_HERE = Path(__file__).resolve().parent
if not (_HERE / "common.py").is_file() and (_HERE.parent / "common.py").is_file():
    sys.path.append(str(_HERE.parent))

from common import parse_answer


def embeddings_unit(embeddings):
    """L2-normalize each supplied candidate embedding."""
    x = np.asarray(embeddings, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def get_answers(bank):
    """Parse final answers using the official numeric parser."""
    return [[parse_answer(candidate["text"]) for candidate in row["candidates"]]
            for row in bank]


def relative_features(embeddings):
    """Concatenate candidate embeddings and within-question differences."""
    relative = embeddings - embeddings.mean(axis=1, keepdims=True)
    return np.concatenate([embeddings, relative], axis=-1)


def aggregate(logits, answers, pool, vote_weight):
    """Pool equal numeric answers, apply vote bonuses and mask invalid answers."""
    scores = np.asarray(logits, dtype=np.float64).copy()
    for i, group in enumerate(answers):
        counts = Counter(answer for answer in group if answer is not None)
        for answer, count in counts.items():
            indices = [j for j, value in enumerate(group) if value == answer]
            if pool:
                scores[i, indices] = np.mean(logits[i, indices])
            scores[i, indices] += vote_weight * np.log(count)
        for j, answer in enumerate(group):
            if answer is None:
                scores[i, j] = -np.inf
    return scores


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
PHRASES = (
    "however", "but", "wait", "alternatively", "assum", "approximately",
    "round", "remaining", "total", "each", "per", "therefore", "check",
    "not", "instead", "equation", "let ", "we need", "double", "half",
)


def arithmetic_value(expression):
    """Evaluate only bounded numeric arithmetic ASTs, never Python code."""
    if len(expression) > 100:
        raise ValueError("Expression too long")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            elif isinstance(node.op, ast.Div):
                value = a / b
            else:
                raise ValueError("Unsupported operator")
        else:
            raise ValueError("Non-numeric expression")
        if not np.isfinite(value) or abs(value) > 1e15:
            raise ValueError("Value out of range")
        return value

    return visit(tree.body)


def arithmetic_checks(text):
    """Conservative checks of explicit numeric equalities in a trace."""
    s = text.replace(",", "").replace("−", "-").replace("×", "*").replace("÷", "/")
    s = re.sub(r"\\(?:times|cdot)", "*", s)
    s = s.replace(r"\div", "/").replace(r"\left", "").replace(r"\right", "")
    for _ in range(3):
        s = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace("$", " ").replace("**", " ")
    good = bad = 0
    pattern = r"([\d.()\s+*/-]+)\s*=\s*([-+]?\d+(?:\.\d+)?)"
    for match in re.finditer(pattern, s):
        lhs = match[1].strip()
        if not re.search(r"\d\s*[+*/-]|\)\s*[+*/-]", lhs):
            continue
        try:
            a, b = arithmetic_value(lhs), float(match[2])
            if abs(a - b) <= max(0.011, 0.001 * abs(b)):
                good += 1
            else:
                bad += 1
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            pass
    return good, bad


def numeric_values(text):
    out = []
    for token in NUMBER.findall(text):
        try:
            value = float(Fraction(token.replace(",", "")))
            if np.isfinite(value):
                out.append(value)
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return out


def text_features(bank):
    """Formatting, answer agreement, and arithmetic evidence; no fitted state."""
    out = []
    for row in bank:
        answers = get_answers([row])[0]
        counts = Counter(a for a in answers if a is not None)
        qnums = numeric_values(row["question"])
        qwords = set(re.findall(r"[a-z]+", row["question"].lower()))
        group = []
        for candidate, answer in zip(row["candidates"], answers):
            text = candidate["text"]
            lower = text.lower()
            nums = numeric_values(text)
            words = re.findall(r"[a-z]+", lower)
            good, bad = arithmetic_checks(text)
            value = float(Fraction(answer)) if answer is not None else 0.0
            log = np.log1p
            features = [
                log(len(text)), log(len(words)), log(text.count("\n")),
                log(len(nums)), log(len(set(nums))), log(text.count("=")),
                log(text.count(r"\frac")), log(text.count(r"\times")),
                log(text.count("$")), log(text.count("**")), log(text.count("#")),
                log(text.count("?")), log(text.count("!")),
                log(good), log(bad), bad / max(1, good + bad),
                float(answer is not None), float(value < 0), float(value == 0),
                float(value == int(value)), np.sign(value) * min(30, log(abs(value))),
                log(counts.get(answer, 0)), log(len(counts)),
                float(any(abs(value - n) < 1e-6 for n in qnums)),
                sum(w in qwords for w in words) / max(1, len(words)),
                log(len(qnums)), log(len(row["question"])),
            ]
            features.extend(log(lower.count(phrase)) for phrase in PHRASES)
            group.append(features)
        out.append(group)
    return np.asarray(out, dtype=np.float32)


def make_features(embeddings, extra=None, mode="relative"):
    if mode == "stats":
        return np.concatenate([extra, extra - extra.mean(axis=1, keepdims=True)], axis=-1)
    e = embeddings_unit(embeddings)
    x = relative_features(e) if mode != "raw" else e
    if mode == "text":
        if extra is None:
            raise ValueError("Text features required")
        x = np.concatenate([x, extra, extra - extra.mean(axis=1, keepdims=True)], axis=-1)
    return x


def tree_logits(x, weights):
    """Vectorized traversal of exported numeric LightGBM trees."""
    x = x.reshape(-1, x.shape[-1])
    out = np.zeros(len(x), dtype=np.float64)
    feature, threshold = weights["feature"], weights["threshold"]
    left, right, value = weights["left"], weights["right"], weights["value"]
    rows = np.arange(len(x))
    for root in weights["roots"]:
        node = np.full(len(x), root, dtype=np.int32)
        while True:
            active = feature[node] >= 0
            if not active.any():
                break
            r = rows[active]
            n = node[active]
            node[active] = np.where(x[r, feature[n]] <= threshold[n], left[n], right[n])
        out += value[node]
    return out.reshape(-1, 4)


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    if checkpoint is None:
        raise ValueError("V3 requires a checkpoint directory")
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    cache = {}
    extra = text_features(bank) if any(m["mode"] in ("text", "stats") for m in config["members"]) else None
    logits = np.zeros((len(bank), 4), dtype=np.float64)
    for member in config["members"]:
        mode = member["mode"]
        if mode not in cache:
            cache[mode] = make_features(arrays["embeddings"], extra, mode)
        with np.load(root / member["file"], allow_pickle=False) as w:
            if member.get("kind", "mlp") == "trees":
                z = tree_logits(cache[mode], w)
            else:
                x = ((cache[mode] - w["mean"]) / w["std"]).reshape(-1, len(w["mean"]))
                hidden = np.maximum(x @ w["w0"] + w["b0"], 0)
                z = (hidden @ w["w1"] + w["b1"]).reshape(-1, 4)
        # Per-question centering cancels irrelevant intercepts before fusion.
        logits += member["weight"] * (z - z.mean(axis=1, keepdims=True))
    scores = aggregate(logits, get_answers(bank), config["pool"], config["vote_weight"])
    return [int(i) for i in scores.argmax(axis=1)]
