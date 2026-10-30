"""Lightweight candidate selection.

Inference uses only the supplied bank, embeddings and saved assets.
"""

import numpy as np

from common import parse_answer


def majority_choices(bank):
    """Choose the first candidate belonging to the most frequent valid answer."""
    choices = []

    for row in bank:
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1

        if not counts:
            choices.append(0)
            continue

        scores = [
            counts.get(answer, 0) if answer is not None else 0
            for answer in answers
        ]
        choices.append(int(np.argmax(scores)))

    return choices


def make_features(bank, arrays, mode):
    """Return one feature row per candidate, preserving bank order."""
    embeddings = np.asarray(
        arrays["embeddings"], dtype=np.float64
    )

    n = len(bank)
    if embeddings.shape != (n, 4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024]")

    # Normalize each candidate vector without using reference answers.
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    x = normalized.reshape(n * 4, 1024)

    if mode == "embedding":
        return x

    if mode != "combined":
        raise ValueError(f"Unknown feature mode: {mode}")

    extra = []

    for row in bank:
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]

        for text, answer in zip(texts, answers):
            agreement = (
                sum(other == answer for other in answers) / 4.0
                if answer is not None
                else 0.0
            )

            extra.append([
                float(answer is not None),
                agreement,
                np.log1p(len(text)),
                np.log1p(len(text.splitlines())),
                np.log1p(text.count("=")),
            ])

    extra = np.asarray(extra, dtype=np.float64).reshape(n * 4, 5)
    return np.concatenate([x, extra], axis=1)


def predict(bank, arrays, checkpoint):
    """Return one built-in Python integer in 0..3 per question."""
    if checkpoint is None:
        return majority_choices(bank)

    with np.load(checkpoint / "model.npz", allow_pickle=False) as saved:
        mode = str(saved["mode"].item())

        if mode == "majority":
            return majority_choices(bank)

        mean = saved["mean"]
        scale = saved["scale"]
        weights = saved["weights"]
        target_mean = float(saved["target_mean"].item())

    x = make_features(bank, arrays, mode)
    scores = ((x - mean) / scale) @ weights + target_mean
    scores = scores.reshape(len(bank), 4)

    return [int(index) for index in np.argmax(scores, axis=1)]