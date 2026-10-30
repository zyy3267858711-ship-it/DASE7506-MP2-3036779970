"""NumPy-only CPU inference for the small candidate-ranking ensemble."""

import json
from pathlib import Path

import numpy as np

from common import parse_answer


def make_features(bank, arrays):
    e = np.asarray(arrays["embeddings"], dtype=np.float32)

    if e.shape != (len(bank), 4, 1024):
        raise ValueError("Expected embeddings [N, 4, 1024]")

    # Absolute representation.
    e = e / np.maximum(
        np.linalg.norm(e, axis=2, keepdims=True), 1e-8
    )

    # Representation relative to the other candidates in this question.
    relative = e - e.mean(axis=1, keepdims=True)

    extras = []
    for row in bank:
        texts = [candidate["text"] for candidate in row["candidates"]]
        answers = [parse_answer(text) for text in texts]

        for text, answer in zip(texts, answers):
            agreement = (
                sum(other == answer for other in answers) / 4.0
                if answer is not None
                else 0.0
            )

            extras.append([
                float(answer is not None),
                agreement,
                np.log1p(len(text)),
                np.log1p(len(text.splitlines())),
                np.log1p(text.count("=")),
            ])

    extras = np.asarray(extras, dtype=np.float32)
    extras = extras.reshape(len(bank), 4, 5)

    return np.concatenate([e, relative, extras], axis=2)


def member_logits(x, path):
    """Inference equivalent to Linear -> ReLU -> Linear."""
    with np.load(path, allow_pickle=False) as model:
        z = (x - model["mean"]) / model["scale"]
        hidden = np.maximum(z @ model["w1"] + model["b1"], 0)
        logits = (hidden @ model["w2"] + model["b2"])[..., 0]

    return logits


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def ensemble_probabilities(x, folder, members):
    result = np.zeros(x.shape[:2], dtype=np.float32)

    for filename in members:
        result += softmax(member_logits(x, Path(folder) / filename))

    return result / len(members)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Pass --checkpoint runs/mlp/selected")

    checkpoint = Path(checkpoint)
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    x = make_features(bank, arrays)
    probabilities = ensemble_probabilities(
        x, checkpoint, config["members"]
    )

    return [
        int(index)
        for index in np.argmax(probabilities, axis=1)
    ]