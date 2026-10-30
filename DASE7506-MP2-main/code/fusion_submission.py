"""CPU inference: heterogeneous score fusion and answer grouping."""

import json
from pathlib import Path

import numpy as np

from common import parse_answer
from submission import make_features as make_ridge_features
from mlp_submission import (
    make_features as make_mlp_features,
    ensemble_probabilities,
)


def ridge_probabilities(bank, arrays, checkpoint):
    """Convert ridge correctness scores into normalized support scores.

    These are support scores, not calibrated correctness probabilities.
    """
    with np.load(
        Path(checkpoint) / "model.npz", allow_pickle=False
    ) as model:
        mode = str(model["mode"].item())

        if mode == "majority":
            result = np.zeros((len(bank), 4), dtype=np.float64)

            for i, row in enumerate(bank):
                answers = [
                    parse_answer(candidate["text"])
                    for candidate in row["candidates"]
                ]
                for j, answer in enumerate(answers):
                    result[i, j] = (
                        sum(other == answer for other in answers)
                        if answer is not None else 0.0
                    )

            result += 1e-6
            return result / result.sum(axis=1, keepdims=True)

        x = make_ridge_features(bank, arrays, mode)

        scores = (
            ((x - model["mean"]) / model["scale"])
            @ model["weights"]
            + float(model["target_mean"].item())
        ).reshape(len(bank), 4)

    # Ridge was trained against binary correctness labels.
    # Clip out-of-range regression predictions, then normalize.
    support = np.clip(scores, 0.001, 0.999)
    return support / support.sum(axis=1, keepdims=True)


def mlp_probabilities(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    x = make_mlp_features(bank, arrays)

    return ensemble_probabilities(
        x, checkpoint, config["members"]
    )


def answer_choices(bank, support, group_power=None):
    """Choose a candidate directly or aggregate by parsed final answer.

    group_score = sum(candidate support) / group_size ** group_power

    None: no answer grouping
    0.0: sum
    0.5: partial penalty for repeated answers
    1.0: mean
    """
    if group_power is None:
        return [
            int(index)
            for index in np.argmax(support, axis=1)
        ]

    choices = []

    for row, scores in zip(bank, support):
        groups = {}

        for j, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(j)

        if not groups:
            choices.append(int(np.argmax(scores)))
            continue

        best_group = None
        best_score = -float("inf")

        for indices in groups.values():
            score = float(scores[indices].sum())
            score /= len(indices) ** group_power

            if score > best_score:
                best_score = score
                best_group = indices

        # Any candidate with the chosen answer receives the same final grade.
        chosen = max(best_group, key=lambda j: float(scores[j]))
        choices.append(int(chosen))

    return choices


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Pass --checkpoint runs/fusion/selected")

    checkpoint = Path(checkpoint)
    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    weight = float(config["mlp_weight"])

    if weight > 0:
        mlp = mlp_probabilities(bank, arrays, checkpoint / "mlp")
    else:
        mlp = None

    if weight < 1:
        ridge = ridge_probabilities(
            bank, arrays, checkpoint / "ridge"
        )
    else:
        ridge = None

    if weight == 1:
        support = mlp
    elif weight == 0:
        support = ridge
    else:
        support = weight * mlp + (1.0 - weight) * ridge

    return answer_choices(
        bank, support, config["group_power"]
    )