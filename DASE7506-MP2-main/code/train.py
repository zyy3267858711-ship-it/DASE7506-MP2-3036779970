"""Train small ridge scorers using train references only.

Validation is used for hyperparameter and method selection.
Test inputs/references are never opened here.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from submission import make_features, majority_choices


def make_labels(bank, references):
    if len(bank) != len(references):
        raise ValueError("Reference count mismatch")

    labels = []

    for row, reference in zip(bank, references):
        if (
            row["id"] != reference["id"]
            or row["question"] != reference["question"]
        ):
            raise ValueError("Reference alignment mismatch")

        gold = parse_answer(reference["answer"])
        if gold is None:
            raise ValueError("Cannot parse a reference answer")

        labels.append([
            float(parse_answer(candidate["text"]) == gold)
            for candidate in row["candidates"]
        ])

    labels = np.asarray(labels, dtype=np.float64)

    if not np.all(labels.sum(axis=1) >= 1):
        raise ValueError("A group has no correct candidate")

    return labels


def accuracy(choices, labels):
    choices = np.asarray(choices, dtype=np.int64)
    return float(
        100.0 * labels[np.arange(len(labels)), choices].mean()
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    started = time.perf_counter()
    output = Path("runs")
    output.mkdir(exist_ok=True)

    print("Loading train and validation inputs...", flush=True)
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")

    train_references = read_jsonl(DATA / "gsm8k_train.jsonl")
    val_references = read_jsonl(DATA / "gsm8k_validation.jsonl")

    train_labels = make_labels(train_bank, train_references)
    val_labels = make_labels(val_bank, val_references)

    y = train_labels.reshape(-1)
    target_mean = float(y.mean())
    y_centered = y - target_mean

    majority_accuracy = accuracy(
        majority_choices(val_bank), val_labels
    )

    print(
        f"majority validation accuracy: {majority_accuracy:.4f}%",
        flush=True,
    )

    majority_dir = output / "majority"
    majority_dir.mkdir(exist_ok=True)
    np.savez(
        majority_dir / "model.npz",
        mode=np.array("majority"),
    )

    records = [{
        "mode": "majority",
        "alpha": None,
        "validation_accuracy": majority_accuracy,
    }]

    best_overall_accuracy = majority_accuracy
    best_overall_mode = "majority"

    # Fixed before looking at test scores.
    alphas = [0.1, 1.0, 10.0]

    for mode in ["embedding", "combined"]:
        print(f"\nBuilding features: {mode}", flush=True)

        x_train = make_features(train_bank, train_arrays, mode)
        x_val = make_features(val_bank, val_arrays, mode)

        # Fit all learned preprocessing on training inputs only.
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)

        z_train = (x_train - mean) / scale
        z_val = (x_val - mean) / scale

        sample_count = len(z_train)
        gram = (z_train.T @ z_train) / sample_count
        rhs = (z_train.T @ y_centered) / sample_count

        best_accuracy = -1.0
        best_alpha = None
        best_weights = None

        for alpha in alphas:
            matrix = gram.copy()
            matrix.flat[::matrix.shape[0] + 1] += alpha
            weights = np.linalg.solve(matrix, rhs)

            scores = (
                z_val @ weights + target_mean
            ).reshape(len(val_bank), 4)

            choices = np.argmax(scores, axis=1)
            score = accuracy(choices, val_labels)

            print(
                f"{mode}, alpha={alpha}: "
                f"validation accuracy={score:.4f}%",
                flush=True,
            )

            records.append({
                "mode": mode,
                "alpha": alpha,
                "validation_accuracy": score,
            })

            if score > best_accuracy:
                best_accuracy = score
                best_alpha = alpha
                best_weights = weights.copy()

        model_dir = output / mode
        model_dir.mkdir(exist_ok=True)

        np.savez(
            model_dir / "model.npz",
            mode=np.array(mode),
            mean=mean,
            scale=scale,
            weights=best_weights,
            target_mean=np.array(target_mean),
            alpha=np.array(best_alpha),
        )

        print(
            f"Saved {mode}: alpha={best_alpha}, "
            f"validation={best_accuracy:.4f}%",
            flush=True,
        )

        if best_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_accuracy
            best_overall_mode = mode

    summary = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": None,
        "seed_note": "No random sampling or random initialization used.",
        "selection_rule": (
            "Highest validation accuracy; ties keep the earlier method."
        ),
        "selected_mode": best_overall_mode,
        "selected_validation_accuracy": best_overall_accuracy,
        "training_and_search_seconds": time.perf_counter() - started,
        "experiments": records,
    }

    write_json(output / "experiments.json", summary)

    selected_dir = output / "selected"
    selected_dir.mkdir(exist_ok=True)
    selected_source = output / best_overall_mode / "model.npz"
    (selected_dir / "model.npz").write_bytes(
        selected_source.read_bytes()
    )

    write_json(
        output / "selection.json",
        {
            "selected_mode": best_overall_mode,
            "checkpoint": "runs/selected",
            "selected_using": "validation accuracy only",
        },
    )

    print("\nTraining and validation search finished.", flush=True)
    print(f"Selected method: {best_overall_mode}", flush=True)
    print("Selected checkpoint: runs/selected", flush=True)
    print("Do not use test scores to change this selection.", flush=True)


if __name__ == "__main__":
    main()