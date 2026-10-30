"""Select a small fusion configuration using validation only."""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from fusion_submission import (
    ridge_probabilities,
    mlp_probabilities,
    answer_choices,
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def labels_for(bank, references):
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
            raise ValueError("Unparseable reference")

        labels.append([
            parse_answer(candidate["text"]) == gold
            for candidate in row["candidates"]
        ])

    labels = np.asarray(labels, dtype=bool)

    if not labels.any(axis=1).all():
        raise ValueError("A question has no correct candidate")

    return labels


def correct_mask(choices, labels):
    choices = np.asarray(choices, dtype=np.int64)
    return labels[np.arange(len(labels)), choices]


def main():
    started = time.perf_counter()

    ridge_dir = Path("runs/selected")
    mlp_dir = Path("runs/mlp/selected")
    output = Path("runs/fusion")
    selected = output / "selected"

    if selected.exists():
        raise FileExistsError(
            "runs/fusion/selected already exists. "
            "Back up the existing fusion experiment first."
        )

    if not (ridge_dir / "model.npz").is_file():
        raise FileNotFoundError(ridge_dir / "model.npz")

    if not (mlp_dir / "config.json").is_file():
        raise FileNotFoundError(mlp_dir / "config.json")

    output.mkdir(parents=True, exist_ok=True)

    print("Loading validation only...", flush=True)
    bank, arrays = load_inputs("validation")
    references = read_jsonl(DATA / "gsm8k_validation.jsonl")
    labels = labels_for(bank, references)

    print("Computing MLP support...", flush=True)
    mlp = mlp_probabilities(bank, arrays, mlp_dir)

    print("Computing ridge support...", flush=True)
    ridge = ridge_probabilities(bank, arrays, ridge_dir)

    mlp_choices = answer_choices(bank, mlp)
    ridge_choices = answer_choices(bank, ridge)

    mlp_correct = correct_mask(mlp_choices, labels)
    ridge_correct = correct_mask(ridge_choices, labels)

    diagnostics = {
        "n": len(bank),
        "mlp_correct": int(mlp_correct.sum()),
        "ridge_support_argmax_correct": int(ridge_correct.sum()),
        "both_correct": int((mlp_correct & ridge_correct).sum()),
        "only_mlp_correct": int(
            (mlp_correct & ~ridge_correct).sum()
        ),
        "only_ridge_correct": int(
            (~mlp_correct & ridge_correct).sum()
        ),
        "both_wrong": int(
            (~mlp_correct & ~ridge_correct).sum()
        ),
        "at_least_one_base_choice_correct": int(
            (mlp_correct | ridge_correct).sum()
        ),
    }

    print("\nValidation complementarity:", flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)

    # The unchanged MLP candidate rule is evaluated first.
    # Ties retain the earlier configuration.
    weights = [1.0, 0.75, 0.5, 0.25, 0.0]
    group_powers = [None, 0.0, 0.5, 1.0]

    records = []
    best_correct = -1
    best_config = None

    for weight in weights:
        support = weight * mlp + (1.0 - weight) * ridge

        for power in group_powers:
            choices = answer_choices(bank, support, power)
            count = int(correct_mask(choices, labels).sum())
            accuracy = 100.0 * count / len(bank)

            record = {
                "mlp_weight": weight,
                "group_power": power,
                "correct": count,
                "n": len(bank),
                "validation_accuracy": accuracy,
            }
            records.append(record)

            print(
                f"MLP weight={weight:.2f}, "
                f"group_power={power}: "
                f"{count}/{len(bank)} = {accuracy:.4f}%",
                flush=True,
            )

            if count > best_correct:
                best_correct = count
                best_config = {
                    "mlp_weight": weight,
                    "group_power": power,
                }

    # Copy only assets actually needed by the selected configuration.
    selected.mkdir()

    weight = best_config["mlp_weight"]

    if weight > 0:
        target = selected / "mlp"
        target.mkdir()

        config_path = mlp_dir / "config.json"
        mlp_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        shutil.copy2(config_path, target / "config.json")

        for filename in mlp_config["members"]:
            shutil.copy2(
                mlp_dir / filename,
                target / filename,
            )

    if weight < 1:
        target = selected / "ridge"
        target.mkdir()
        shutil.copy2(
            ridge_dir / "model.npz",
            target / "model.npz",
        )

    write_json(selected / "config.json", best_config)

    asset_bytes = sum(
        path.stat().st_size
        for path in selected.rglob("*")
        if path.is_file()
    )

    if asset_bytes > 16 * 1024 * 1024:
        raise RuntimeError(
            f"Assets exceed 16 MiB: {asset_bytes} bytes"
        )

    summary = {
        "selection_split": "validation",
        "selection_rule": (
            "Highest validation correct count; "
            "ties retain earlier configuration."
        ),
        "diagnostics": diagnostics,
        "selected": best_config,
        "selected_validation_correct": best_correct,
        "selected_validation_accuracy": (
            100.0 * best_correct / len(bank)
        ),
        "asset_bytes": asset_bytes,
        "selection_seconds": time.perf_counter() - started,
        "experiments": records,
    }

    write_json(output / "experiments.json", summary)

    print("\nSelected configuration:", flush=True)
    print(json.dumps(best_config, indent=2), flush=True)
    print(
        f"Validation: {best_correct}/{len(bank)} "
        f"= {100.0 * best_correct / len(bank):.4f}%",
        flush=True,
    )
    print(
        f"Assets: {asset_bytes / 1024**2:.3f} MiB",
        flush=True,
    )
    print("Checkpoint: runs/fusion/selected", flush=True)


if __name__ == "__main__":
    main()