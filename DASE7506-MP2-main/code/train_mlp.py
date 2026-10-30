"""Train small groupwise candidate rankers.

Only train references fit model parameters.
Validation selects epochs and one of two fixed configurations.
"""

import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import DATA, load_inputs, parse_answer, read_jsonl
from mlp_submission import (
    make_features,
    member_logits,
    ensemble_probabilities,
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


def accuracy(scores, labels):
    choices = np.argmax(scores, axis=1)
    return float(
        100.0 * labels[np.arange(len(labels)), choices].mean()
    )


def export_model(path, model, mean, scale):
    # PyTorch Linear stores [out, in]; NumPy expects [in, out].
    np.savez(
        path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        w1=model[0].weight.detach().cpu().numpy().T.copy(),
        b1=model[0].bias.detach().cpu().numpy().copy(),
        w2=model[3].weight.detach().cpu().numpy().T.copy(),
        b2=model[3].bias.detach().cpu().numpy().copy(),
    )


def train_member(
    train_x, train_y, val_x, val_y,
    mean, scale, config, seed, output_path, device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = nn.Sequential(
        nn.Linear(train_x.shape[-1], config["hidden"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden"], 1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(val_x).to(device)

    best_accuracy = -1.0
    best_epoch = 0
    best_logits = None
    history = []
    batch_size = 128

    for epoch in range(1, 121):
        model.train()
        order = torch.randperm(len(x), device=device)
        loss_sum = 0.0

        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            logits = model(x[indices]).squeeze(-1)

            # Probability mass assigned to ANY correct candidate.
            positive_logits = logits.masked_fill(
                ~y[indices], float("-inf")
            )
            loss = (
                torch.logsumexp(logits, dim=1)
                - torch.logsumexp(positive_logits, dim=1)
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(loss.item()) * len(indices)

        model.eval()
        with torch.no_grad():
            val_logits = model(vx).squeeze(-1).cpu().numpy()

        score = accuracy(val_logits, val_y)
        train_loss = loss_sum / len(x)

        history.append({
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_accuracy": score,
        })

        if score > best_accuracy:
            best_accuracy = score
            best_epoch = epoch
            best_logits = val_logits.copy()
            export_model(output_path, model, mean, scale)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"seed={seed}, epoch={epoch}, "
                f"loss={train_loss:.4f}, val={score:.4f}%, "
                f"best={best_accuracy:.4f}% at {best_epoch}",
                flush=True,
            )

        # Stop after 20 epochs without improved validation accuracy.
        if epoch - best_epoch >= 20:
            break

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
    }, best_logits


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}", flush=True)

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    root = Path("runs/mlp")
    root.mkdir(parents=True, exist_ok=True)

    # Prevent mixing a fresh run with old exported models.
    if (root / "selected").exists():
        raise FileExistsError(
            "runs/mlp/selected already exists. "
            "Back up the previous experiment before a new run."
        )

    print("Loading train and validation only...", flush=True)
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")

    train_y = labels_for(
        train_bank, read_jsonl(DATA / "gsm8k_train.jsonl")
    )
    val_y = labels_for(
        val_bank, read_jsonl(DATA / "gsm8k_validation.jsonl")
    )

    print("Constructing features...", flush=True)
    raw_train = make_features(train_bank, train_arrays)
    raw_val = make_features(val_bank, val_arrays)

    # Train-only learned preprocessing.
    mean = raw_train.mean(axis=(0, 1))
    scale = raw_train.std(axis=(0, 1))
    scale = np.where(scale < 1e-6, 1.0, scale)

    train_x = np.ascontiguousarray(
        (raw_train - mean) / scale, dtype=np.float32
    )
    val_x = np.ascontiguousarray(
        (raw_val - mean) / scale, dtype=np.float32
    )

    seeds = [17, 29, 43]
    configs = [
        {
            "name": "small",
            "hidden": 64,
            "dropout": 0.35,
            "lr": 0.0003,
            "weight_decay": 0.01,
        },
        {
            "name": "medium",
            "hidden": 128,
            "dropout": 0.50,
            "lr": 0.0003,
            "weight_decay": 0.03,
        },
    ]

    results = []
    best_score = -1.0
    best_folder = None

    for config in configs:
        folder = root / config["name"]
        folder.mkdir(exist_ok=True)
        members = []
        member_records = []

        print(f"\nConfiguration: {config['name']}", flush=True)

        for seed in seeds:
            filename = f"seed_{seed}.npz"
            path = folder / filename

            record, best_torch_logits = train_member(
                train_x, train_y, val_x, val_y,
                mean, scale, config, seed, path, device
            )

            # Check exported NumPy inference against the saved epoch.
            numpy_logits = member_logits(raw_val, path)
            max_error = float(
                np.max(np.abs(numpy_logits - best_torch_logits))
            )

            if not np.allclose(
                numpy_logits, best_torch_logits,
                rtol=1e-3, atol=1e-3
            ):
                raise RuntimeError(
                    f"Export mismatch: max error={max_error}"
                )

            record["numpy_export_max_abs_error"] = max_error
            members.append(filename)
            member_records.append(record)

            print(
                f"Saved {filename}; export error={max_error:.6g}",
                flush=True,
            )

        bundle_config = {
            "members": members,
            "architecture": config,
            "feature_dimension": int(raw_train.shape[-1]),
            "aggregation": "mean of member softmax probabilities",
        }
        write_json(folder / "config.json", bundle_config)

        probabilities = ensemble_probabilities(
            raw_val, folder, members
        )
        score = accuracy(probabilities, val_y)

        print(
            f"{config['name']} ensemble validation: {score:.4f}%",
            flush=True,
        )

        results.append({
            "config": config,
            "ensemble_validation_accuracy": score,
            "members": member_records,
        })

        if score > best_score:
            best_score = score
            best_folder = folder

    selected = root / "selected"
    shutil.copytree(best_folder, selected)

    asset_bytes = sum(
        path.stat().st_size
        for path in selected.rglob("*")
        if path.is_file()
    )

    if asset_bytes > 16 * 1024 * 1024:
        raise RuntimeError("Inference assets exceed 16 MiB")

    write_json(root / "experiments.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda" else None
        ),
        "seeds": seeds,
        "selection_rule": (
            "Highest validation ensemble accuracy; "
            "ties retain the first configuration."
        ),
        "selected_configuration": best_folder.name,
        "selected_validation_accuracy": best_score,
        "asset_bytes": asset_bytes,
        "training_and_search_seconds": (
            time.perf_counter() - started
        ),
        "experiments": results,
    })

    print("\nFinished.", flush=True)
    print(f"Selected: {best_folder.name}", flush=True)
    print(f"Validation: {best_score:.4f}%", flush=True)
    print(f"Assets: {asset_bytes / 1024**2:.3f} MiB", flush=True)
    print("Checkpoint: runs/mlp/selected", flush=True)


if __name__ == "__main__":
    main()