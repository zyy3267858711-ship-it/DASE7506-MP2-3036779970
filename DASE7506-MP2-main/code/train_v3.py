"""Train the fixed V3 recipe using only this file and submission_v3.py.

Fit: supplied training split only. Validation: reporting and parity checks only.
Test is never opened here. AI assistance: OpenAI ChatGPT/Codex.

Use the official assignment's code/ directory with common.py and data/ present.
Install: python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 threadpoolctl==3.6.0 lightgbm==4.6.0
Train: python train_v3.py --output runs/v3_shared_model
No V2 files, existing checkpoints, or experiment outputs are required.
"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

_HERE = Path(__file__).resolve().parent
if not (_HERE / "common.py").is_file() and (_HERE.parent / "common.py").is_file():
    sys.path.append(str(_HERE.parent))

from common import DATA, load_inputs, parse_answer, read_jsonl
from submission_v3 import get_answers, make_features, predict, text_features, tree_logits


SEEDS = (2026, 2027, 2028)
TREE_PARAMS = dict(
    objective="lambdarank", learning_rate=0.03, num_leaves=7,
    min_data_in_leaf=80, lambda_l2=10., max_bin=63, feature_fraction=0.8,
    verbosity=-1, num_threads=4, seed=2026, deterministic=True, force_col_wise=True,
)


def labels(bank, split):
    """Read training/validation supervision with strict alignment checks."""
    if split not in ("train", "validation"):
        raise ValueError("Test references cannot enter training.")
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference count mismatch.")
    out = []
    for row, reference in zip(bank, references):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference alignment mismatch.")
        gold = parse_answer(reference["answer"])
        if gold is None:
            raise ValueError("Invalid reference answer.")
        out.append([int(parse_answer(candidate["text"]) == gold)
                    for candidate in row["candidates"]])
    return np.asarray(out, dtype=np.int32)


def hard_groups(answers):
    """Questions containing at least two distinct valid numeric answers."""
    return np.asarray([len(set(group) - {None}) >= 2 for group in answers])


def training_stats(x):
    """Fit normalization statistics on the supplied training features only."""
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(axis=0), np.maximum(flat.std(axis=0), 1e-4)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_trees(booster):
    feature, threshold, left, right, value, roots = [], [], [], [], [], []

    def visit(node):
        index = len(feature)
        feature.append(-1)
        threshold.append(0.)
        left.append(-1)
        right.append(-1)
        value.append(0.)
        if "leaf_value" in node:
            value[index] = node["leaf_value"]
        else:
            if node["decision_type"] != "<=":
                raise ValueError("Only numeric tree splits are supported")
            feature[index] = node["split_feature"]
            threshold[index] = node["threshold"]
            left[index] = visit(node["left_child"])
            right[index] = visit(node["right_child"])
        return index

    for tree in booster.dump_model()["tree_info"]:
        roots.append(visit(tree["tree_structure"]))
    return dict(
        roots=np.asarray(roots, dtype=np.int32), feature=np.asarray(feature, dtype=np.int32),
        threshold=np.asarray(threshold, dtype=np.float64), left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32), value=np.asarray(value, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/v3_model"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a fresh directory to preserve prior runs")
    threadpool_limits(4)
    started = time.perf_counter()
    tb, ta = load_inputs("train")
    vb, va = load_inputs("validation")
    y, vy = labels(tb, "train"), labels(vb, "validation")
    th, vh = hard_groups(get_answers(tb)), hard_groups(get_answers(vb))
    tx = make_features(ta["embeddings"], text_features(tb), "text")
    vx = make_features(va["embeddings"], text_features(vb), "text")
    mean, std = training_stats(tx)
    flat = ((tx - mean) / std).reshape(-1, tx.shape[-1])
    vflat = ((vx - mean) / std).reshape(-1, vx.shape[-1])
    sample_weight = np.repeat(np.where(th, 1., 0.25), 4)
    args.output.mkdir(parents=True)
    members = []
    max_error = 0.
    for seed in SEEDS:
        network = MLPClassifier(
            hidden_layer_sizes=(64,), alpha=10., batch_size=128,
            learning_rate_init=0.0005, random_state=seed, max_iter=1,
        )
        for _ in range(6):
            network.partial_fit(flat, y.ravel(), classes=[0, 1], sample_weight=sample_weight)
        weights = dict(mean=mean, std=std, w0=network.coefs_[0], b0=network.intercepts_[0], w1=network.coefs_[1], b1=network.intercepts_[1])
        reference = network.predict_proba(vflat)[:, 1]
        x = ((vx - weights["mean"]) / weights["std"]).reshape(-1, vx.shape[-1])
        numpy_logits = (np.maximum(x @ weights["w0"] + weights["b0"], 0) @ weights["w1"] + weights["b1"]).reshape(-1, 4)
        numpy_probability = 1. / (1. + np.exp(-np.clip(numpy_logits.ravel(), -40, 40)))
        error = float(np.max(np.abs(reference - numpy_probability)))
        max_error = max(error, max_error)
        file = f"mlp_{seed}.npz"
        np.savez(args.output / file, **weights)
        members.append(dict(kind="mlp", mode="text", file=file, weight=0.25, seed=seed))
        print(f"MLP seed={seed}, NumPy parity error={error:.3g}", flush=True)

    data = lgb.Dataset(tx[th].reshape(-1, tx.shape[-1]), label=y[th].ravel(), group=[4] * int(th.sum()))
    booster = lgb.train(TREE_PARAMS, data, num_boost_round=400)
    weights = export_trees(booster)
    reference = booster.predict(vx.reshape(-1, vx.shape[-1]), raw_score=True, num_threads=4).reshape(-1, 4)
    tree_error = float(np.max(np.abs(reference - tree_logits(vx, weights))))
    max_error = max(max_error, tree_error)
    if max_error > 1e-5:
        raise RuntimeError(f"Export parity failed: {max_error}")
    np.savez(args.output / "rank_trees.npz", **weights)
    members.append(dict(kind="trees", mode="text", file="rank_trees.npz", weight=0.25))
    config = dict(version=3, members=members, pool=True, vote_weight=0., feature_version=1)
    write_json(args.output / "config.json", config)

    choices = np.asarray(predict(vb, va, args.output))
    correct = vy[np.arange(len(vy)), choices]
    files = sorted(args.output.iterdir())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    array_bytes = 0
    for path in args.output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as asset:
            array_bytes += sum(asset[k].nbytes for k in asset.files)
    asset_bytes = sum(p.stat().st_size for p in files)
    if asset_bytes > 16 * 2**20:
        raise RuntimeError("Checkpoint exceeds 16 MiB")
    info = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        recipe="Three MLPs (75%) plus shallow LambdaRank trees (25%); mean-pool equal answers",
        seeds=list(SEEDS), train_groups=len(tb), hard_train_groups=int(th.sum()),
        neural=dict(hidden=64, alpha=10., epochs=6, batch_size=128, learning_rate=0.0005, easy_group_weight=0.25),
        trees=dict(params=TREE_PARAMS, rounds=400),
        validation=dict(correct=int(correct.sum()), n=len(vy), accuracy=100 * float(correct.mean()), hard_correct=int(correct[vh].sum()), hard_n=int(vh.sum()), hard_accuracy=100 * float(correct[vh].mean())),
        asset_sha256=hashes, asset_bytes=asset_bytes, uncompressed_array_bytes=array_bytes,
        numpy_parity_max_error=max_error, tree_parity_max_error=tree_error,
        elapsed_seconds=time.perf_counter() - started,
        export_from_development_assets=False,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lgb.__version__,
        test_access="No test inputs or references loaded. Recipe selected using validation only; ties prefer first registered recipe.",
    )
    write_json(args.output.parent / f"{args.output.name}_training_info.json", info)
    print(json.dumps(info["validation"], indent=2), flush=True)
    print(f"Frozen: {args.output}, {asset_bytes / 2**20:.3f} MiB", flush=True)


if __name__ == "__main__":
    main()
