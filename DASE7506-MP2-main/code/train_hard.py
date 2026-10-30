"""Fixed experiment: disagreement-only fitting and stratified validation.

Never loads test inputs/references. PyTorch is imported only for training.
"""
import argparse
import json
import platform
import random
import shutil
import time
from pathlib import Path
import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl
from hard_submission import (features, disagreement_mask, choose,
                             member_logits, ensemble)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def labels(bank, refs):
    if len(bank) != len(refs):
        raise ValueError('Reference count mismatch')
    y = []
    for row, ref in zip(bank, refs):
        if row['id'] != ref['id'] or row['question'] != ref['question']:
            raise ValueError('Reference alignment mismatch')
        gold = parse_answer(ref['answer'])
        if gold is None:
            raise ValueError('Unparseable gold')
        y.append([parse_answer(c['text']) == gold for c in row['candidates']])
    y = np.asarray(y, dtype=bool)
    if not y.any(axis=1).all():
        raise ValueError('A group has no correct candidate')
    return y


def evaluate(choices, y, hard):
    ok = y[np.arange(len(y)), np.asarray(choices, dtype=np.int64)]
    def section(mask):
        n = int(mask.sum())
        c = int(ok[mask].sum())
        return {'correct': c, 'n': n, 'accuracy': 100*c/n if n else None}
    return {'all': section(np.ones(len(y), dtype=bool)),
            'disagreement': section(hard), 'other': section(~hard)}


def score_key(stats):
    return stats['disagreement']['correct'], stats['all']['correct']


def baseline_audit(bank, y, hard):
    result = {}
    paths = {'mlp': Path('runs/mlp/validation-predictions.json'),
             'fusion': Path('runs/fusion/validation-predictions.json')}
    for name, path in paths.items():
        if not path.exists():
            print(f'Skip missing baseline: {path}', flush=True)
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['split'] != 'validation' or data['ids'] != [r['id'] for r in bank]:
            raise ValueError(f'Baseline alignment mismatch: {path}')
        choices = data['selected']
        if len(choices) != len(bank) or any(type(i) is not int or not 0 <= i < 4 for i in choices):
            raise ValueError('Invalid baseline choices')
        result[name] = evaluate(choices, y, hard)
    return result


def export(path, model, mean, scale):
    payload = {'mean': mean, 'scale': scale}
    for name, layer in zip(['1', '2', '3'], [model[0], model[3], model[6]]):
        payload['w' + name] = layer.weight.detach().cpu().numpy().T.copy()
        payload['b' + name] = layer.bias.detach().cpu().numpy().copy()
    np.savez(path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/hard_rank'))
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    print('Loading TRAIN and VALIDATION only...', flush=True)
    train_bank, train_arrays = load_inputs('train')
    val_bank, val_arrays = load_inputs('validation')
    ty = labels(train_bank, read_jsonl(DATA / 'gsm8k_train.jsonl'))
    vy = labels(val_bank, read_jsonl(DATA / 'gsm8k_validation.jsonl'))
    th, vh = disagreement_mask(train_bank), disagreement_mask(val_bank)
    if not th.any() or not vh.any():
        raise ValueError('No disagreement groups; inspect the data')
    audit = {
        'definition': 'At least two distinct non-None parsed candidate answers',
        'train': {'total': len(train_bank), 'disagreement': int(th.sum())},
        'validation': {'total': len(val_bank), 'disagreement': int(vh.sum())},
        'baselines': baseline_audit(val_bank, vy, vh),
    }
    write_json(args.output / 'audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    selected = args.output / 'selected'
    if selected.exists():
        raise FileExistsError('Output already has selected/. Use a fresh --output directory.')

    import torch
    from torch import nn
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Training device: {device}', flush=True)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0), flush=True)

    configs = [
        {'name': 'local', 'contextual': False, 'hidden': 96, 'dropout': 0.4},
        {'name': 'context', 'contextual': True, 'hidden': 128, 'dropout': 0.5},
    ]
    seeds = [17, 29, 43]
    results = []
    best_bundle_key = (-1, -1)
    best_folder = None
    hard_bank = [row for row, keep in zip(train_bank, th) if keep]
    hard_arrays = {'embeddings': train_arrays['embeddings'][th]}

    for config in configs:
        print(f"\nConfiguration: {config['name']}", flush=True)
        folder = args.output / config['name']
        folder.mkdir(exist_ok=True)
        raw_train = features(hard_bank, hard_arrays, config['contextual'])
        raw_val = features(val_bank, val_arrays, config['contextual'])
        mean = raw_train.mean(axis=(0, 1)).astype(np.float32)
        scale = raw_train.std(axis=(0, 1)).astype(np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        tx = torch.from_numpy(np.clip((raw_train-mean)/scale, -10, 10)).to(device)
        vx = torch.from_numpy(np.clip((raw_val-mean)/scale, -10, 10)).to(device)
        targets_np = ty[th].astype(np.float32)
        target = torch.from_numpy(targets_np).to(device)
        soft_target = target / target.sum(dim=1, keepdim=True)
        members, histories = [], []

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            model = nn.Sequential(
                nn.Linear(tx.shape[-1], config['hidden']), nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(config['hidden'], 32), nn.ReLU(),
                nn.Dropout(config['dropout']), nn.Linear(32, 1),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)
            path = folder / f'seed_{seed}.npz'
            best_key, best_epoch = (-1, -1), 0
            history, best_logits = [], None

            for epoch in range(1, 161):
                model.train()
                order = torch.randperm(len(tx), device=device)
                loss_sum = 0.0
                for start in range(0, len(tx), 128):
                    idx = order[start:start+128]
                    logits = model(tx[idx]).squeeze(-1)
                    # Treat all correct candidates equally instead of allowing
                    # the probability-mass loss to concentrate on just one.
                    rank_loss = -(soft_target[idx] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                    binary_loss = nn.functional.binary_cross_entropy_with_logits(logits, target[idx])
                    loss = rank_loss + 0.2 * binary_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    loss_sum += float(loss.item()) * len(idx)

                model.eval()
                with torch.no_grad():
                    logits_np = model(vx).squeeze(-1).cpu().numpy()
                stats = evaluate(choose(val_bank, logits_np), vy, vh)
                key = score_key(stats)
                history.append({'epoch': epoch, 'training_loss': loss_sum/len(tx), 'validation': stats})
                if key > best_key:
                    best_key, best_epoch = key, epoch
                    best_logits = logits_np.copy()
                    export(path, model, mean, scale)
                if epoch == 1 or epoch % 10 == 0:
                    print(f"seed={seed} epoch={epoch} loss={loss_sum/len(tx):.4f} "
                          f"hard={stats['disagreement']['accuracy']:.3f}% "
                          f"all={stats['all']['accuracy']:.3f}% best_epoch={best_epoch}", flush=True)
                if epoch - best_epoch >= 25:
                    break

            exported = member_logits(raw_val, path)
            error = float(np.max(np.abs(exported-best_logits)))
            if not np.allclose(exported, best_logits, rtol=2e-3, atol=2e-3):
                raise RuntimeError(f'NumPy export mismatch: {error}')
            members.append(path.name)
            histories.append({'seed': seed, 'best_epoch': best_epoch,
                              'export_max_abs_error': error, 'history': history})
            print(f'Saved {path.name}; export error={error:.6g}', flush=True)

        bundle = dict(config, members=members)
        write_json(folder / 'config.json', bundle)
        stats = evaluate(choose(val_bank, ensemble(raw_val, folder, members)), vy, vh)
        print('Ensemble: ' + json.dumps(stats), flush=True)
        results.append({'config': bundle, 'validation': stats, 'training': histories})
        if score_key(stats) > best_bundle_key:
            best_bundle_key, best_folder = score_key(stats), folder
        del tx, vx, target, soft_target, model, optimizer

    shutil.copytree(best_folder, selected)
    asset_bytes = sum(p.stat().st_size for p in selected.rglob('*') if p.is_file())
    if asset_bytes > 16*1024*1024:
        raise RuntimeError('Inference assets exceed 16 MiB')
    summary = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'device': str(device), 'seeds': seeds,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'fit_subset': 'training groups with >=2 distinct parsed answers',
        'selection_rule': 'Disagreement-validation correct count, then full-validation correct count; earlier config wins remaining ties',
        'selected': best_folder.name, 'asset_bytes': asset_bytes,
        'training_and_search_seconds': time.perf_counter()-started,
        'audit': audit, 'experiments': results,
    }
    write_json(args.output / 'experiments.json', summary)
    print(f'Finished. Selected={best_folder.name}; assets={asset_bytes/1024**2:.3f} MiB', flush=True)
    print(f'Checkpoint: {selected}', flush=True)


if __name__ == '__main__':
    main()
