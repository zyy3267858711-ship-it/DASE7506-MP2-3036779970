"""Train-independent NumPy inference; no references or PyTorch are loaded."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer


def parsed_answers(bank):
    return [[parse_answer(c['text']) for c in row['candidates']] for row in bank]


def disagreement_mask(bank):
    return np.array([len({a for a in group if a is not None}) >= 2
                     for group in parsed_answers(bank)], dtype=bool)


def features(bank, arrays, contextual):
    e = np.asarray(arrays['embeddings'], dtype=np.float32)
    if e.shape != (len(bank), 4, 1024):
        raise ValueError('Expected embeddings [N,4,1024]')
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-8)
    center = e.mean(axis=1, keepdims=True)
    similarity = e @ np.swapaxes(e, 1, 2)
    for j in range(4):
        similarity[:, j, j] = -2.0
    extras = []
    answers = parsed_answers(bank)
    for i, row in enumerate(bank):
        group = answers[i]
        distinct = len({a for a in group if a is not None})
        for j, candidate in enumerate(row['candidates']):
            text, answer = candidate['text'], group[j]
            extras.append([
                float(answer is not None),
                sum(a == answer for a in group) / 4 if answer is not None else 0,
                np.log1p(len(text)), np.log1p(len(text.splitlines())),
                np.log1p(text.count('=')), distinct / 4,
                float((e[i, j] * center[i, 0]).sum()),
                float(similarity[i, j].max()),
            ])
    extra = np.asarray(extras, dtype=np.float32).reshape(len(bank), 4, 8)
    parts = [e, e - center, np.abs(e - center), e * center, extra] if contextual else [e, extra]
    return np.concatenate(parts, axis=-1)


def probabilities(logits):
    value = np.exp(logits - logits.max(axis=1, keepdims=True))
    return value / value.sum(axis=1, keepdims=True)


def member_logits(raw, path):
    with np.load(path, allow_pickle=False) as m:
        z = np.clip((raw - m['mean']) / m['scale'], -10, 10)
        h = np.maximum(z @ m['w1'] + m['b1'], 0)
        h = np.maximum(h @ m['w2'] + m['b2'], 0)
        return (h @ m['w3'] + m['b3'])[..., 0]


def ensemble(raw, folder, members):
    total = np.zeros(raw.shape[:2], dtype=np.float32)
    for filename in members:
        total += probabilities(member_logits(raw, Path(folder) / filename))
    return total / len(members)


def choose(bank, scores):
    result = []
    for group, values in zip(parsed_answers(bank), scores):
        valid = [j for j, answer in enumerate(group) if answer is not None]
        # An unparseable candidate cannot receive a point from the fixed scorer.
        chosen = max(valid, key=lambda j: float(values[j])) if valid else int(np.argmax(values))
        result.append(int(chosen))
    return result


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/hard_rank/selected')
    folder = Path(checkpoint)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    raw = features(bank, arrays, config['contextual'])
    return choose(bank, ensemble(raw, folder, config['members']))
