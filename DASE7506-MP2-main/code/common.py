"""Fixed bank IO, numeric parser and grading; no model or training algorithm."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import numpy as np

DATA = Path(__file__).resolve().parent/'data'
PROTOCOL = 'dase7506-mp2-open-hard-verifier-v5'
SPLITS = ('train', 'validation', 'test', 'audit')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_answer(text):
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def check_data(data_dir=DATA, splits=None, inputs_only=False):
    data = Path(data_dir)
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Wrong bank protocol')
    chosen = list(manifest['splits']) if splits is None else ([splits] if isinstance(splits, str) else list(splits))
    for split in chosen:
        if split not in SPLITS or split not in manifest['splits']:
            raise ValueError('Unavailable split')
        files = [f'candidates/gsm8k_{split}.jsonl', f'candidates/gsm8k_{split}.npz']
        if not inputs_only:
            files.append(f'gsm8k_{split}.jsonl')
        for name in files:
            digest = hashlib.sha256((data/name).read_bytes()).hexdigest()
            if digest != manifest['sha256'].get(name):
                raise ValueError(f'Data hash mismatch: {name}')
    return manifest


def load_bank(split, data_dir=DATA):
    data = Path(data_dir)
    manifest = json.loads((data/'manifest.json').read_text())
    if split not in SPLITS or manifest['protocol'] != PROTOCOL:
        raise ValueError('Unknown split or wrong protocol')
    bank = read_jsonl(data/'candidates'/f'gsm8k_{split}.jsonl')
    if len(bank) != manifest['splits'][split] or len({row['id'] for row in bank}) != len(bank):
        raise ValueError('Incomplete bank or repeated IDs')
    if any([c['index'] for c in row['candidates']] != [0,1,2,3] for row in bank):
        raise ValueError('Expected four ordered candidates')
    return bank


def load_inputs(split, data_dir=DATA):
    check_data(data_dir, (split,), inputs_only=True)
    bank = load_bank(split, data_dir)
    with np.load(Path(data_dir)/'candidates'/f'gsm8k_{split}.npz', allow_pickle=False) as f:
        arrays = {'embeddings': f['embeddings'].copy()}
    n = len(bank)
    if arrays['embeddings'].shape != (4*n,1024):
        raise ValueError('Frozen feature shapes differ from the contract')
    arrays = {key: value.reshape(n,4,*value.shape[1:]) for key,value in arrays.items()}
    if any(not np.isfinite(arrays[k]).all() for k in ('embeddings',)):
        raise ValueError('Nonfinite features')
    return bank, arrays


def validate_choices(bank, ids, choices):
    if ids != [row['id'] for row in bank] or len(choices) != len(bank):
        raise ValueError('Exactly one choice per question in bank order is required')
    if any(type(i) is not int or not 0 <= i < 4 for i in choices):
        raise ValueError('Choices must be integer candidate indices 0, 1, 2, or 3')


def score_choices(split, ids, choices, data_dir=DATA):
    bank = load_bank(split, data_dir)
    validate_choices(bank, ids, choices)
    gold = read_jsonl(Path(data_dir)/f'gsm8k_{split}.jsonl')
    if len(gold) != len(bank):
        raise ValueError('Incomplete scoring references')
    correct = 0
    for q,row,i in zip(gold, bank, choices):
        if q['id'] != row['id'] or q['question'] != row['question']:
            raise ValueError('Reference/bank alignment mismatch')
        answer = parse_answer(row['candidates'][i]['text'])
        correct += int(answer is not None and answer == parse_answer(q['answer']))
    return {'correct': correct, 'n': len(bank), 'accuracy': 100*correct/len(bank)}
