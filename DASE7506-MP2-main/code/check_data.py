"""Verify packaged data and feature alignment without making predictions."""
import argparse
import json
from pathlib import Path

from common import DATA, check_data, load_inputs, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    args = parser.parse_args()
    manifest = check_data(args.data_dir)
    seen = set()
    result = {'protocol': manifest['protocol'], 'splits': {}}
    for split in manifest['splits']:
        bank, arrays = load_inputs(split, args.data_dir)
        ids = [row['id'] for row in bank]
        if seen.intersection(ids):
            raise ValueError('A question occurs in more than one split')
        seen.update(ids)
        references = read_jsonl(args.data_dir/f'gsm8k_{split}.jsonl')
        if len(references) != len(bank) or any(
            r['id'] != b['id'] or r['question'] != b['question']
            for r, b in zip(references, bank)
        ):
            raise ValueError('Reference alignment mismatch')
        result['splits'][split] = {'questions': len(bank),
                                   'embedding_shape': list(arrays['embeddings'].shape)}
    result['status'] = 'passed'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
