"""Independent scoring of already-frozen choices."""
import argparse
import json
from pathlib import Path
import time

from common import DATA, PROTOCOL, check_data, score_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--split', choices=('train','validation','test','audit'), default='test')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tick = time.perf_counter()
    predictions = json.loads(args.predictions.read_text())
    if predictions.get('protocol') != PROTOCOL or predictions.get('split') != args.split:
        raise ValueError('Prediction protocol or split does not match')
    check_data(args.data_dir, (args.split,))
    result = {'protocol': PROTOCOL, 'split': args.split,
              **score_choices(args.split, predictions['ids'], predictions['selected'], args.data_dir),
              'scoring_seconds': time.perf_counter()-tick}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
