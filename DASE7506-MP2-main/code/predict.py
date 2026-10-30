"""Prediction runner. Custom methods return choices; this runner never opens gold."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

from common import DATA, PROTOCOL, load_inputs, validate_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=Path, required=True,
                        help='Your Python file exposing predict(bank, arrays, checkpoint)')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=('train','validation','test','audit'), default='test')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tick = time.perf_counter()
    bank, arrays = load_inputs(args.split, args.data_dir)
    method_path = args.method.resolve()
    sys.path.insert(0, str(method_path.parent))
    spec = importlib.util.spec_from_file_location('student_method', method_path)
    method = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = method
    spec.loader.exec_module(method)
    choices = method.predict(bank, arrays, args.checkpoint)
    ids = [row['id'] for row in bank]
    validate_choices(bank, ids, choices)
    result = {'protocol': PROTOCOL, 'split': args.split, 'ids': ids, 'selected': choices,
              'prediction_seconds': time.perf_counter()-tick}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('ids','selected')}))


if __name__ == '__main__':
    main()
