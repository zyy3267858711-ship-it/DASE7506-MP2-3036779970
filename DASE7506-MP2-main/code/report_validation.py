"""Report stratified validation accuracy for a saved prediction file."""
import argparse
import json
from pathlib import Path
from common import DATA, load_inputs, read_jsonl, validate_choices
from hard_submission import disagreement_mask
from train_hard import labels, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text(encoding='utf-8'))
    if data['split'] != 'validation':
        raise ValueError('This diagnostic accepts validation predictions only')
    bank, _ = load_inputs('validation')
    validate_choices(bank, data['ids'], data['selected'])
    y = labels(bank, read_jsonl(DATA / 'gsm8k_validation.jsonl'))
    result = evaluate(data['selected'], y, disagreement_mask(bank))
    text = json.dumps(result, indent=2) + '\n'
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
