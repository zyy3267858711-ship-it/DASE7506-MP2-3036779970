"""Candidate selection with boosted trees and restricted arithmetic features."""
import ast
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from common import parse_answer

FEATURE_VERSION = 'tree-arithmetic-v1'
EQ = re.compile(r'([\d.,+*/()\- \t]{3,80})\s*=\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)')


def restricted_arithmetic(text):
    """Parse only arithmetic AST nodes, without eval or executing text."""
    node = ast.parse(text.replace(',', '').strip(), mode='eval').body
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if abs(n.value) > 1e12:
                raise ValueError('Number too large')
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            x = visit(n.operand)
            return -x if isinstance(n.op, ast.USub) else x
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if isinstance(n.op, ast.Div): return a/b
        raise ValueError('Unsupported expression')
    return visit(node)


def arithmetic_features(text):
    text = text.replace('×', '*').replace('÷', '/').replace('−', '-')
    correct = wrong = skipped = 0
    matches = list(EQ.finditer(text))[:64]
    for match in matches:
        left, right = match.groups()
        if not any(op in left for op in ['+', '-', '*', '/']):
            skipped += 1
            continue
        try:
            a = restricted_arithmetic(left)
            b = Fraction(right.replace(',', ''))
            tolerance = Fraction(1, 1000000) * max(Fraction(1), abs(b))
            if abs(a-b) <= tolerance:
                correct += 1
            else:
                wrong += 1
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            skipped += 1
    checked = correct+wrong
    return [np.log1p(correct), np.log1p(wrong), np.log1p(skipped),
            wrong/max(checked, 1), float(wrong > 0)]


def make_features(bank, arrays):
    e = np.asarray(arrays['embeddings'], dtype=np.float32)
    if e.shape != (len(bank), 4, 1024):
        raise ValueError('Expected embeddings [N,4,1024]')
    norm = np.linalg.norm(e, axis=-1, keepdims=True)
    u = e/np.maximum(norm, 1e-8)
    relative = u-u.mean(axis=1, keepdims=True)
    extra = []
    for question_index, row in enumerate(bank):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        similarity = u[question_index] @ u[question_index].T
        unique = len({a for a in answers if a is not None})
        for j, candidate in enumerate(row['candidates']):
            text, answer = candidate['text'], answers[j]
            others = [similarity[j,k] for k in range(4) if k != j]
            values = [float(answer is not None),
                      sum(a == answer for a in answers)/4 if answer is not None else 0,
                      unique/4, np.log1p(len(text)),
                      np.log1p(len(text.splitlines())), np.log1p(text.count('=')),
                      float(max(others)), float(min(others)),
                      float(norm[question_index,j,0])]
            extra.append(values+arithmetic_features(text))
    extra = np.asarray(extra, dtype=np.float32).reshape(len(bank),4,14)
    return np.concatenate([u,relative,extra], axis=-1).reshape(len(bank)*4,2062)


def choices(bank, scores):
    result=[]
    for row, group_scores in zip(bank, np.asarray(scores).reshape(len(bank),4)):
        valid=[j for j,c in enumerate(row['candidates']) if parse_answer(c['text']) is not None]
        j=max(valid,key=lambda k:float(group_scores[k])) if valid else int(np.argmax(group_scores))
        result.append(int(j))
    return result


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/tree_route/selected')
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    if config['feature_version'] != FEATURE_VERSION:
        raise ValueError('Feature version mismatch')
    model=CatBoostClassifier()
    model.load_model(str(checkpoint/'model.cbm'))
    support=model.predict_proba(make_features(bank,arrays),thread_count=4,task_type='CPU')[:,1]
    return choices(bank,support)
