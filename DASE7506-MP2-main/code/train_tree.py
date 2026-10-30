"""Fixed small GPU/CPU boosted-tree search; never opens test data."""
import argparse
import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
from catboost import CatBoostClassifier
from common import DATA, load_inputs, parse_answer, read_jsonl
from tree_submission import FEATURE_VERSION, make_features, choices


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def labels(bank, refs):
    if len(bank)!=len(refs): raise ValueError('Reference count mismatch')
    result=[]
    for row, ref in zip(bank,refs):
        if row['id']!=ref['id'] or row['question']!=ref['question']:
            raise ValueError('Reference alignment mismatch')
        gold=parse_answer(ref['answer'])
        if gold is None: raise ValueError('Unparseable reference')
        result.append([parse_answer(c['text'])==gold for c in row['candidates']])
    y=np.asarray(result,dtype=bool)
    if not y.any(axis=1).all(): raise ValueError('No correct candidate in a group')
    return y


def disagreement(bank):
    return np.asarray([len({parse_answer(c['text']) for c in row['candidates']}
                          -{None})>=2 for row in bank],dtype=bool)


def evaluate(selected,y,hard):
    ok=y[np.arange(len(y)),np.asarray(selected,dtype=np.int64)]
    def part(mask):
        n=int(mask.sum()); correct=int(ok[mask].sum())
        return {'correct':correct,'n':n,'accuracy':100*correct/n if n else None}
    return {'all':part(np.ones(len(y),dtype=bool)), 'disagreement':part(hard)}


def key(stats):
    return stats['disagreement']['correct'],stats['all']['correct']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-type',choices=['GPU','CPU'],default='GPU')
    parser.add_argument('--output',type=Path,default=Path('runs/tree_route'))
    args=parser.parse_args()
    output=args.output
    if (output/'selected').exists():
        raise FileExistsError('selected/ exists; preserve this experiment and use a fresh --output')
    started=time.perf_counter(); output.mkdir(parents=True,exist_ok=True)
    print('Loading TRAIN and VALIDATION only...',flush=True)
    tb,ta=load_inputs('train'); vb,va=load_inputs('validation')
    ty=labels(tb,read_jsonl(DATA/'gsm8k_train.jsonl'))
    vy=labels(vb,read_jsonl(DATA/'gsm8k_validation.jsonl'))
    th,vh=disagreement(tb),disagreement(vb)
    if not vh.any(): raise ValueError('No disagreement validation groups')
    print('Building vector and arithmetic features...',flush=True)
    tx,vx=make_features(tb,ta),make_features(vb,va)
    print(f'Train features: {tx.shape}; validation: {vx.shape}',flush=True)
    # Easy/unanimous groups still contribute, with lower weight.
    weights=np.repeat(np.where(th,1.,.2),4)
    results=[]; best_key=(-1,-1); best_model=None; best_record=None
    for depth in [4,6]:
        params=dict(iterations=1000,depth=depth,learning_rate=.04,
                    loss_function='Logloss',l2_leaf_reg=10,random_seed=17,
                    border_count=32,thread_count=4,allow_writing_files=False,
                    task_type=args.task_type)
        if args.task_type=='GPU': params['devices']='0'
        print(f'\nTraining depth={depth}; device={args.task_type}',flush=True)
        model=CatBoostClassifier(**params)
        model.fit(tx,ty.reshape(-1).astype(np.int32),sample_weight=weights,verbose=100)
        for trees in [200,400,600,800,1000]:
            support=model.predict_proba(vx,ntree_end=trees,thread_count=4,task_type='CPU')[:,1]
            stats=evaluate(choices(vb,support),vy,vh)
            record={'depth':depth,'trees':trees,'validation':stats}
            results.append(record)
            print(json.dumps(record),flush=True)
            if key(stats)>best_key:
                best_key=key(stats); best_record=record
                best_model=model.copy(); best_model.shrink(trees)
    selected=output/'selected'; selected.mkdir()
    best_model.save_model(str(selected/'model.cbm'))
    write_json(selected/'config.json',{'feature_version':FEATURE_VERSION,'selection':best_record})
    # Verify saved-model predictions match the selected validation score.
    reloaded=CatBoostClassifier(); reloaded.load_model(str(selected/'model.cbm'))
    actual=evaluate(choices(vb,reloaded.predict_proba(vx,thread_count=4,task_type='CPU')[:,1]),vy,vh)
    if actual != best_record['validation']:
        raise RuntimeError('Reloaded-model validation mismatch')
    size=sum(p.stat().st_size for p in selected.iterdir() if p.is_file())
    if size>16*1024*1024: raise RuntimeError('Assets exceed 16 MiB')
    write_json(output/'experiments.json',{'python':platform.python_version(),'numpy':np.__version__,
        'catboost':catboost.__version__,'device':args.task_type,'seed':17,
        'selection_rule':'disagreement validation correct count, then all validation correct count; earlier wins ties',
        'training_weight':'disagreement=1; other=0.2','selected':best_record,
        'asset_bytes':size,'training_search_seconds':time.perf_counter()-started,'experiments':results})
    print('\nFinished. '+json.dumps(best_record),flush=True)
    print(f'Assets: {size/1024**2:.3f} MiB; checkpoint: {selected}',flush=True)


if __name__=='__main__': main()
