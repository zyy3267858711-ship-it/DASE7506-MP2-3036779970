"""Train a compact hashed text/candidate interaction model.

Fit uses supplied train references only. Epoch/model choice uses validation.
Test data are never loaded here.
"""
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import DATA, load_inputs, parse_answer, read_jsonl
from text_route_submission import (
    VOCAB_SIZE, EMBED_DIM, text_sequences, auxiliary,
)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def make_labels(bank, refs):
    if len(bank) != len(refs):
        raise ValueError("Reference count mismatch")
    result=[]
    for row, ref in zip(bank, refs):
        if row['id'] != ref['id'] or row['question'] != ref['question']:
            raise ValueError("Reference alignment mismatch")
        gold=parse_answer(ref['answer'])
        if gold is None:
            raise ValueError("Unparseable gold answer")
        result.append([parse_answer(c['text']) == gold for c in row['candidates']])
    y=np.asarray(result, dtype=bool)
    if not y.any(axis=1).all():
        raise ValueError("A question has no correct candidate")
    return y


def hard_mask(bank):
    result=[]
    for row in bank:
        answers={parse_answer(c['text']) for c in row['candidates']}
        answers.discard(None)
        result.append(len(answers)>=2)
    return np.asarray(result,dtype=bool)


def score(choices,y,mask):
    ok=y[np.arange(len(y)),np.asarray(choices,dtype=np.int64)]
    def part(m):
        n=int(m.sum()); c=int(ok[m].sum())
        return {'correct':c,'n':n,'accuracy':100*c/n if n else None}
    return {'all':part(np.ones(len(y),dtype=bool)),'disagreement':part(mask)}


def make_bag_batch(seqs, rows, device):
    flat=[]
    for i in rows:
        flat.extend(range(i*4,i*4+4))
    selected=[seqs[i] for i in flat]
    qidx=[]; qoff=[0]; cidx=[]; coff=[0]
    for k in flat:
        qidx.extend(seqs[ k ][0]); qoff.append(len(qidx))
        cidx.extend(seqs[ k ][1]); coff.append(len(cidx))
    return (torch.tensor(qidx,dtype=torch.long,device=device),torch.tensor(qoff[:-1],dtype=torch.long,device=device),
            torch.tensor(cidx,dtype=torch.long,device=device),torch.tensor(coff[:-1],dtype=torch.long,device=device),flat)


class Net(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.table=nn.EmbeddingBag(VOCAB_SIZE,width,mode='mean',include_last_offset=False)
        self.fc1=nn.Linear(width*4+1024+8,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,1)
        self.drop=nn.Dropout(.35)
    def forward(self,qids,qoff,cids,coff,emb,extra):
        q=self.table(qids,qoff); c=self.table(cids,coff)
        x=torch.cat([q,c,torch.abs(q-c),q*c,emb,extra],dim=-1)
        h=self.drop(torch.relu(self.fc1(x)))
        h=self.drop(torch.relu(self.fc2(h)))
        return self.fc3(h).reshape(-1,4)


def export(path,model,mean,scale):
    np.savez(path,table=model.table.weight.detach().cpu().numpy(),
             mean=mean,scale=scale,
             w1=model.fc1.weight.detach().cpu().numpy().T,b1=model.fc1.bias.detach().cpu().numpy(),
             w2=model.fc2.weight.detach().cpu().numpy().T,b2=model.fc2.bias.detach().cpu().numpy(),
             w3=model.fc3.weight.detach().cpu().numpy().T,b3=model.fc3.bias.detach().cpu().numpy())


def main():
    start=time.perf_counter(); out=Path('runs/text_route')
    if (out/'selected').exists():
        raise FileExistsError('runs/text_route/selected exists; choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    print('Loading train and validation only...',flush=True)
    tb,ta=load_inputs('train'); vb,va=load_inputs('validation')
    ty=make_labels(tb,read_jsonl(DATA/'gsm8k_train.jsonl'))
    vy=make_labels(vb,read_jsonl(DATA/'gsm8k_validation.jsonl'))
    tm,vm=hard_mask(tb),hard_mask(vb)
    print(f"Disagreement groups: train {int(tm.sum())}/{len(tm)}, validation {int(vm.sum())}/{len(vm)}",flush=True)

    # Build deterministic hashed word and adjacent-token bags; no learned vocabulary.
    print('Hashing question and candidate text...',flush=True)
    tq,tc=text_sequences(tb); vq,vc=text_sequences(vb)
    train_seq=list(zip(tq,tc)); val_seq=list(zip(vq,vc))
    e_train,ex_train,_=auxiliary(tb,ta); e_val,ex_val,_=auxiliary(vb,va)
    # Fixed features are independent of labels; normalization fits training only.
    # Text embeddings are learned and trained end-to-end; append standardized
    # fixed vector and auxiliary data inside each mini-batch below.
    norm_train=np.concatenate([e_train,ex_train],axis=-1).reshape(-1,1032)
    norm_val=np.concatenate([e_val,ex_val],axis=-1).reshape(-1,1032)
    fixed_mean=norm_train.mean(0).astype(np.float32); fixed_scale=norm_train.std(0).astype(np.float32)
    fixed_scale=np.where(fixed_scale<1e-6,1.,fixed_scale).astype(np.float32)
    norm_train=np.clip((norm_train-fixed_mean)/fixed_scale,-10,10).reshape(len(tb),4,1032).astype(np.float32)
    norm_val=np.clip((norm_val-fixed_mean)/fixed_scale,-10,10).reshape(len(vb),4,1032).astype(np.float32)

    torch.set_num_threads(4); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training device: {device}',flush=True)
    if device.type=='cuda': print(torch.cuda.get_device_name(0),flush=True)
    seeds=[17,29,43]; all_results=[]; best_key=(-1,-1); best_dir=None
    train_y=torch.from_numpy(ty.astype(np.float32)).to(device)
    val_fixed=torch.from_numpy(norm_val).to(device)
    train_fixed=torch.from_numpy(norm_train).to(device)
    val_target=vy
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(seed)
        model=Net().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.02)
        best_hard=-1; best_all=-1; best_epoch=0; path=out/f'member_{seed}.npz'; history=[]
        for epoch in range(1,101):
            model.train(); order=torch.randperm(len(tb),device=device); total=0.
            for start_i in range(0,len(tb),64):
                idx=order[start_i:start_i+64].detach().cpu().numpy().tolist()
                qids,qoff,cids,coff,flat=make_bag_batch(train_seq,idx,device)
                emb=train_fixed.reshape(-1,1032)[flat,:1024]
                extra=train_fixed.reshape(-1,1032)[flat,1024:]
                logits=model(qids,qoff,cids,coff,emb,extra)
                y=train_y[idx]
                target=y/y.sum(dim=1,keepdim=True)
                rank=-(target*torch.log_softmax(logits,dim=1)).sum(1).mean()
                bce=nn.functional.binary_cross_entropy_with_logits(logits,y)
                loss=rank+.2*bce
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                total+=float(loss.item())*len(idx)
            model.eval(); val_logits=[]
            with torch.no_grad():
                for base in range(0,len(vb),64):
                    idx=list(range(base,min(base+64,len(vb))))
                    qids,qoff,cids,coff,flat=make_bag_batch(val_seq,idx,device)
                    fixed=val_fixed.reshape(-1,1032)[flat]
                    val_logits.extend(model(qids,qoff,cids,coff,fixed[:,:1024],fixed[:,1024:]).cpu().numpy())
            logits=np.asarray(val_logits)
            choices=[]
            from text_route_submission import auxiliary as _aux
            _,_,answers=_aux(vb,va)
            for group,values in zip(answers,logits):
                valid=[j for j,a in enumerate(group) if a is not None]
                choices.append(max(valid,key=lambda j:float(values[j])) if valid else int(np.argmax(values)))
            stats=score(choices,vy,vm); key=(stats['disagreement']['correct'],stats['all']['correct'])
            history.append({'epoch':epoch,'loss':total/len(tb),'validation':stats})
            if key>(best_hard,best_all):
                best_hard,best_all=key;best_epoch=epoch;export(path,model,fixed_mean,fixed_scale)
            if epoch==1 or epoch%10==0:
                print(f"seed={seed} epoch={epoch} loss={total/len(tb):.4f} hard={stats['disagreement']['accuracy']:.3f}% all={stats['all']['accuracy']:.3f}% best={best_epoch}",flush=True)
            if epoch-best_epoch>=18: break
        all_results.append({'seed':seed,'best_epoch':best_epoch,'best_validation_key':[best_hard,best_all],'history':history})
        # Reload exported NumPy model for exact deployment scoring.
        with np.load(path,allow_pickle=False) as m: model_arrays={k:m[k].copy() for k in m.files}
        from text_route_submission import candidate_features,scores_from_features
        z,_=candidate_features(vb,va,model_arrays['table'],model_arrays['mean'],model_arrays['scale'])
        choices=[]
        _,_,answers=auxiliary(vb,va)
        for group,values in zip(answers,scores_from_features(z,model_arrays)):
            valid=[j for j,a in enumerate(group) if a is not None]
            choices.append(int(max(valid,key=lambda j:float(values[j]))) if valid else int(np.argmax(values)))
        stats=score(choices,vy,vm)
        print(f'Seed {seed} exported model: {json.dumps(stats)}',flush=True)
        this_key=(stats['disagreement']['correct'],stats['all']['correct'])
        if this_key>best_key:
            best_key=this_key;best_dir=path
        del model,optimizer
    # Fixed selection: highest disagreement-validation accuracy, then all-validation accuracy.
    ensemble_dir=out/'selected';ensemble_dir.mkdir()
    # Save the best single member to keep assets compact and avoid adding another selection degree.
    # Its exact validation score is recorded above; the per-seed comparison is in the log.
    shutil.copy2(best_dir,ensemble_dir/'model.npz')
    write_json(ensemble_dir/'config.json',{'model':'model.npz','feature':'hashed word/bigram EmbeddingBag + supplied vector','seeds_compared':seeds})
    size=sum(p.stat().st_size for p in ensemble_dir.iterdir() if p.is_file())
    if size>16*1024*1024: raise RuntimeError('Inference assets exceed 16 MiB')
    write_json(out/'experiments.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'seeds':seeds,'selected_seed_model':best_dir.name,'asset_bytes':size,'training_seconds':time.perf_counter()-start,'validation_results':all_results,'selection':'highest disagreement validation count; then all validation count'})
    print(f'Finished. Selected={best_dir.name}; assets={size/1024**2:.3f} MiB; checkpoint={ensemble_dir}',flush=True)


if __name__=='__main__': main()
