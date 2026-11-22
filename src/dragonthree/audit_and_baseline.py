"""E0: audit approved snapshots and reproduce the original validated V4 scores.

No training or data collection occurs. Historical labels/groups remain unchanged.
The source checkpoint with 4,800 training rows is not scored on course development.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from collections import Counter, defaultdict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/New Experiments/E0'
PKG=ROOT

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def metric(y, score):
    pred=(np.asarray(score)>0).astype(int)
    y=np.asarray(y,dtype=int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(n=len(y), correct=int((pred==y).sum()), accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)), macro_f1=float(f1_score(y,pred,average='macro')),
                auc=float(roc_auc_score(y,score)), fox_recall=float(cm[0,0]/cm[0].sum()),
                nbc_recall=float(cm[1,1]/cm[1].sum()), confusion_matrix=cm.tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((PKG/'data/manifests/approved_inputs.json').read_text())['data']
    preprocess=load_module(ROOT/'models/original_v4/validated/preprocess.py','reviewed_preprocess')
    all_data={}; audit={}
    for key, entry in manifest.items():
        path=PKG/entry['path']
        raw=path.read_bytes()
        if path.suffix=='.csv':
            with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        else: rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if len(rows)!=entry['rows']:raise ValueError(f'Unexpected row count: {key}')
        all_data[key]=rows
        audit[key]={'rows':len(rows),'role':entry['role'],'path':entry['path'],
                    'sha256':hashlib.sha256(raw).hexdigest(),'columns':list(rows[0]) if rows else []}
    rows=all_data['audited_training']
    profile=defaultdict(Counter); bad=[]; text_labels=defaultdict(set); group_origins=defaultdict(set)
    for row in rows:
        text=row['text']; origin=row['origin']; label=row['label']
        if isinstance(label,str): label={'FoxNews':0,'NBC':1,'0':0,'1':1}.get(label,label)
        derived=preprocess.label_from_url(row['url'])
        expected={'FoxNews':0,'NBC':1}.get(derived)
        if label!=expected or text!=preprocess.url_to_text(row['url']):
            bad.append({'id':row['id'],'origin':origin,'label_mismatch':label!=expected,'text_mismatch':text!=preprocess.url_to_text(row['url'])})
        n=len(text.split()); length='0' if n==0 else '1-3' if n<=3 else '4-5' if n<=5 else '6-8' if n<=8 else '9-12' if n<=12 else '13+'
        profile[origin][f'label={label};length={length}']+=1
        text_labels[preprocess.text_group(text)].add(label)
        group_origins[row['group']].add(origin)
    duplicate_counts=Counter(preprocess.text_group(r['text']) for r in rows)
    overlap={}
    train_groups=set(duplicate_counts)
    for panel in ['course_dev','legacy_dev','paired_dev']:
        dev=all_data[panel]
        overlap[panel]=sum(preprocess.text_group(preprocess.url_to_text(r['url'])) in train_groups for r in dev)
    save(OUT/'data_manifest.json',audit)
    save(OUT/'data_audit.json',{'training_rows':len(rows),'origin_counts':dict(Counter(r['origin'] for r in rows)),
         'label_or_preprocess_mismatches':bad,'duplicate_text_groups':sum(v>1 for v in duplicate_counts.values()),
         'conflicting_label_text_groups':sum(len(v)>1 for v in text_labels.values()),
         'approved_dev_exact_text_overlap_rows':overlap,
         'profile':{k:dict(v) for k,v in profile.items()},
         'action':'No rows or groups changed. Historical grouping retained; no sealed data inspected.',
         'human_review_status':'A reproducible review queue is supplied; independent human review is not claimed.'})
    # A review queue, not a claim that a person has independently validated the rows.
    rng=np.random.default_rng(5190); sample=[]
    for origin,n in [('course',40),('new',40),('legacy',20)]:
        candidates=[r for r in rows if r['origin']==origin]
        for label in (0,1):
            pool=[r for r in candidates if str(r['label']) in (str(label),('FoxNews','NBC')[label])]
            chosen=rng.choice(len(pool),size=n//2,replace=False)
            sample.extend(pool[i] for i in chosen)
    paired=defaultdict(list)
    for r in rows:
        if r['origin']=='paired':paired[r['group']].append(r)
    eligible=[g for g,rs in paired.items() if len(rs)==2 and len(set(str(r['label']) for r in rs))==2]
    for g in rng.choice(eligible,size=10,replace=False):sample.extend(paired[g])
    with (OUT/'review_queue_120.jsonl').open('w') as f:
        for r in sample:f.write(json.dumps(dict(r,review_status='pending_human_review'),ensure_ascii=False)+'\n')
    # Load only the copied, reviewed inference module and known validated checkpoint.
    folder=ROOT/'models/original_v4/validated'
    model_module=load_module(folder/'model.py','reviewed_v4_validated')
    torch.set_num_threads(1)
    model=model_module.get_model()
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)); model.eval()
    torch.set_num_threads(1)
    result={'model':'original_v4_validated','training_rows':3849,'evaluation_level':'reused development',
            'torch':torch.__version__,'panels':{}}
    predictions=[]
    for panel in ['course_dev','legacy_dev','paired_dev']:
        panel_rows=all_data[panel]
        texts=[preprocess.url_to_text(r['url']) for r in panel_rows]
        y=[{'FoxNews':0,'NBC':1}[preprocess.label_from_url(r['url'])] for r in panel_rows]
        scores=model.decision_function(texts)
        result['panels'][panel]=metric(y,scores)
        predictions.extend({'panel':panel,'row':i,'actual':int(label),'score':float(score),
                            'prediction':int(score>0),'correct':bool(int(score>0)==label),
                            'length':len(texts[i].split())} for i,(label,score) in enumerate(zip(y,scores)))
    save(OUT/'baseline_metrics.json',result)
    with (OUT/'baseline_predictions.jsonl').open('w') as f:
        for row in predictions:f.write(json.dumps(row)+'\n')
    provenance={}
    for mode in ('validated','submission'):
        folder=ROOT/'models/original_v4'/mode
        provenance[mode]={'manifest':json.loads((folder/'manifest.json').read_text()),'files':{}}
        for name in ('model.py','preprocess.py','model.pt'):
            path=folder/name
            provenance[mode]['files'][name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    save(OUT/'original_model_provenance.json',provenance)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
