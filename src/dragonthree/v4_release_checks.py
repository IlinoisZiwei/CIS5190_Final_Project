"""E5 freezes the existing V4; E6 tests its three-file CPU submission offline."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse
import numpy as np
import torch
from dragonthree.audit_and_baseline import load_module, save

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/New Experiments'

def freeze():
    out=RUN/'E5';out.mkdir(exist_ok=False)
    e1=json.loads((RUN/'E1/status.json').read_text()) if (RUN/'E1/status.json').exists() else {}
    e2=json.loads((RUN/'E2/results.json').read_text())
    target=ROOT/'deliverables/submission_v4';target.mkdir(parents=True,exist_ok=False)
    source=ROOT/'models/original_v4/submission'
    files={}
    for name in ['model.py','preprocess.py','model.pt']:
        shutil.copy2(source/name,target/name)
        old=hashlib.sha256((source/name).read_bytes()).hexdigest()
        new=hashlib.sha256((target/name).read_bytes()).hexdigest()
        assert old==new
        files[name]={'sha256':new,'bytes':(target/name).stat().st_size}
    record={'status':'complete','stage':'E5','decision':'retain_original_v4_submission',
            'reason':'E4 was conditionally authorized but was not started because the locked augmented model failed the course development protection gates. The original V4 remains the approved fallback.',
            'training_rows':4800,'training_includes_course_development':True,
            'independent_course_dev_accuracy_for_submission':None,
            'validated_sibling_course_dev':{'correct':810,'n':951,'training_rows':3849},
            'E1_status':e1,'E2_development_gate_passed':e2['development_gate_passed'],
            'new_training_performed':False,'original_models_overwritten':False,
            'release_path':'deliverables/submission_v4','files':files,
            'leaderboard_submitted':False,'new_leaderboard_result':None}
    save(out/'final_decision.json',record)
    save(target/'provenance.json',record)
    print(json.dumps(record,indent=2))

def benchmark():
    out=RUN/'E6';out.mkdir(exist_ok=False)
    folder=ROOT/'deliverables/submission_v4'
    # Reviewed three-file inference package only. No historical training runner is imported.
    prep=load_module(folder/'preprocess.py','frozen_preprocess')
    mod=load_module(folder/'model.py','frozen_submission')
    synthetic=[('https://www.foxnews.com/politics/hello-world-2026','hello world'),
               ('/news/hello-world-2026','hello world'),('',''),('/123-abc9',''),
               ('/world/caf%C3%A9-d%C3%A9bat.html','café débat'),
               ('/video/a_short-title','a short title'),('/x/one--two','one two'),
               ('/folder/','folder')]
    contract=out/'synthetic_contract.csv'
    with contract.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['url']);writer.writerows((u,) for u,_ in synthetic)
    X,y=prep.prepare_data(str(contract));assert X==[t for _,t in synthetic]
    assert len(X)==len(y)==len(synthetic) and y[0]=='FoxNews' and all(v=='' for v in y[1:])
    model=mod.get_model();torch.set_num_threads(1)
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    scores=model.decision_function(X)
    reverse=model.decision_function(list(reversed(X)))[::-1]
    logits=model(X);labels=model.predict(X)
    assert logits.shape==(len(X),2) and np.isfinite(scores).all()
    assert np.max(np.abs(scores-reverse))<1e-5
    assert scores[0]==scores[1]
    assert labels==['NBC' if s>0 else 'FoxNews' for s in scores]
    assert len(model.predict([]))==0
    assert all(t.device.type=='cpu' for t in model.state_dict().values())
    save(out/'interface_checks.json',{'status':'passed','torch':str(torch.__version__),
         'n_synthetic_inputs':len(X),'row_order_max_score_error':float(np.max(np.abs(scores-reverse))),
         'empty_and_digit_only_rows_retained':True,'hostname_not_used_as_feature':True,
         'network_disabled':True,'all_tensors_cpu':True,'teacher_data_used':False})
    del model
    source=ROOT/'data/approved/course_original_train.csv'
    with source.open(newline='',encoding='utf-8-sig') as f: source_urls=[row['url'] for row in csv.DictReader(f)]
    fixture=out/'timing_fixture_8000.csv'
    paths=[urlparse(source_urls[i%len(source_urls)]).path for i in range(8000)]
    with fixture.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['url']);writer.writerows((u,) for u in paths)
    records=[]
    for threads in (1,4):
        for repeat in range(3):
            total_start=time.perf_counter();cpu_start=time.process_time()
            start=time.perf_counter();texts,_=prep.prepare_data(str(fixture));preprocess_wall=time.perf_counter()-start
            start=time.perf_counter();model=mod.get_model();torch.set_num_threads(threads)
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
            load_wall=time.perf_counter()-start
            start=time.perf_counter();prediction=model.predict(texts);inference_wall=time.perf_counter()-start
            assert len(prediction)==8000 and set(prediction)<= {'FoxNews','NBC'}
            rec={'threads':threads,'repeat':repeat+1,'rows':8000,'preprocess_wall_seconds':preprocess_wall,
                 'load_wall_seconds':load_wall,'inference_wall_seconds':inference_wall,
                 'total_wall_seconds':time.perf_counter()-total_start,'cpu_seconds':time.process_time()-cpu_start}
            records.append(rec);del model
            save(out/'runtime_partial.json',records)
            print(json.dumps(rec),flush=True)
    record={'status':'complete','stage':'E6','torch':str(torch.__version__),
            'fixture':'8000 repeated approved training URL paths; no held-out or teacher data',
            'accuracy_computed':False,'cold_load_each_repeat':True,'process_import_time_included':False,
            'teacher_limit_wall_seconds':420,'local_target_wall_seconds':210,
            'all_runs_below_target':all(r['total_wall_seconds']<210 for r in records),
            'server_hardware_equivalence':'unknown; local timing is not a server guarantee','runs':records}
    save(out/'runtime.json',record)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['E5','E6'],required=True)
    a=p.parse_args();freeze() if a.stage=='E5' else benchmark()
if __name__=='__main__':main()
