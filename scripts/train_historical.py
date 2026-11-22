"""Optional NEW v1-v5 run. Never invoked by default notebook review.

Creates an exclusive new output directory. Does not execute historical source, resume
the exhausted follow-up study, or create a leaderboard-compatible checkpoint.
"""
from pathlib import Path
import argparse
import json
import math
import os
import resource
import signal
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version",required=True,choices=["v1","v2","v3","v4","v5"])
    parser.add_argument("--cpu-budget",required=True,type=float,
                        help="Separately authorized new process CPU seconds, including imports")
    parser.add_argument("--pretrained-dir",type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.cpu_budget) or args.cpu_budget <= 0:
        parser.error("A positive finite new CPU budget is required")
    if args.version in {"v4","v5"} and (not args.pretrained_dir or not args.pretrained_dir.is_dir()):
        parser.error("V4/v5 require an explicit local fixed-revision MiniLM directory")
    limit = max(1,math.ceil(args.cpu_budget))
    resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))
    def exhausted(signum,frame):
        raise TimeoutError("New run CPU budget exhausted; no automatic continuation")
    signal.signal(signal.SIGXCPU,exhausted)
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("MPLCONFIGDIR",str(Path(__import__("tempfile").gettempdir())/"dragonthree-mpl"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from dragonthree.project_io import ROOT, labeled_urls, install_data_guard
    install_data_guard()
    import joblib
    from dragonthree.models_v1_v5 import fit_v1,LexicalV2V3,EnsembleV4,HybridV5,metrics_from_scores
    out = ROOT/"experiments"/"runs"/(time.strftime("%Y%m%d_%H%M%S")+"_"+args.version+"_"+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    record = {"version":args.version,"authorized_new_cpu_budget":args.cpu_budget,
              "status":"started","original_models_overwritten":False,
              "sealed_data_accessed":False,
              "note":"New diagnostic fit, not archived experiment evidence or leaderboard export"}
    (out/"run.json").write_text(json.dumps(record,indent=2)+"\n")
    try:
        train,dev = labeled_urls("course_original_train"),labeled_urls("course_dev")
        texts,y = train.text.tolist(),train.y.to_numpy()
        if args.version == "v1":
            model,history = fit_v1(texts,y,dev.text.tolist(),dev.y.to_numpy())
            record["history"] = history
        elif args.version in {"v2","v3"}:
            model = LexicalV2V3(args.version).fit(texts,y)
        else:
            v4 = EnsembleV4(args.pretrained_dir).fit(texts,y)
            model = v4 if args.version=="v4" else HybridV5(v4,args.pretrained_dir).fit(texts,y)
        record["development"] = metrics_from_scores(dev.y.to_numpy(),model.decision_function(dev.text.tolist()))
        joblib.dump(model,out/"diagnostic_model.joblib")
        record["status"] = "complete"
        record["checkpoint"] = "diagnostic_model.joblib"
    except BaseException as exc:
        record["status"] = "incomplete"
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        r = resource.getrusage(resource.RUSAGE_SELF)
        record["process_cpu_seconds"] = r.ru_utime+r.ru_stime
        (out/"run.json").write_text(json.dumps(record,indent=2)+"\n")
        print(out)

if __name__ == "__main__":
    main()
