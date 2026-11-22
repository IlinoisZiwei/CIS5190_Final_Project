"""Install process and data-access guards before importing experiment code."""
import json
import math
import os
from pathlib import Path
import resource
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
limit = int(sys.argv[1])
module = sys.argv[2]
sys.argv = [module, *sys.argv[3:]]
resource.setrlimit(resource.RLIMIT_CPU, (max(1, limit-1), limit))
manifest = json.loads((ROOT/'data/manifests/approved_inputs.json').read_text())
allowed = {(ROOT/v['path']).resolve() for v in manifest['data'].values()}
cache = ROOT/'experiments/historical/v3_feature_regularization_2026-10-29'
allowed.update((cache/name).resolve() for name in ['fixed_grid_complete.npz', 'nested_oof.npz'])
controlled = (ROOT/'experiments/New Experiments', ROOT/'models', ROOT/'deliverables')

def guard(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'os.forkpty'):
        raise PermissionError('Experiment workers cannot spawn child processes.')
    if event in ('socket.connect', 'socket.bind'):
        raise PermissionError('Experiment workers are offline.')
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    lower = str(path).lower()
    if any(t in lower for t in ('external_metadata.csv', 'external_test', 'sealed', 'holdout')):
        raise PermissionError('Sealed or mixed data access is prohibited.')
    # Dataset and checkpoint formats require explicit inputs or new-run outputs.
    if path.suffix.lower() in ('.csv', '.tsv', '.jsonl', '.npz', '.npy', '.pt', '.pth', '.joblib'):
        if path not in allowed and not any(path.is_relative_to(p) for p in controlled):
            # Numerical package resources are not project datasets.
            if 'site-packages' not in path.parts:
                raise PermissionError(f'Input is not explicitly approved: {path}')

sys.addaudithook(guard)
runpy.run_module(module, run_name='__main__')
