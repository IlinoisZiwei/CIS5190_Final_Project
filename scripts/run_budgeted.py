"""Run one reviewed experiment with a shared, persistent CPU allowance.

The worker is a single process (numerical threads are allowed). Descendant
processes and network access are disabled by worker_bootstrap.py. wait4 records
all user and system CPU consumed by that worker, including failed runs.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments' / 'New Experiments'
CAPS = {'E1': 600.0, 'E2': 300.0, 'E4': 600.0, 'other': 300.0}

def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6'])
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--module', required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.module.startswith('dragonthree.'):
        p.error('Only reviewed dragonthree package modules are accepted.')
    if (ROOT / 'configs/BASE_RUN_CLOSED.txt').exists():
        raise RuntimeError('The completed base run is closed. A separately approved run configuration is required.')
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'ledger.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RUN / 'cpu_ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else {
            'new_budget_cpu_seconds': 1800.0, 'historical_budget_closed_cpu_seconds': 298.88928,
            'accounting': 'wait4 worker user+system; threads included; child spawning blocked',
            'stage_caps_cpu_seconds': CAPS, 'runs': [], 'cpu_used_seconds': 0.0}
        if any(r['status'] == 'running' for r in ledger['runs']):
            raise RuntimeError('Unclosed process reservation: reconcile it before further work.')
        ledger['stage_caps_cpu_seconds'] = CAPS
        bucket = a.stage if a.stage in ('E1', 'E2', 'E4') else 'other'
        used = sum(r.get('cpu_seconds', 0) for r in ledger['runs'] if r['bucket'] == bucket)
        remaining = min(CAPS[bucket] - used, 1800 - ledger['cpu_used_seconds'])
        # Reserve one CPU second for signal delivery and shutdown at the hard limit.
        limit = math.floor(remaining - 1)
        if limit < 2:
            raise RuntimeError('The approved stage or total CPU allowance is exhausted.')
        run_id = f'{len(ledger["runs"])+1:02d}_{a.stage}'
        record = {'id': run_id, 'stage': a.stage, 'bucket': bucket, 'module': a.module,
                  'arguments': a.arguments, 'status': 'running', 'cpu_limit_seconds': limit,
                  'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        ledger['runs'].append(record)
        write_json(path, ledger)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', MPLCONFIGDIR=str(RUN / 'mpl_cache'))
        args = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
        command = [a.python, str(ROOT/'scripts/worker_bootstrap.py'), str(limit), a.module, *args]
        start = time.monotonic()
        with (RUN / f'{run_id}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            write_json(path, ledger)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        cpu = usage.ru_utime + usage.ru_stime
        record.update(status='complete' if process.returncode == 0 else 'failed_or_budget_stopped',
                      exit_code=process.returncode, user_cpu_seconds=usage.ru_utime,
                      system_cpu_seconds=usage.ru_stime, cpu_seconds=cpu,
                      wall_seconds=time.monotonic()-start, peak_rss_native=usage.ru_maxrss)
        ledger['cpu_used_seconds'] = sum(r.get('cpu_seconds', 0) for r in ledger['runs'])
        ledger['cpu_remaining_seconds'] = 1800-ledger['cpu_used_seconds']
        write_json(path, ledger)
        print(json.dumps(record, indent=2))
        print((RUN/f'{run_id}.log').read_text()[-6000:])
        sys.exit(0 if process.returncode == 0 else 1)

if __name__ == '__main__':
    main()
