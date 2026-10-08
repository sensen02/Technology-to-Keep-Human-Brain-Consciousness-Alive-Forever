"""Top-level orchestrator: runs every implemented layer and collects metrics.

Each stage runs in its own process; a failing stage is recorded, never hidden.
This does not run experiments. It runs code written in this project.
"""
from pathlib import Path
import json, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / 'venv/bin/python')
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

STAGES = [
    ('model_self_test', [PY, '-c',
      "import json,model; print(json.dumps(model.self_test(), default=float))"]),
    ('measurement_unit_test', [PY, str(ROOT / 'check_measurement.py')]),
    ('validation', [PY, str(ROOT / 'run_validation.py')]),
    ('mechanics', [PY, str(ROOT / 'run_mechanics.py')]),
    ('cellstate', [PY, str(ROOT / 'run_cellstate.py')]),
    ('late_phase', [PY, str(ROOT / 'run_late.py')]),
]


def main():
    summary = {}
    for name, cmd in STAGES:
        targets = [ROOT / p for p in ('run_cellstate.py', 'run_late.py')]
        if not (ROOT / 'run_mechanics.py').exists():
            pass
        if name in ('cellstate', 'late_phase') and not (ROOT / f'run_{"cellstate" if name=="cellstate" else "late"}.py').exists():
            summary[name] = {'status': 'no_runner_yet'}
            continue
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        summary[name] = {'status': 'ok' if proc.returncode == 0 else 'failed',
                         'returncode': proc.returncode,
                         'seconds': round(time.perf_counter() - start, 2),
                         'stderr_tail': proc.stderr[-2000:] if proc.returncode else ''}
    (OUT / 'run_all_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
