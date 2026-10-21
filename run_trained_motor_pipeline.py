"""One bounded training -> frozen artifact -> body preflight entry.

INTERFACE_TEST only, not behavioral learning. Default never integrates body.
Body trials require a separate explicit --evaluate-body and a legal initial gate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--evaluate-body', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    artifact = args.output/'trained.json'
    train = [sys.executable, '-B', str(here/'train_motor_translation.py'),
             '--manifest', str(args.manifest.resolve()), '--dataset', str(args.dataset.resolve()),
             '--output', str(artifact.resolve()), '--steps', str(args.steps),
             '--learning-rate', str(args.learning_rate)]
    result = subprocess.run(train, capture_output=True, text=True)
    (args.output/'training.log').write_text(result.stdout+result.stderr)
    report = dict(training_exit=result.returncode, body_attempted=False, behavior_success=False)
    if result.returncode == 0:
        body = args.output/'body'; body.mkdir()
        (body/'PREREGISTRATION.md').write_text('Training artifact integration. INTERFACE_TEST not behavioral learning. No pose/physics changes. Default preflight only.\n')
        command = [sys.executable, '-B', str(here/'run_motor_body_integration.py'),
                   '--output', str(body.resolve()), '--motor-translator', str(artifact.resolve()),
                   '--initial-state', str(args.initial_state.resolve())]
        if not args.evaluate_body: command.append('--preflight-only')
        evaluated = subprocess.run(command, capture_output=True, text=True)
        (args.output/'body.log').write_text(evaluated.stdout+evaluated.stderr)
        report.update(body_entry_exit=evaluated.returncode, body_attempted=args.evaluate_body,
                      body_summary=str((body/'summary.json').resolve()))
    (args.output/'pipeline.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return result.returncode or report.get('body_entry_exit', 1)


if __name__ == '__main__':
    raise SystemExit(main())
