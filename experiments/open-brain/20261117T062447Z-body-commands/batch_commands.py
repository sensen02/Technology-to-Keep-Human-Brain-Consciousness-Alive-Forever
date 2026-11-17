"""Dispatch predeclared whole episodes and record each child exit status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import expansion
import cases


def batch(split, mode, out, checkpoint, jobs):
    sets = {'dev': expansion.DEVELOP, 'correct': expansion.CORRECT,
            'held': (*expansion.HELD, expansion.CUT), 'old': cases.HELD}
    script = expansion.FIRST/'drive.sh' if split == 'old' else expansion.HERE/'drive.sh'
    filename = 'body_run.py' if split == 'old' else 'expansion.py'
    out = out.resolve()
    out.mkdir()
    def run(case):
        cmd = [str(script), filename, mode, case.name, str(out/case.name)]
        if checkpoint:
            cmd += ['--checkpoint',str(checkpoint.resolve())]
        with (out/f'{case.name}.log').open('x') as log:
            code = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        (out/f'{case.name}.exit').write_text(f'{code}\n')
        print(case.name, code, flush=True)
        return code
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return int(any(tuple(pool.map(run, sets[split]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('dev','correct','held','old'))
    parser.add_argument('mode', choices=('teacher','evaluate','correct'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--jobs',type=int,default=8)
    args=parser.parse_args()
    raise SystemExit(batch(args.split,args.mode,args.out,args.checkpoint,args.jobs))
