"""Evaluate a frozen checkpoint with bounded worker count."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
out=root/"RUNS"
out.mkdir()
scenes=json.loads((root/"SCENES.json").read_text())

def run(case):
    with (out/f"{case}.log").open("x") as log:
        status=subprocess.run([str(root/"drive.sh"),"run.py",case,str(out/case)],
                              stdout=log,stderr=subprocess.STDOUT,check=False).returncode
    (out/f"{case}.exit").write_text(f"{status}\n")
    print(case,status,flush=True)
    return status

with ThreadPoolExecutor(max_workers=5) as pool:
    errors=list(pool.map(run,scenes))
raise SystemExit(int(any(errors)))
