"""Bound worker count and save exit codes separately from behavior gates."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
scenes = json.loads((ROOT/"SCENES.json").read_text())["scenes"]
root = ROOT/"RUNS"
root.mkdir()

def job(case):
    scene, policy = case
    tag = f"{scene}__{policy}"
    with (root/f"{tag}.log").open("x") as log:
        status = subprocess.run([str(ROOT/"drive.sh"),"evaluate.py",scene,policy,str(root/tag)],
                                stdout=log,stderr=subprocess.STDOUT,check=False).returncode
    (root/f"{tag}.exit").write_text(f"{status}\n")
    print(tag,status,flush=True)
    return status

# Long courses start first so shorter controls can reuse finished worker slots.
jobs = sorted(((name,policy) for name,cfg in scenes.items() for policy in cfg["policies"]),
              key=lambda case:scenes[case[0]]["ms"],reverse=True)
with ThreadPoolExecutor(max_workers=5) as pool:
    codes = list(pool.map(job,jobs))
raise SystemExit(1 if any(codes) else 0)
