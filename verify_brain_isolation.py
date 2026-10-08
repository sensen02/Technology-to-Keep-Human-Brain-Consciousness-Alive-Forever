"""Run bounded regression suites and save reproducible evidence, not biology proof.
No shell expansion; project-owned scripts only. Output records current source hashes.
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'brain_isolation'
SUITES=['run_cond_selftest.py','run_cable_geometry_selftest.py',
        'run_electrode_selftest.py','run_local_tissue_selftest.py',
        'run_hybrid_selftest.py','run_isolation_selftest.py',
        'run_graded_vision_selftest.py','run_injury_tissue_selftest.py',
        'run_multirate_selftest.py','run_da_protocol_selftest.py',
        'run_synapse_map_selftest.py','run_da_real_comparison_selftest.py',
        'run_electrode_damage_selftest.py','run_neck_cut_selftest.py',
        'run_touch_damage_selftest.py',
        'run_receptor_unit_selftest.py',
        'audit_evidence_consistency.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-missing',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for name in SUITES:
        path=ROOT/name
        if not path.exists():
            rows.append({'script':name,'status':'missing'});continue
        started=time.perf_counter()
        try:
            proc=subprocess.run([sys.executable,'-B',str(path)],cwd=ROOT,capture_output=True,text=True,timeout=180)
            row={'script':name,'returncode':proc.returncode,'status':'passed' if proc.returncode==0 else 'failed',
                 'wall_seconds':time.perf_counter()-started,'stdout':proc.stdout,'stderr':proc.stderr}
        except subprocess.TimeoutExpired as exc:
            row={'script':name,'status':'timeout','wall_seconds':time.perf_counter()-started}
        rows.append(row);print(name,row['status'],flush=True)
    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
            for x in sorted((ROOT/'engine').glob('*.py'))}
    for name in SUITES:
        if (ROOT/name).is_file():hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    versions={}
    for package in ('numpy','scipy','matplotlib','neuron'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    passed=all(r['status']=='passed' or (args.allow_missing and r['status']=='missing') for r in rows)
    report={'scope':'numerical implementation regression; not fruit-fly calibration or consciousness evidence',
            'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'versions':versions,'source_sha256':hashes,'suites':rows,'passed':passed,
            'allow_missing':args.allow_missing,'missing_count':sum(r['status']=='missing' for r in rows)}
    (OUT/'verification_report.json').write_text(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
