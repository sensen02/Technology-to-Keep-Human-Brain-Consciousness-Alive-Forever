"""Recover failed preflight import without overwriting first-attempt evidence or extending training."""
from pathlib import Path
import json,hashlib,ast
from adapt_source import build
HERE=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 jobs=json.loads((HERE/'JOBS.json').read_text())
 if len(jobs)!=1 or jobs[0]['stage']!='PREFLIGHT' or jobs[0]['exit_code']!=1:raise ValueError('only verified first preflight failure can recover')
 if any((HERE/p).exists() for p in ('FROZEN.json','DATASET.json','TRAIN/checkpoint_000000.pt')):raise ValueError('no model updates or partial data may be blindly rerun')
 if not any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in ast.parse((HERE/'generate_data.py').read_text()).body):raise ValueError('public data generator main required')
 evidence={str(HERE/n):sha(HERE/n) for n in ('PIPELINE.log','PREFLIGHT_EXECUTION.log','JOBS.json','run_recovery_v1.py','generate_data.py')}
 report=dict(first_job='bash-91 exit1 collected',error='ImportError generate_data public main missing',first_stages=1,first_training_updates=0,first_Core_constructions=0,repair='Export same data generator as public main before first data generation or source freezing',evidence_sha256=evidence,training_budget=1000,budget_extension=False,old_sources_changed=['generate_data.py in this not-yet-frozen candidate only'],first_logs_preserved=True)
 with (HERE/'RECOVERY_V1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 source=build('run_candidate.py').replace("HERE/'JOBS.json'","HERE/'JOBS_RECOVERY_V1.json'").replace("('preflight.py',[],'PREFLIGHT')","('preflight.py',[],'PREFLIGHT_RECOVERY_V1')")
 ns={'__name__':'_opponent_preflight_import_recovery','__file__':str(HERE/'run_recovery_v1.py')};exec(compile(source,str(HERE/'run_recovery_v1.py')+'::same_pipeline_new_logs','exec'),ns);ns['main']()
if __name__=='__main__':main()
