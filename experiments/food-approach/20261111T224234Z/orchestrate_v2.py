"""Pure-stdlib orchestration, inherited sequential pipeline; minimal v2 entry names."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def build_source():
 source=(ROOT/'orchestrate.py').read_text()
 for old,new in [("run('runner.py'","run('runner_v2.py'"),("run('audit.py'","run('audit_v2.py'"),("run('summarize_B.py'","run('summarize_B_v2.py'")]:
  if source.count(old)!=1:raise ValueError('orchestration source adaptation mismatch '+old)
  source=source.replace(old,new)
 return source
if __name__=='__main__':
 ns={'__name__':'_AB_sequential_orchestration_v2','__file__':str(Path(__file__).resolve())};exec(compile(build_source(),str(Path(__file__).resolve())+'::unchanged_sequential_pipeline','exec'),ns);ns['main']()
