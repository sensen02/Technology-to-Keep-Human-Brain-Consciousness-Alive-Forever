import subprocess,sys,json,hashlib
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'holdout/20261106T200439Z';O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists();f=R/'operating_point/20261106T195609Z/FROZEN_CANDIDATE.json';frozen=json.loads(f.read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in frozen['hashes'].items());old=D/'operating_point_20261106T195609Z.py';new=D/'holdout_20261106T200439Z.py'
p=dict(importance='核心必需',frozen_candidate_path=str(f),frozen_candidate_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),holdout=frozen['next_holdout'],change_ledger=dict(old_file=str(old),old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_file=str(new),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),reason='execute frozen out-of-selection perturbations/finer dt, all selected parameters fixed; no load extension'))
(O/'PREREGISTRATION.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
for vel in ['0','-.2','-.05','.05','.2']:subprocess.run([sys.executable,'-B',str(new),vel],check=True)
