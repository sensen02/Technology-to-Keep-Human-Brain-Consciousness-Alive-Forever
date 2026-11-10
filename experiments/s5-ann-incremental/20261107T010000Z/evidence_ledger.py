"""Record current sources and user-authorized rule changes without rewriting history."""
from pathlib import Path
import json,hashlib,subprocess,datetime
D=Path(__file__).parent;R=Path('/run/media/sensen/Data2/cell_wound_prototype');O=Path('/home/sensen/Desktop/engine修复验收/s5-ann-incremental/20261107T010000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[R/'AGENTS.md',R/'GOAL_ROADMAP.zh-CN.md',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py',R/'tools/dynamic_support_metrics.py',R/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')]
ledger=[]
for name in ['AGENTS.md','GOAL_ROADMAP.zh-CN.md']:
 old=subprocess.check_output(['git','show','HEAD:'+name],cwd=R)
 ledger.append(dict(old_file=str(R/name)+' (clean HEAD before this turn)',old_sha256=hashlib.sha256(old).hexdigest(),new_file=str(R/name),new_sha256=sha(R/name),reason='Explicit latest user authorization: incremental actuator, ANN translator or replacement controller allowed; muscle route stopped'))
for new,old in [('incremental_actuator.py','experiments/s5-actuator-descending/20261106T161620Z/joint_load.py'),('ann_translator.py','experiments/s5-neural-translator/20261105T200021Z/translator.py'),('learn_stand.py','experiments/s5-neural-translator/20261105T200021Z/experiment.py')]:
 ledger.append(dict(old_file=str(R/old),old_sha256=sha(R/old),new_file=str(D/new),new_sha256=sha(D/new),reason='Full free-body incremental actuator and all12000-cell temporal ANN, newly trained incremental actions; original files preserved'))
p=O/'CHANGE_LEDGER.json';assert not p.exists();p.write_text(json.dumps(dict(importance='核心必需',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ledger=ledger,supplemental_dependency_hashes={str(f):sha(f) for f in files},timing='supplemental dependency snapshot during first training, not retroactively claimed as preregistration; simulation preregistration is separately saved'),ensure_ascii=False,indent=2)+'\n')
print(str(p))
