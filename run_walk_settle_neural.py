"""Actual walking, quintic assisted settling, hold, then fresh neural control."""
from pathlib import Path
import ast,json,hashlib
import run_physical_walk_prewarm as w
import run_neural_muscle_closure as c
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-descending-command/20261028T071000Z/run_physical_walk_prewarm.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=Path(w.__file__);current=source.read_text();old=OLD.read_text()
 rev=current.replace(',warmup_settle_s=0.,warmup_hold_s=0.):','):')
 lines=rev.splitlines();lines=[l for l in lines if "raise ValueError('nonnegative finite appended warmup durations required')" not in l];rev='\n'.join(lines)+'\n'
 rev=rev.replace('warmup_settle_s=warmup_settle_s,warmup_hold_s=warmup_hold_s,handover_time_s=duration+warmup_settle_s+warmup_hold_s','handover_time_s=duration').replace('n=round((duration+warmup_settle_s+warmup_hold_s)/dt)','n=round(duration/dt)')
 import re
 rev=re.sub(r'\n +if warmup_settle_s\+warmup_hold_s>0 and step>=round\(duration/dt\):\n.*?\n +torque=kp', '\n   torque=kp',rev,flags=re.S)
 assert ast.dump(ast.parse(rev),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False),'Unexpected change outside optional settling stage'
 files=[Path(__file__),source,P/'engine/embodied/continuous_handover.py',P/'engine/embodied/motor_output.py',P/'engine/embodied/sensory_channels.py',P/'engine/embodied/runtime_pool_guard.py',P/'engine/neural_cond.py',P/'engine/embodied/adapters.py',P/'engine/neck_cut_data.py',P/'data/flywire/banc_connectome.npz',c.POOLS,w.STANCE,w.REF,w.BASE/'INTEGRATED_MODEL.xml'];hashes={str(x):sha(x) for x in files}
 ledger=dict(old_file=str(OLD),old_sha256=sha(OLD),new_file=str(source),new_sha256=sha(source),default_zero_AST_identical_after_declared_reversal=True,changes=['optional warmup_settle_s=0 and warmup_hold_s=0','extend physical warmup total only when requested','quintic blend of continuing gait target into native base stance and its analytic velocity','preregistration appended stages/handover time'],no_model_or_neural_changes=True)
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 w.dump(pre,dict(stage='Real1s algorithm walking then150ms algorithm quintic settling then50ms algorithm static-target hold then250ms pure neural',walking_s=1.,settle_s=.15,assisted_hold_s=.05,handover_s=1.2,neural_s=.25,body_dt_s=.0000125,neural_dt_s=.0005,command_dt_s=.001,motor='defaulttau5ms100Hz',background_current=0,seed=0,descending_pulse=False,root_assistance=False,post_neural_assistance=False,source_hashes=hashes,source_ledger=ledger,success_gate='Exact first1s gait prefix, no state overwrites, lower measured handover velocity, no neural auxiliary force, actual support outcome including failure'))
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05)
 w.dump(O/'SOURCE_PARAMETERIZATION_LEDGER.json',ledger);w.dump(O/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items()),source_hashes=hashes));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'OLD_RUNNER.py').write_bytes(OLD.read_bytes())
 print('ACTUAL_SETTLE_COMPLETE',O,flush=True)
if __name__=='__main__':main()
