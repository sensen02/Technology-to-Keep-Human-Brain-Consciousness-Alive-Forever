"""Actual trial with declared length-range recentering (engineering, not physiology)."""
from pathlib import Path
import json,hashlib
import run_physical_walk_prewarm as w
P=Path(__file__).parent
O=Path('/home/sensen/Desktop/engine修复验收/s5-lengthrange-neural/20261028T089000Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-lengthrange-recenter/20261028T088000Z/MODEL_recenter_1.0.xml')
def main():
 pre=O.parent/(O.name+'_PREDECLARED.json');pre.parent.mkdir(parents=True,exist_ok=True);assert not pre.exists() and not O.exists()
 sources=[Path(__file__),P/'run_physical_walk_prewarm.py',P/'engine/embodied/continuous_handover.py',BASE]
 pre.write_text(json.dumps(dict(scope='Isolate the settled-pose muscle floor defect. Same body/pipeline as086 but every muscle length-range is translated so normalized length at the settled pose is 1.0. Engineering calibration, NOT physiology; no anatomy or connection change.',base_xml=str(BASE),neural_prewarm_s=.2,neural_only_s=.25,handover_s=1.2,control_reference='086 same protocol, 15ms first nonfoot',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
 w.run(O,1.,100.,50.,.0000125,neural_duration=.25,neural_command_dt_s=.001,warmup_settle_s=.15,warmup_hold_s=.05,neural_prewarm_s=.2,base_xml=BASE)
if __name__=='__main__':main()
