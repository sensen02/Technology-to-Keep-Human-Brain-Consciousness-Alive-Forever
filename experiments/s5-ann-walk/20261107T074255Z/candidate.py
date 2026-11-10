"""Single isolated frequency candidate. Same ANN learner/runtime/body/actuator as 073258Z.
Change FREQ only, preregister independently, no heading trims or gain sweep.
"""
import sys
from pathlib import Path
import json
import hashlib
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T073258Z')
sys.path.insert(0,str(BASE))
import neural_walk as n
n.FREQ=5.5
n.OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T074255Z')

if __name__=='__main__':
    n.OUT.mkdir(parents=True,exist_ok=True)
    ledger=n.OUT/'FREQUENCY_LEDGER.json'
    if not ledger.exists():
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        ledger.write_text(json.dumps(dict(
            base_runtime=str(BASE/'neural_walk.py'),base_sha256=sha(BASE/'neural_walk.py'),
            driver=str(Path(__file__)),driver_sha256=sha(__file__),
            previous_frequency_hz=3.,candidate_frequency_hz=5.5,
            reason='3Hz neural10s run stays up but D5 height oscillation fails; existing teacher diagnostics at5.5Hz had smaller height dip. Test one isolated faster neural limit cycle, no support gate change.',
            actuator='same scale2,30rad/s, original torque bounds and physics',
            assessment='same4s screen then11s if candidate supports/steps; no moving reference window or dropping failed frames',
            brain='replacement ANN arm, no original connectome in this experiment'),indent=2)+'\n')
    n.main()
