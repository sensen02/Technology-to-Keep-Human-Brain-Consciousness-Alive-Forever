"""Use the original warm-release entry point to execute complete 250 ms rollouts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from bench import BASE, HERE, PARENT, HANDOFF, np, sha

ENTRY = PARENT.parent/'run_stance.py'


def run(name, checkpoint, velocity, mode='warm_release'):
    out=HERE/'validation'/name
    subprocess.run([sys.executable,'-B',str(ENTRY),str(out),'--mode',mode,'--parameters',str(checkpoint),
                    '--initial-vx',str(velocity)],check=True)
    with np.load(out/(mode+'.npz')) as a:
        if mode=='warm_release':
            np.testing.assert_array_equal(a['command'][HANDOFF:], np.repeat(a['neural_command'][HANDOFF//5-1:-1],5,axis=0))
        assert not np.any(a['teacher_on'][HANDOFF:])
    return json.loads((out/(mode+'.json')).read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('checkpoint',type=Path)
    p.add_argument('velocity',type=float)
    p.add_argument('--mode',default='warm_release')
    a=p.parse_args()
    run(a.name,a.checkpoint.resolve(),a.velocity,a.mode)
