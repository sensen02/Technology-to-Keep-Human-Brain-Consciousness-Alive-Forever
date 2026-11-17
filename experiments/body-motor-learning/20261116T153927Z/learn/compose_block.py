"""Recheck true tactile block contact through the shared response selector."""
import argparse
from pathlib import Path
import sys

import numpy as np

from compose import BRAKE, HERE, STOCK_SPEED, Response, checksum, read, sources, write

BLOCK=HERE.parent/'block'


def run(output):
    _,gap,weight=sources()
    sys.path.insert(0,str(BLOCK))
    import collision

    original=collision.Controller
    old_capture=collision.stan.identity.capture
    known=BLOCK/'PROBE_TACTILE_ON'
    expected=read(known/'RESULT.json')
    if not expected['triggered']:
        raise ValueError('Chosen reference has no actual tactile trigger')
    routes=[]

    class Integrated(original):
        def __init__(self,body,plan,terrain_aware):
            super().__init__(body,plan,terrain_aware)
            self.plan=plan
            self.control=original(body,plan,terrain_aware)
            self.gap_local=gap.Local(body,9600)
            self.gap=weight(self.gap_local,read(BRAKE))
            self.inputs=None

        def action(self,body,t):
            if self.inputs is None:
                teacher=self.control.teacher
                self.control.reference=collision.TactileReference(body,teacher,self.control.reference.order,
                                              self.plan,self.reference.parts,False)
                teacher.reference=self.control.reference
                self.inputs=Response(lambda *_:self.corrected,self.gap,lambda *_:self.nominal,'none')
            self.corrected,command=super().action(body,t)
            self.nominal,_=self.control.action(body,t)
            now=body.measure()
            observed=collision.turn.features(body,now)
            elapsed=int(round(t*1.e3))+self.plan['warm_ms']
            a,route=self.inputs.act(observed,[STOCK_SPEED,0.],np.array([self.reference.sensed]),
                                    self.gap_local.sense(now),now,elapsed)
            routes.append((elapsed,route))
            return a,command

    def pin(core,scope,*,extra_paths):
        return old_capture(core,{**scope,'common_scheme':'same response selector with tactile engineering backend'},
                           extra_paths=[*extra_paths,HERE/'compose.py',Path(__file__),BRAKE])

    collision.Controller=Integrated
    collision.stan.identity.capture=pin
    try:
        collision.run(output,BLOCK/'PROVOCATION.json',True,expected['actual_ms'])
    finally:
        collision.Controller=original
        collision.stan.identity.capture=old_capture
    actual=read(output/'RESULT.json')
    names=('TRACE.npz','DECISIONS.npz','NATIVE.npz')
    exact={name:checksum(output/name)==expected['files'][name] for name in names}
    selected={kind:[ms for ms,role in routes if role==kind] for kind in ('nominal','stumble','gap')}
    result=dict(independent_local_course_native_steps=actual['native_integrations'],source=str(known),
                shared_selector=str(HERE/'compose.py'),closed_loop=True,qualified_learned_stumble=False,
                local_stumble_engineering_backend=True,detected_local_tactile=actual['triggered'],
                active_stumble_decisions=len(selected['stumble']),active_gap_decisions=len(selected['gap']),
                first_stumble_action_ms=selected['stumble'][0] if selected['stumble'] else None,
                exact_reproduction=exact,passed=all(exact.values()) and bool(selected['stumble']) and not selected['gap'])
    write(output/'ASSEMBLY_AUDIT.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.output.resolve())
