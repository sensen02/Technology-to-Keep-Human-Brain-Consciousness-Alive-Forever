"""Apply identical initial tension and the identical bounded feedback law to both floors."""
import json

import numpy as np
from solve import HERE, BodyObservations, body, load_pose, mj
from teacher import MuscleTeacher
from warm_release import support, SETTLE


def compare():
    pose=json.loads((HERE/'stance.json').read_text())
    q=np.array(pose['qpos'])
    activation=np.array(pose['tonic_activation'])
    controller=MuscleTeacher(body(1.),q,activation)
    reports={}
    for contact_ms in (20.,1.):
        m=body(contact_ms)
        d=load_pose(m,q)
        obs=BodyObservations(m)
        d.act[:]=activation
        d.ctrl[:]=activation
        mj.mj_forward(m,d)
        h=d.subtree_com[0,2]
        good=[]
        first_nonfoot=None
        deepest=0.
        for tick in range(SETTLE+1):
            z=obs.measure(d)
            depth=min((c.dist for c in d.contact if obs.floor in c.geom),default=0.)
            good.append(support(z,depth,h,True))
            deepest=min(deepest,depth)
            if first_nonfoot is None and z['ground_load_w'][-1]>0:
                first_nonfoot=tick*.1
            if tick==0:
                initial=float(z['ground_load_w'][:6].sum())
            if tick==SETTLE:
                break
            if tick%5==0:
                d.ctrl[:]=controller.command(d)
            mj.mj_step(m,d)
            mj.mj_forward(m,d)
        assert not any(d.warning.number) and not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
        reports[str(contact_ms)]=dict(initial_foot_weight=initial,eligible_for_warmup=bool(all(good[-1001:])),
                    first_nonfoot_ms=first_nonfoot, deep_penetration_mm=-float(deepest),
                    final_foot_weight=float(z['ground_load_w'][:6].sum()))
    assert reports['1.0']['eligible_for_warmup'] and not reports['20.0']['eligible_for_warmup']
    (HERE/'FLOORS.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(reports)


if __name__=='__main__':
    compare()
