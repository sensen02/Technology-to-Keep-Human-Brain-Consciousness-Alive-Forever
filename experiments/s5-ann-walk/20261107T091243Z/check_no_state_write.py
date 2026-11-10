"""Zero-integration controller mechanism check; not a walking test."""
import numpy as np
import mujoco as mj
from anchor_teacher import AnchorTeacher,n,KNOBS
from proprioception import features

if __name__=='__main__':
    p=n.Plant();d=p.fresh();ref=n.Reference(p);teacher=AnchorTeacher(p,ref)
    # Scratch synthetic observation at the declared warmup terminal time; no
    # actual physics is integrated and no gait success follows from this test.
    d.time=1.
    before=p.snapshot(d).copy()
    teacher.capture(d,d.time)
    desired,meta=teacher.desired(d,d.time)
    assert np.array_equal(before,p.snapshot(d))
    assert desired.shape==(42,) and np.isfinite(desired).all()
    assert np.all(desired>=ref.lo-1e-12) and np.all(desired<=ref.hi+1e-12)
    old=teacher.anchor_pos.copy()
    teacher.desired(d,d.time)
    assert np.array_equal(old,teacher.anchor_pos)
    assert np.array_equal(before,p.snapshot(d))
    x=features(p,d,d.qpos[p.qa],np.zeros(6))
    assert x.shape==(183,)
    print(dict(physics_steps=0,controller_calls=2,state_unchanged=True,bounds_pass=True,
               feature_dimensions=183,scratch_is_distinct=teacher.scratch is not d,
               root_velocity_reference=KNOBS['desired_speed'],not_a_walking_test=True))
