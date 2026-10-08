import sys, os
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco, numpy as np
from flygym.compose.fly.musculoskeletal import _load_mjcf
SRC="venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml"
DST="outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml"
def stats(p):
    m = _load_mjcf(p).compile()
    nm = sum(1 for i in range(m.nu) if m.actuator_dyntype[i]==mujoco.mjtDyn.mjDYN_MUSCLE)
    return m, dict(nq=m.nq, nv=m.nv, nu=m.nu, ntendon=m.ntendon, nbody=m.nbody, ngeom=m.ngeom,
                   njnt=m.njnt, muscle_actuators=nm,
                   total_mass=float(np.sum(m.body_mass)),
                   mass_kg=float(np.sum(m.body_mass))*9810.0)
a,sa = stats(SRC); b,sb = stats(DST)
print(f"{'quantity':<20}{'shipped (1 leg)':>18}{'mirrored (6 legs)':>20}{'delta':>10}")
for k in sa:
    d = sb[k]-sa[k] if isinstance(sa[k],(int,float)) else '-'
    ds = f"{d:+g}" if isinstance(d,float) or isinstance(d,int) else ''
    print(f"{k:<20}{sa[k]:>18.6g}{sb[k]:>20.6g}{ds:>10}")
# do the original muscle actuators survive unchanged?
na=[mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(a.nu)]
nb=[mujoco.mj_id2name(b, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(b.nu)]
missing=[x for x in na if x not in nb]
print(f"\noriginal actuators preserved: {len(na)-len(missing)}/{len(na)}  missing: {missing}")
import collections
print("actuator counts by leg prefix (mirrored):", dict(collections.Counter(
    n[:2] for n in nb if n and n[0] in 'LR')))
