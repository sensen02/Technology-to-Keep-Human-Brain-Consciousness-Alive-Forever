import sys, time, os
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
for tag, P in (("MIRRORED (6 legs, 90 muscles)",
                "/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml"),):
    t0=time.time()
    try:
        spec = _load_mjcf(P)
        print(f"{tag}: spec loaded ({time.time()-t0:.1f}s), compiling...", flush=True)
        m = spec.compile()
        nm = sum(1 for i in range(m.nu) if m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE)
        print(f"{tag}: COMPILED  nq={m.nq} nu={m.nu} ntendon={m.ntendon} muscle actuators={nm} "
              f"({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {str(e)[:300]} ({time.time()-t0:.1f}s)")
