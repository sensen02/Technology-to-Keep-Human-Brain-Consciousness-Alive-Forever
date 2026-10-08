import sys, time
import mujoco
for tag, P in (("SHIPPED (1 leg, 15 muscles)",
                "venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml"),
               ("MIRRORED (6 legs, 90 muscles)",
                "outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml")):
    t0=time.time()
    try:
        m = mujoco.MjModel.from_xml_path(P)
        n_musc = sum(1 for i in range(m.nu) if m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE)
        n_tend = m.ntendon
        import numpy as np
        print(f"{tag}: COMPILED  nq={m.nq} nu={m.nu} ntendon={n_tend} "
              f"muscle-typed actuators={n_musc}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {str(e)[:160]}  ({time.time()-t0:.1f}s)")
