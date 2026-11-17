"""Record prolonged nonfoot support in tilted bodies and point to real fallen states."""
import numpy as np

from native_contact import DT, STEPS_PER_MS, W, spans


def supplement(native, old, ms):
    foot_w = native["foot"][:, :, 2] / W
    few_legs = (foot_w >= .02).sum(1) < 3
    tilted = native["upright"] <= np.cos(np.deg2rad(50.))
    ground = native["nonfoot_normal"] >= .10 * W
    candidate = [(a, b) for a,b in spans(few_legs & tilted & ground) if b-a >= 20*STEPS_PER_MS]
    full = np.zeros(len(few_legs), dtype=bool)
    for start, stop in [*old["fall_intervals_steps"], *candidate]:
        full[start:stop] = True
    all_falls = spans(full)
    seed = None
    state_ms = len(native["qvel_ms"]) - 1
    if not old["recovered"] and all_falls and all_falls[-1][1] == len(full):
        body = native["qvel_ms"][state_ms, :6]
        seed = dict(frame_ms=state_ms, absolute_time_s=float(ms["time_s"][state_ms]),
                    tilt_deg=float(np.rad2deg(np.arccos(np.clip(native["upright"][-1], -1., 1.)))),
                    root_speed_native_s=float(np.linalg.norm(body[:3])),
                    root_angular_speed_rad_s=float(np.linalg.norm(body[3:])),
                    tail_contact_ms=float((all_falls[-1][1] - all_falls[-1][0])*DT*1000))
    label = "fall_recovered" if old["recovered"] and all_falls else old["behavior"]
    return {**old, "behavior":label, "fall_intervals_steps":all_falls,
            "thorax_abdomen_fall_steps":old["fall_intervals_steps"],
            "tilt_and_nonfoot_support_steps":candidate,
            "terminal_fallen_pose":seed}
