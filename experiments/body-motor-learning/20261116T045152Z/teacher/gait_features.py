"""Compare measured foot duty, advances, and contacts without equating morphologies."""
import argparse
from pathlib import Path

import numpy as np

from retarget import HERE, SOURCE, WALK_START, arc, bench, off_ground


def yaw_axes(quaternions):
    w, x, y, z = quaternions.T
    return np.column_stack((1. - 2. * (y*y + z*z), 2. * (x*y + w*z)))


def summarize(time, legs, local_x, ground_z, supported, root, heading, length):
    assert len(time) == len(local_x) == len(root)
    dt = np.median(np.diff(time))
    assert np.isclose(dt, .001)
    observed_cycles = {}
    for li, leg in enumerate(legs):
        swings = []
        for a, b in off_ground(~supported[:, li]):
            if a == 0 or b == len(time) or b - a < 10:
                continue
            h = arc(ground_z[a-1:b+1, li]).max()/length
            advance = (local_x[b, li] - local_x[a-1, li])/length
            if h < .005 or advance < .01:
                continue
            swings.append(dict(liftoff_s=float(time[a]), touchdown_s=float(time[b]),
                duration_s=float(time[b] - time[a]), forward_advance_lengths=float(advance),
                clearance_above_chord_lengths=float(h)))
        observed_cycles[leg] = swings
    support = dict(zip(legs, supported.mean(0).tolist()))
    reference = np.array([r["liftoff_s"] for r in observed_cycles["RM"]])
    limbs = {}
    for leg, swings in observed_cycles.items():
        times = np.array([r["liftoff_s"] for r in swings])
        periods = np.diff(times)
        valid = times[(times >= reference[0]) & (times < reference[-1])] if len(reference) > 1 else np.empty(0)
        phase = []
        for t in valid:
            i = np.searchsorted(reference, t, side="right") - 1
            phase.append(float((t-reference[i]) / (reference[i+1]-reference[i])))
        if phase:
            mean = np.exp(2j*np.pi*np.asarray(phase)).mean()
            phase_stats = dict(offset_vs_RM_cycle=float(np.angle(mean) / (2*np.pi) % 1.),
                               circular_resultant=float(abs(mean)), matched_samples=len(phase))
        else:
            phase_stats = dict(offset_vs_RM_cycle=None, circular_resultant=None, matched_samples=0)
        median = lambda key: float(np.median([s[key] for s in swings])) if swings else None
        limbs[leg] = dict(
            ground_supported_fraction=support[leg], qualified_swings=len(swings), cycle_frequency_hz=float(1./np.median(periods)) if len(periods) else None,
            swing_duration_s_median=median("duration_s"),
            step_advance_body_lengths_median=median("forward_advance_lengths"),
            swing_clearance_body_lengths_median=median("clearance_above_chord_lengths"),
            **phase_stats, swings=swings,
        )
    axes = heading[:-1] / np.linalg.norm(heading[:-1], axis=1, keepdims=True)
    displacement = np.einsum("ij,ij->", np.diff(root, axis=0), axes)
    return dict(analyzed_window_s=[float(time[0]), float(time[-1])],
                speed_local_forward_body_lengths_s=float(displacement/(time[-1]-time[0])/length),
                stages_only="walking; 1s startup discarded from each comparison",
                limb_features=limbs)


def measure(directory, is_fly, project_settle_ms=1000):
    result = bench.read(directory / "RESULT.json")
    path = directory / "TRACE.npz"
    assert bench.stand.sha(path) == result["trace_sha256"]
    with np.load(path, allow_pickle=False) as z:
        if is_fly:
            t = z["time_s"]
            use = t >= 1.
            data = summarize(t[use], z["legs"].tolist(), z["foot_body_mm"][use, :, 0],
                z["foot_world_mm"][use, :, 2], z["contact_count"][use] > 0,
                z["thorax_world_mm"][use, :2], z["thorax_rotation"][use, :2, 0], result["body_length_mm"])
            return dict(**data, trace=str(path), trace_sha256=result["trace_sha256"],
                        foot_point="Tarsus5 body origin", support="FlyGym leg ground contact sensor count > 0",
                        length=result["body_length_mm"], length_units="mm", length_definition=result["length_definition"],
                        body_motion="thorax body center", symmetric_signal=result["descending_signal"][0])
        t = z["time"] - z["time"][0]
        first = WALK_START + project_settle_ms
        end = WALK_START + result["walk_ms"] + 1
        use = slice(first, end)
        qpos = z["qpos"][use]
        data = summarize(t[use], bench.LEGS, z["foot"][use, :, 0], z["padz"][use],
            z["loads"][use] >= .05*bench.W, qpos[:, :2], yaw_axes(qpos[:, 3:7]), bench.stand.LENGTH)
        data["stages_only"] = f"walking after discarding {project_settle_ms} ms of the project's walk command"
        return dict(**data, trace=str(path), trace_sha256=result["trace_sha256"], foot_point="mean Tarsus5 pad centers in thorax frame, minimum pad bottom for clearance",
                    support="foot normal load >= 0.05 W", length=bench.stand.LENGTH, length_units="native length",
                    length_definition="existing project stand.LENGTH", body_motion="free-root translation")


def run(baseline, teacher, output, project_settle_ms):
    project = {
        "prototype": measure(baseline, False, project_settle_ms),
        "same_body_teacher": measure(teacher, False, project_settle_ms),
    }
    fly_source = {"shape_source": measure(SOURCE, True)}
    for d in HERE.glob("FLYGYM_SPEED_*"):
        if (d / "RESULT.json").is_file():
            trial = measure(d, True)
            fly_source[f"signal_{trial['symmetric_signal']}"] = trial
    speed = project["prototype"]["speed_local_forward_body_lengths_s"]
    distances = {key: abs(d["speed_local_forward_body_lengths_s"]-speed) for key,d in fly_source.items()}
    nearest = min(distances, key=distances.get)
    limit = .25*speed
    comparable = [key for key in fly_source if distances[key] <= limit and
                  all(v["qualified_swings"] >= 2 for v in fly_source[key]["limb_features"].values())]
    ratio = fly_source["shape_source"]["speed_local_forward_body_lengths_s"]/speed
    bench.stand.put(output, dict(
        source_and_sweep=fly_source, project=project,
        closest_speed_trial=nearest, closest_speed_difference=distances[nearest],
        matching_speed_tolerance_body_lengths_s=limit, comparable_measured_six_leg_trials=comparable,
        source_to_prototype_dimensionless_speed_ratio=ratio,
        comparison="observable clear swings: >=10ms off support, >=0.005 axial lengths above endpoint chord, and >=0.01 axial lengths sagittal advance in thorax frame; count and duty use each platform's physical support signal",
        caveats=["Different operational axial lengths and foot points",
                 "The comparison threshold does not alter original project acceptance",
                 "Shape only was transferred; stride frequency, relative gait timing and support fraction remain project properties"],
    ))
    print(nearest, distances[nearest], comparable, ratio, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("teacher", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--project-settle-ms", type=int, default=1000)
    a = p.parse_args()
    run(a.baseline.resolve(), a.teacher.resolve(), a.output.resolve(), a.project_settle_ms)
