"""Run a righting candidate from a pinned fallen state through native standing recovery."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from search import ContactLog, HERE, INITIALS, STARTS, W, evaluate, load, model, read, stand


def qualifies(native, ms, handoff):
    native_support = native["foot"][:, :, 2]/W
    n_ms = len(native_support)//80
    shaped = (n_ms,80)
    foot_sum = native_support.sum(1).reshape(shaped).mean(1)
    feet_ok = ((native_support >= .02).sum(1)>=3).reshape(shaped).mean(1)
    posture_ok = ((native["upright"] >= np.cos(np.deg2rad(20.)))
                  & (native["com_z"] >= 1.2)
                  & (native["nonfoot_normal"] < .01*W)).reshape(shaped).all(1)
    speed_2 = (np.diff(ms["com"][:, :2],axis=0)**2).sum(1)/(.001*stand.LENGTH)**2

    def window(start):
        i = slice(start,start+200)
        return bool(posture_ok[i].all() and foot_sum[i].mean()>=.9
                    and feet_ok[i].mean()>=.95 and speed_2[i].mean()<=.01**2)

    first = next((i for i in range(handoff,n_ms-199) if window(i)),None)
    standing = first is not None and window(n_ms-200)
    theta = np.rad2deg(np.arccos(np.clip(native["upright"],-1,1)))
    return dict(passed=standing, handoff_ms=handoff, first_stand_ms=first if standing else None,
                tilt_min_deg=float(theta.min()), terminal_tilt_deg=float(theta[-1]),
                max_com=float(native["com_z"].max()), terminal_com=float(native["com_z"][-1]),
                tail_foot_w=float(foot_sum[-200:].mean()), tail_support_fraction=float(feet_ok[-200:].mean()),
                tail_peak_nonfoot_w=float(native["nonfoot_normal"][-16000:].max()/W),
                tail_speed_rms_body_lengths_s=float(np.sqrt(speed_2[-200:].mean())))


@torch.no_grad()
def run(output, seed, goal_path=None, tail=1000, controller=None, recovery_limit=None):
    output.mkdir()
    bench = stand.Bench(output)
    root = stand.sha(INITIALS/f"{seed}.npz")
    seed_bank = read(INITIALS/"BANK.json")["seeds"]
    load(bench, INITIALS /f"{seed}.npz", seed_bank[seed]["snapshot_sha256"])
    plant = bench.body
    initial = plant.d.qpos.copy()
    body_model = model()
    goals = None
    digest = None
    if goal_path is not None:
        digest = stand.sha(goal_path)
        with np.load(goal_path,allow_pickle=False) as track:
            goals = track["goals"]
        if goals.ndim != 2 or goals.shape[1]!=42 or not np.isfinite(goals).all():
            raise ValueError("Invalid recovered target goals")
    if recovery_limit is not None:
        handoff = recovery_limit
    else:
        handoff = len(goals)*5 if goals is not None else 0
    duration = handoff+tail
    recorder = ContactLog(plant,duration*80)
    plant.recorder = recorder
    time_start = time.monotonic()
    states, x, actions, source = [], [], [], []

    def sample():
        return (plant.d.subtree_com[plant.th].copy(),plant.actuator.target_rad.copy(),
                plant.d.qpos[plant.a_qpos].copy(),float(plant.d.time))

    states.append(sample())
    try:
        for tick in range(duration):
            if tick%5==0:
                sample_x=stand.observe(plant)
                if tick < handoff:
                    if controller is None:
                        rate = np.clip((goals[tick//5]-plant.actuator.target_rad)/.005,-2.,2.)
                        label = "teacher"
                    else:
                        rate = controller(torch.from_numpy(sample_x)).numpy()*2.
                        label = "learned"
                else:
                    rate = body_model(torch.from_numpy(sample_x)).numpy()*2.
                    label = "standing"
                if not np.isfinite(rate).all() or np.max(np.abs(rate))>2.:
                    raise RuntimeError("Policy exceeded authorized joint rate")
                x.append(sample_x)
                actions.append(rate.copy()/2.)
                source.append(label)
            plant.tick_rate(rate)
            if np.any(plant.d.xfrc_applied) or np.any(plant.d.qfrc_applied):
                raise RuntimeError("Unplanned wrench")
            states.append(sample())
    finally:
        plant.recorder=None
    native=recorder.arrays()
    ms={k:np.asarray(v) for k,v in zip(("com","target","joints","time"),zip(*states),strict=True)}
    np.testing.assert_array_equal(native["qpos_ms"][0],initial)
    if not np.isclose(ms["time"][-1]-ms["time"][0],duration*.001,atol=1e-8):
        raise ValueError("Continuous physical time duty")
    trace=output/"TRACE.npz"
    np.savez_compressed(trace,**native,**ms,observation=np.asarray(x),action=np.asarray(actions),
                        source=np.asarray(source),initial_qpos=initial)
    summary=qualifies(native,ms,handoff)
    stand.identity.verify_runtime(bench.manifest,plant)
    if digest is not None and stand.sha(goal_path)!=digest:
        raise RuntimeError("Candidate file changed")
    result=dict(**summary,seed=seed,seed_sha256=root,goal_sha256=digest,
                mode="learned" if controller else "teacher" if goals is not None else "baseline",
                native_steps=duration*80,settle_steps=80000,trace_sha256=stand.sha(trace),
                wall_s=time.monotonic()-time_start)
    stand.put(output/"RESULT.json",result)
    print(result,flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("seed",choices=STARTS)
    a.add_argument("output",type=Path)
    a.add_argument("--goals",type=Path)
    a.add_argument("--tail",type=int,default=1000)
    args=a.parse_args()
    torch.set_num_threads(1)
    run(args.output.resolve(),args.seed,args.goals,args.tail)


if __name__=="__main__":
    main()
