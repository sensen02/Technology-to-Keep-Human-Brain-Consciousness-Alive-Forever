"""Compare five physical continuations from a single verified moving checkpoint."""
import argparse
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from replay import HERE, PAIRED, course, put, read
from snapshot import read_checkpoint, restore, write_checkpoint
from model import SACBrain
from sac_replay import initial_prefix


@torch.inference_mode()
def restored_memory(agent, history):
    _,h = agent.brain.state(torch.from_numpy(history).reshape(1,-1,7))
    return h


def check_immutable_visual(teacher, candidate):
    for name in ("vision","visual_head","range_vision"):
        reference = getattr(teacher,name).state_dict()
        compared = getattr(candidate,name).state_dict()
        assert reference.keys()==compared.keys()
        assert all(torch.equal(reference[k],compared[k]) for k in reference)


@torch.inference_mode()
def action_probabilities(agent, env, obs):
    h = env.brain.memory[0,0]
    features = torch.cat((h,torch.from_numpy(obs[1])))
    logs,mu,_ = agent.distribution(features)
    return logs.exp().tolist(), float((mu.tanh()+1.)*.5)


def rollout(env, checkpoint, prefix, plan, name, teacher, agent, history):
    out = HERE/name
    out.mkdir()
    brain = agent.brain if agent is not None else teacher
    restore(env,checkpoint,brain)
    if agent is not None:
        env.brain.memory = restored_memory(agent,history)
    env.scene["ticks"] = env.scene["deadline_ticks"] = env.n+round(plan["maximum_suffix_s"]/.020)
    start_time = env.time
    start_n = env.n
    initial = env.distance()
    initial_pose = env.pose()[0]
    initial_memory = env.brain.memory.numpy().copy()
    initial_speed = env.body.com_velocity()
    expected = prefix["vision"][start_n]
    np.testing.assert_array_equal(env.observe()[1],expected)
    start_obs = env.observe()
    first = action_probabilities(agent,env,start_obs) if agent is not None else None
    observations,frames = [start_obs], [env.rgb.copy()]
    poses = [env.body.d.qpos.copy()]
    coms = [initial_pose]
    margins = [env.footprint.bounds(poses[-1])[0]-env.box_edge]
    logs = []
    delivered, selected = [], []
    start = monotonic()
    while env.end is None:
        idx = env.n
        if name == "teacher":
            mode,pace = int(prefix["mode"][idx]),float(prefix["pace"][idx])
        elif name == "neutral":
            mode,pace = 0,.5
        else:
            mode,pace = agent.choose(env,observations[-1],deterministic=True)
        reward,physical,com,distance,bearing,contact,actual,gated = env.step(mode,pace)
        logs.append((mode,pace,distance,bearing,contact,reward["reward"],reward["reason"] or "",
                     env.reward.settle_s,float(physical[-1]),float(env.measured["nonfoot"]/course.W)))
        delivered.append(actual)
        selected.append(gated)
        observations.append(env.observe())
        frames.append(env.rgb.copy())
        poses.append(env.body.d.qpos.copy())
        margins.append(env.footprint.bounds(poses[-1])[0]-env.box_edge)
        coms.append(com)
        if name == "teacher":
            np.testing.assert_array_equal(poses[-1],prefix["qpos"][idx+1])
            np.testing.assert_array_equal(env.past,prefix["x"][idx+1])
            np.testing.assert_array_equal(env.rgb,prefix["eyes"][idx+1])
            np.testing.assert_array_equal(actual,prefix["delivered"][idx])
            np.testing.assert_array_equal(gated,prefix["gated"][idx])
            np.testing.assert_array_equal(com,prefix["com"][idx])
            np.testing.assert_allclose(reward["reward"],prefix["reward"][idx],rtol=0.,atol=1.e-12)
        if len(logs)%250==0:
            print("suffix",name,len(logs),round(distance,4),round(monotonic()-start,1),flush=True)
    (mode,pace,distance,bearing,contact,reward,reason,settle,upright,nonfoot) = map(np.asarray,zip(*logs,strict=True))
    delivered, selected = np.asarray(delivered),np.asarray(selected)
    obs,vision = map(np.asarray,zip(*observations,strict=True))
    flat = delivered.reshape(-1,2)[1:]
    forward_s = .005*np.count_nonzero(flat[:,0]>0.)
    tail = np.asarray(margins)
    com = np.asarray(coms)
    r = np.linalg.norm(com[:,:2]-env.target[:2],axis=1)
    early = round(plan["maximum_early_window_s"]/.020)
    net = float(r[0]-r[-1])
    finished = env.end["reason"]=="success"
    clear = bool(tail.min()>=plan["minimum_clearance"])
    assert env.forces.count-checkpoint["forces"]["count"]==len(logs)*1600
    assert env.time-start_time==len(logs)*20 == env.neck.link.step-start_time
    source = env.neck.link.verify_final()
    contact_count = int(env.forces.counts.sum()-np.sum(checkpoint["forces"]["counts"]))
    advanced = bool(forward_s>=plan["minimum_forward_s_to_count_as_continuation"]
                    and net>=plan["minimum_additional_com_progress"])
    out_npz = out/"SUFFIX.npz"
    np.savez_compressed(out_npz,x=obs,vision=vision,qpos=poses,com=com,tail_margin=tail,
                        eyes=frames,mode=mode,pace=pace,requested=np.array([course.action_vector(m,p)
                            for m,p in zip(mode,pace,strict=True)]),
                        distance=distance,bearing=bearing,contact_W=contact,reward=reward,reason=reason,
                        settled_s=settle,upright=upright,nonfoot_W_20ms=nonfoot,delivered=delivered,
                        gated=selected,initial_memory=initial_memory,goal=env.target)
    put(out/"TRANSPORT.json",env.neck.link.journal())
    result = dict(arm=name,tick_at_handoff=start_n,first_abs_ms=start_time,native_steps=len(logs)*1600,
                  duration_s=len(logs)*.020,ending=env.end["reason"],
                  initial_range=initial,final_range=float(distance[-1]),minimum_range=float(distance.min()),
                  initial_com_xy_speed=float(np.linalg.norm(initial_speed[:2])),
                  com_target_net_progress=net,
                  first_2s_com_progress=float(r[0]-r[min(early,len(r)-1)]),
                  autonomous_dn_forward_s=float(forward_s),
                  autonomous_dn_turn_s=.005*np.count_nonzero(flat[:,1]),
                  first_2s_requested_forward_ticks=int(np.count_nonzero(mode[:early]==1)),
                  first_2s_requested_stop_ticks=int(np.count_nonzero(mode[:early]==0)),
                  initial_action_probabilities=first,first_actor_mode=int(mode[0]),
                  first_2s_pace_mean=float(pace[:early][mode[:early]==1].mean()) if (mode[:early]==1).any() else None,
                  min_tail_margin=float(tail.min()),native_box_contact_count=contact_count,
                  final_neutral_settle_s=float(settle[-1]),final_bearing_deg=float(bearing[-1]),
                  resumed_sufficient_forward=advanced,full_body_stays_beyond=clear,
                  reached_and_stood=finished,pass_all=advanced and clear and contact_count==0 and finished,
                  exact_archive_control=name=="teacher",source=source,geometry=env.geometry,
                  episode_sha256=course.sha(out_npz),transport_sha256=course.sha(out/"TRANSPORT.json"),
                  wall_s=monotonic()-start)
    put(out/"RESULT.json",result)
    print("completed suffix",name,result["ending"],round(net,4),forward_s,result["pass_all"],flush=True)
    return result


def run_from_checkpoint(env,prefix,source):
    plan = read(HERE/"PLAN.json")
    path = HERE/"MOVING_CHECKPOINT.pkl"
    digest = write_checkpoint(env,path)
    checkpoint = read_checkpoint(path,digest)
    metadata = dict(snapshot_sha256=digest,physics_clock_ms=env.time,actor_tick=env.n,
                    source_course=plan["scene"],choices=plan["models"],bindings_sha256=env.neck.link.digest,
                    pending_packet_count=len(checkpoint["link"]["_queue"]),initial_contacts=env.body.d.ncon,
                    native_initial_qpos=env.body.d.qpos.tolist(),
                    native_initial_qvel=env.body.d.qvel.tolist(),pending_eye_ms=env.eyes.pending[0],
                    initial_support_W=float(env.measured["loads"].sum()/course.W))
    put(HERE/"CHECKPOINT.json",metadata)
    # Check continuity of the actual named queue up to the branching point.
    reference_journal = read(source/"TRANSPORT.json")
    assert env.neck.link._journal==reference_journal[:len(env.neck.link._journal)]
    metadata["prefix_transport_event_count_exact"] = len(env.neck.link._journal)
    put(HERE/"CHECKPOINT.json",metadata)
    del reference_journal
    warm_prefix=initial_prefix(source/"TRANSPORT.json",prefix["x"][0])
    history=np.concatenate((warm_prefix,prefix["x"][:env.n+1]),axis=0)
    np.save(HERE/"BRAIN_HISTORY.npy",history,allow_pickle=False)
    metadata["history_sha256"] = course.sha(HERE/"BRAIN_HISTORY.npy")
    put(HERE/"CHECKPOINT.json",metadata)
    jobs = {}
    names = ("teacher","neutral",*plan["models"])
    for name in names:
        with (HERE/f"{name}.log").open("xb") as log:
            jobs[name] = subprocess.Popen([sys.executable,"-B",str(HERE/"suffix.py"),name],
                                          stdout=log,stderr=subprocess.STDOUT)
    ends = {}
    try:
        for name,process in jobs.items():
            ends[name] = process.wait(timeout=540)
    finally:
        for process in jobs.values():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    put(HERE/"WORKER_EXIT.json",ends)
    assert not any(ends.values()),ends
    results = [read(HERE/name/"RESULT.json") for name in names]
    assert results[0]["pass_all"] and not results[1]["resumed_sufficient_forward"]
    put(HERE/"RESULTS.json",dict(checkpoint=metadata,tests=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=("teacher","neutral","warm","original","route"))
    args = parser.parse_args()
    name = args.arm
    torch.set_num_threads(1)
    plan = read(HERE/"PLAN.json")
    source = PAIRED/plan["scene"]
    metadata = read(HERE/"CHECKPOINT.json")
    checkpoint = read_checkpoint(HERE/"MOVING_CHECKPOINT.pkl",metadata["snapshot_sha256"])
    assert course.sha(HERE/"BRAIN_HISTORY.npy")==metadata["history_sha256"]
    history = np.load(HERE/"BRAIN_HISTORY.npy",allow_pickle=False)
    with np.load(source/"EPISODE.npz",allow_pickle=False) as z:
        prefix = {key:z[key] for key in ("qpos","com","eyes","x","vision","mode","pace",
                                        "delivered","gated","reward")}
    torch.manual_seed(3373)
    course.SCENES = PAIRED/"SCENES.json"
    teacher = course.new_brain()
    pin = PAIRED/plan["models"][name] if name in plan["models"] else None
    agent = None
    if pin is not None:
        agent = SACBrain().eval()
        agent.load_state_dict(torch.load(pin,map_location="cpu",weights_only=True)["model"])
        check_immutable_visual(teacher,agent.brain)
    env = course.Course(plan["scene"],HERE/("fixture_"+name),teacher)
    try:
        result = rollout(env,checkpoint,prefix,plan,name,teacher,agent,history)
        if pin is not None:
            result["policy"] = dict(path=str(pin),sha256=course.sha(pin))
            put(HERE/name/"RESULT.json",result)
    finally:
        env.close()


if __name__=="__main__":
    main()
