"""Short matched physical steering probes using source-identified descending delivery."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
VISUAL = HERE.parent / "20261116T023521Z"
BLOCK = EXP / "body-motor-learning/20261116T153927Z/block"
ARCHIVE = EXP / "neck-real-structure/20261112T225331Z/dnp09_interface"
sys.path.insert(0, str(BLOCK))
from rollout import INITIAL, LEGS, Record, W, build_scene, native_metrics, stan, turn
from controller import CourseReference, previous
sys.path.insert(0, str(ARCHIVE))
from source_transport import SourceTransport

DNG = HERE / "DNG13_SOURCE.json"
DNA = VISUAL / "OFFICIAL_GRAPH.json"
BINDING = HERE / "STEERING_BINDING.json"
WARM = 500
WALK = 14400
STOP = 1200
STIM_FROM = 2400
STIM_TO = 12000
PLAN = dict(warm_ms=WARM, course_ms=WALK, stop_ms=STOP, speed=.113, brake_ms=800,
            center=[0.,0.,0.],half_size=[0.,0.,0.],gate_forward_native_s=.05,gate_raw_slip=.5)
CHOICES = {"off": [0.,0.,0.,0.], "dna": [1.,0.,0.,0.], "dng": [0.,0.,1.,0.], "both":[1.,0.,1.,0.]}


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return dict(path=str(path.resolve()),sha256=stan.sha(path))


def binding_and_projection():
    visual = read(VISUAL / "DNa02_BINDING.json")
    dna,dng = read(DNA),read(DNG)
    new_ports = [
        dict(root_id=rid, direction="DN", side=meta["side"], source_reviewed=True,
             structural_evidence_status="SOURCE_GEOMETRY_REVIEWED",source_cell_type="DNg13",
             evidence=meta["source_evidence"])
        for rid,meta in dng["ports"].items()]
    old_dn = [port for port in visual["ports"] if port["direction"]=="DN"]
    old_an = [port for port in visual["ports"] if port["direction"]=="AN"]
    binding = dict(schema=visual["schema"],scope=visual["scope"],ports=old_dn+new_ports+old_an,
                   transport_config=visual["transport_config"],source_files=visual["source_files"]+[pin(DNA),pin(DNG)])
    roots = [row["root_id"] for row in old_dn+new_ports]
    all_edges = {rid: {} for rid in roots}
    for edge in dna["connections"]:
        all_edges[edge["pre"]][edge["post"]] = edge["synapses"]
    for rid,port in dng["ports"].items():
        all_edges[rid] = {edge["post"]:edge["synapses"] for edge in port["vnc_connections"]}
    # The documented engineering projection uses maxima per target. Read each
    # port only from targets exclusive to it so the named perturbations are separable.
    targets = sorted(set().union(*(set(edge) for edge in all_edges.values())))
    weights = np.array([[all_edges[rid].get(post,0) for post in targets] for rid in roots],dtype=float)
    exclusive = (weights > 0).sum(axis=0) == 1
    reads = [np.flatnonzero(exclusive & (weights[index]>0)) for index in range(len(roots))]
    assert min(map(len,reads))>0
    return binding,weights/weights.max(axis=0),reads


class StrideReference(CourseReference):
    """Source-scoped engineering target response without a yaw trajectory."""
    delivered = None

    def _foothold(self, endpoint):
        side = LEGS[self.order[self.slot % 6]][0]
        dna_left,dna_right,dng_left,dng_right = self.delivered
        if side=="L":
            change = .08*(-dna_left+dng_right)
        else:
            change = .08*(-dna_right+dng_left)
        endpoint[:2] += self.heading * change
        return 0.


def run(choice, output):
    torch.set_num_threads(1)
    binding,projection,reads = binding_and_projection()
    assert read(BINDING) == binding
    transport = SourceTransport(BINDING,stan.sha(BINDING))
    output.mkdir()
    body,_,start_sha = build_scene(None)
    total = WARM+WALK+STOP
    data = Record(body,total)
    body.recorder = data
    local = turn.straight_policy()
    measure = body.measure()
    samples = [turn.sample(body,measure)]
    decisions = []
    latest = np.zeros(4)
    teacher = None
    reference = None
    action = np.zeros(42)
    unsafe = 0
    sources = [Path(__file__),INITIAL,BINDING,DNA,DNG,BLOCK/"rollout.py",BLOCK/"controller.py",turn.TEACHER]
    identity = stan.identity.capture(body,dict(scope="flat engineered stride asymmetry probe",
                                    control=choice,start_sha256=start_sha,plan=PLAN),extra_paths=sources)
    stan.put(output/"RUN_IDENTITY.json",identity)
    for step in range(total):
        assert step == transport.step
        if step%5==0:
            relative = step-WARM
            values = CHOICES[choice] if STIM_FROM<=relative<STIM_TO else CHOICES["off"]
            transport.send(dict(direction="DN",root_ids=transport.ports["DN"],values=values,
                                capture_step=step,source_bindings_sha256=transport.digest,**{"from":"brain","to":"neck"}))
            if step < WARM:
                action = local(turn.features(body,measure),0)
            else:
                if teacher is None:
                    teacher = turn.make_teacher(body,"arc_left",PLAN["speed"],.115)
                    reference = StrideReference(body,teacher,teacher.reference.order,PLAN,True)
                    teacher.reference = reference
                reference.delivered = latest
                action,_ = teacher.action(body,relative*.001)
            action = np.clip(action,-1.,1.)
            decisions.append((step,*latest,*action))
        measure = body.tick(action)
        if any(body.d.qfrc_applied) or any(body.d.xfrc_applied):
            raise RuntimeError("Unexpected external force")
        samples.append(turn.sample(body,measure))
        for packet in transport.advance():
            post = np.asarray(packet["values"])@projection
            latest = np.array([post[index].mean() for index in reads])
        unsafe = unsafe+1 if measure["com"]<.8 or measure["nonfoot"]>.5*W else 0
        if unsafe>=100:
            break
    trace = {key:np.array([row[key] for row in samples]) for key in samples[0]}
    count = len(samples)-1
    assert data.count == 80*count
    np.testing.assert_array_equal(data.endpoints[:count+1],trace["qpos"])
    joint = native_metrics(trace,data,PLAN,count) if count==total else None
    cycles = turn.PivotCycles(W)
    previous_event = np.full(6,WARM+STIM_FROM,dtype=int)
    events = []
    for i in range(WARM+STIM_FROM-10,min(WARM+STIM_TO,count)+1):
        last = len(cycles.events)
        cycles.update(trace["time"][i],trace["loads"][i],trace["padz"][i])
        for event in cycles.events[last:]:
            leg = event["leg_index"]
            prior = previous_event[leg]
            if i >= WARM+STIM_FROM and any(e["leg"]==LEGS[leg] for e in events):
                delta = trace["foot"][i,leg,0]-trace["foot"][prior,leg,0]
            else:
                delta = None
            events.append(dict(ms=i,leg=LEGS[leg],relative_foot_x_at_reload=float(trace["foot"][i,leg,0]),
                               stride_reload_to_reload_relative_x=None if delta is None else float(delta),
                               peak_clearance=event["peak_clearance"]))
            previous_event[leg] = i
    yaw = np.unwrap(trace["yaw"])
    window_yaw = float(np.rad2deg(yaw[min(count,WARM+STIM_TO)]-yaw[WARM+STIM_FROM])) if count>WARM+STIM_FROM else None
    np.savez_compressed(output/"TRACE.npz",**trace)
    np.savez_compressed(output/"NATIVE.npz",force=data.forces[:data.count],pose=data.poses[:data.count])
    np.save(output/"DECISIONS.npy",np.asarray(decisions))
    ledger = transport.journal()
    stan.put(output/"TRANSPORT.json",ledger)
    nonzero_release = [i["release_step"] for i in ledger if i["event"]=="release" and any(i["values"])]
    stan.identity.verify_runtime(identity,body)
    result = dict(mode=choice,actual_ms=count,native_integrations=data.count,fit_updates=0,
                  engineering_joint=joint,yaw_steering_window_deg=window_yaw,
                  measured_reload_events=events,source_exclusive_targets=list(map(len,reads)),
                  first_nonzero_release_ms=min(nonzero_release) if nonzero_release else None,
                  latest_verified=transport.verify_final(),
                  trace_files={f:stan.sha(output/f) for f in ("TRACE.npz","NATIVE.npz","DECISIONS.npy","TRANSPORT.json")})
    stan.put(output/"RESULT.json",result)
    print(choice,count,result["yaw_steering_window_deg"],joint and joint["passed"])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("choice",choices=("prepare",*CHOICES))
    parser.add_argument("output",type=Path,nargs="?")
    args = parser.parse_args()
    if args.choice=="prepare":
        plan,_,_ = binding_and_projection()
        stan.put(BINDING,plan)
    else:
        run(args.choice,args.output.resolve())
