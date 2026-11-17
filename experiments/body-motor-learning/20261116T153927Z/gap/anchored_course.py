"""Continue the accepted clear-recovery prefix with the verified foothold teacher."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import time

import numpy as np
import torch

import crossing as q
from safe_crossing import ClearHandoff

BLOCK=q.HERE.parent/"block"
BLOCK_CONTROLLER=BLOCK/"controller.py"
BLOCK_SOURCE=BLOCK/"COURSE_EXTENDED_REPEAT"/"RESULT.json"
BLOCK_PLAN=BLOCK/"EXTENDED.json"
WALK_MS=33_600
END=q.RELEASE+WALK_MS+q.TERMINAL


spec=spec_from_file_location("anchored_block_teacher",BLOCK_CONTROLLER)
block_teacher=module_from_spec(spec)
spec.loader.exec_module(block_teacher)


class SlotReference(block_teacher.CourseReference):
    def _foothold(self, endpoint):
        xy = endpoint[:2]
        margin=self.plan["edge_margin"]
        if np.all(xy>=self.box_min[:2]-margin) and np.all(xy<=self.box_max[:2]+margin):
            xy[0]=self.box_max[0]+margin+.016
        return 0.


def start_teacher(c, shape, inherited):
    x0,x1,y0,y1=shape["opening"]
    plan=dict(inherited, course_ms=WALK_MS, center=[(x0+x1)/2,(y0+y1)/2,-shape["depth"]/2],
              half_size=[(x1-x0)/2,(y1-y0)/2,shape["depth"]/2])
    assert q.read(BLOCK_SOURCE)["native_engineering"]["passed"]
    before=tuple(a.copy() for a in (c.d.qpos,c.d.qvel,c.d.act,c.actuator.target_rad))
    actor=block_teacher.Controller(c,plan,terrain_aware=True)
    old=actor.teacher.reference
    actor.reference=SlotReference(c,actor.teacher,old.order,plan,terrain_aware=True)
    actor.teacher.reference=actor.reference
    for saved,current in zip(before,(c.d.qpos,c.d.qvel,c.d.act,c.actuator.target_rad),strict=True):
        np.testing.assert_array_equal(saved,current)
    return actor


def run(output, duration, native):
    torch.set_num_threads(1)
    output.mkdir()
    begin=time.monotonic()
    initial=q.bench.STARTS/"nominal.npz"
    initial_sha=q.read(q.bench.STARTS/"BANK.json")["nominal"]["sha256"]
    shape=q.read(q.SOURCE)
    c,delta=q.build_scene(shape,initial,initial_sha)
    sensor=q.Local(c,q.RESUMED_WALK)
    branch=ClearHandoff(sensor,q.read(q.PLAN),shape,"far",.036,20.,400,holding=False)
    base=q.physical.frozen_policy()
    record=q.Capture(c,duration) if native else None
    c.recorder=record
    inherited=q.read(BLOCK_PLAN)
    sources=[Path(__file__),q.HERE/"crossing.py",q.HERE/"safe_crossing.py",q.SOURCE,q.PLAN,q.BASIS,
             initial,q.PRIOR/"physical.py",q.PRIOR/"gap_stop.py",q.PRIOR/"gap_weight.py",q.PRIOR/"geometry.py",
             q.physical.BASE/"prototype.py",q.physical.BASE/"walking_bench.py",q.physical.ARCHIVE/"TRACE.npz",
             q.PRIOR/"floor_contacts.cpp",q.physical.NATIVE_LIB,BLOCK_CONTROLLER,BLOCK_SOURCE,BLOCK_PLAN,
             Path(block_teacher.turn.__file__),Path(block_teacher.previous.__file__)]
    identity=q.stan.identity.capture(c,dict(scope="slot recovery then world-anchored forward course",
            initial_snapshot_sha256=initial_sha,initial_observation_max_abs_delta=delta,
            plan=shape,previous_block_source=str(BLOCK_SOURCE.resolve()),renewed_at_ms=q.RELEASE,
            continued_walk_ms=WALK_MS,actuator_forward_request_peak_native_s=inherited["speed"]),
            extra_paths=sources)
    q.stan.put(output/"RUN_IDENTITY.json",identity)
    record_rows=[]
    decisions=[]
    pads=np.asarray([c.pads[leg] for leg in q.bench.LEGS])
    m=c.measure()
    a=np.zeros(42)
    delta_action=a.copy()
    command=np.zeros(2)

    def snapshot():
        points=c.d.geom_xpos[pads].mean(1)
        record_rows.append((float(c.d.time),c.d.qpos.copy(),c.d.qvel.copy(),c.d.subtree_com[c.th].copy(),
            m["loads"].copy(),float(m["nonfoot"]),m["padz"].copy(),m["speed"].copy(),m["foot"].copy(),
            float(c.d.xmat[c.th,8]),c.actuator.target_rad.copy(),c.d.ctrl.copy(),a.copy(),delta_action.copy(),
            points[q.RF].copy(),points.copy(),command.copy()))

    snapshot()
    course=None
    unsafe=0
    for t in range(duration):
        if t%5==0:
            x=q.bench.features(c,m)
            l=sensor.sense(m)
            if t<q.RELEASE:
                intended=int(t>=q.FIRST_WALK)
                command=np.array([.071713*intended,0.])
                b=np.asarray(base(x,intended),dtype=np.float64)
                a=branch(t,x,intended,l,b,m)
            else:
                if course is None:
                    course=start_teacher(c,shape,inherited)
                a,command=course.action(c,(t-q.RELEASE)/1000.)
                intended=int(t<q.RELEASE+WALK_MS)
                b=np.asarray(base(x,intended),dtype=np.float64)
            a=q.bench.legacy.bounded_action(a)
            delta_action=a-b
            decisions.append((t,x.copy(),command.copy(),intended,l.copy(),b.copy(),a.copy()))
        m=c.tick(a)
        if np.any(c.d.qfrc_applied) or np.any(c.d.xfrc_applied):
            raise RuntimeError("External force detected")
        snapshot()
        unsafe=unsafe+1 if m["com"]<.8 or m["nonfoot"]>.5*q.W else 0
        if unsafe>=100:
            break
    arrays=dict(zip(q.TRACE_KEYS,map(np.asarray,zip(*record_rows)),strict=True))
    np.savez_compressed(output/"TRACE.npz",**arrays)
    labels=("ms","x","command","intent","local","base","action")
    np.savez_compressed(output/"DECISIONS.npz",**dict(zip(labels,map(np.asarray,zip(*decisions)),strict=True)))
    last=len(record_rows)-1
    stat=q.bench.standing(arrays,q.RELEASE-260,q.RELEASE) if last>=q.RELEASE else None
    summary=dict(mode="far_anchored_course",actual_ms=last,native_iterations=last*q.PER_MS,
                 onset_ms=branch.onset,edges=branch.edges,stage=branch.stage,
                 initial_stand=q.bench.standing(arrays,0,q.FIRST_WALK),before_renewal_stand=stat,
                 scenario=shape,walk_ms=WALK_MS,tail_native_end=arrays["all_pad_xyz"][-1].tolist(),
                 COM_native_end=arrays["com"][-1].tolist(),min_com=float(arrays["com"][:,2].min()),
                 max_nonfoot_W=float(arrays["nonfoot"].max()/q.W),
                 trace_sha256=q.stan.sha(output/"TRACE.npz"),decisions_sha256=q.stan.sha(output/"DECISIONS.npz"))
    if last==END:
        evaluation=q.bench.metrics(arrays,q.RELEASE,WALK_MS,END)
        evaluation["walk_ms"]=WALK_MS
        summary["post_renewal"]=evaluation
        if record is not None:
            before=q.physical.WARM
            q.physical.WARM=q.RELEASE
            try:
                q.physical.native_metrics(record,output,evaluation,arrays)
            finally:
                q.physical.WARM=before
            summary["native_audit_sha256"]=q.stan.sha(output/"NATIVE_AUDIT.json")
    if record is not None:
        n=record.count
        assert n==last*q.PER_MS and not record.poses[:n,7].any()
        if last!=END:
            np.savez_compressed(output/"NATIVE.npz",forces=record.forces[:n],pose=record.poses[:n],
                                rf_floor=record.rf_contact[:n],rf_pad_bottom_z=record.fk_z[:n])
        np.savez_compressed(output/"GEOMETRY_NATIVE.npz",rf_spatial=record.rf_spatial[:n])
        summary.update(native_sha256=q.stan.sha(output/"NATIVE.npz"),
                       geometry_native_sha256=q.stan.sha(output/"GEOMETRY_NATIVE.npz"),
                       native_contact_cross_checks=record.cross_checks)
    q.stan.identity.verify_runtime(identity,c)
    summary["elapsed_wall_s"]=time.monotonic()-begin
    q.stan.put(output/"RESULT.json",summary)
    print(json.dumps({k:summary[k] for k in ("mode","actual_ms","edges","before_renewal_stand",
                      "COM_native_end","tail_native_end","min_com","max_nonfoot_W","elapsed_wall_s")}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--duration",type=int,default=END)
    p.add_argument("--native",action="store_true")
    a=p.parse_args()
    assert q.RELEASE<a.duration<=END
    run(a.output.resolve(),a.duration,a.native)
