"""Exercise frozen navigation controllers on preregistered MuJoCo courses."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent / "20261116T232400Z-joint-brain"
SEPARATED = HERE.parent / "20261116T225900Z-separated-afferents"
TOUCH = HERE.parent / "20261116T180500Z-touch"
APPROACH = HERE.parent / "20261116T184600Z-approach"
FAR = HERE.parent / "20261116T191700Z-far-obstacle"
REJOIN = HERE.parent / "20261116T203100Z-rejoin"
sys.path[:0] = list(map(str, (JOINT, SEPARATED, HERE.parent / "20261116T211600Z-touch-prediction", TOUCH, APPROACH, FAR, REJOIN)))
import approach
from approach_runtime import ApproachLink, Motion, Eyes, stand_window, turn, sha
from far_fixture import BODY, LIB, head_tip
from approach_vision import CHECKPOINT as APPROACH_WEIGHTS, COLOR, WIDTH
from run_brain import BINDING as AN_BINDING, FRONT, SCALE, SeparatedLink, SharedBrain, START, W, load_body_policy
from wider import WiderDetour
from visual_rejoin import VisualRejoin

SCENES = HERE / "SCENES.json"
JOINT_WEIGHTS = JOINT / "MODEL/joint.pt"
START_MS = 500
GATES = json.loads(SCENES.read_text())["common_gates"]


class Footprint:
    def __init__(self, body):
        self.model = body.plant
        self.data = mj.MjData(self.model)
        self.pads = np.asarray([body.pads[leg] for leg in BODY.LEG_ORDER])
        self.meshes = []
        for i in np.flatnonzero(self.model.geom_type == mj.mjtGeom.mjGEOM_MESH):
            mesh = self.model.geom_dataid[i]
            start = self.model.mesh_vertadr[mesh]
            self.meshes.append((i, self.model.mesh_vert[start:start+self.model.mesh_vertnum[mesh]]))
        assert len(self.meshes) == 71 and self.pads.size == 18

    def bounds(self, qpos):
        self.data.qpos[:] = qpos
        mj.mj_kinematics(self.model, self.data)
        px = self.data.geom_xpos[self.pads, 0]
        r = self.model.geom_size[self.pads, 0]
        mesh_x = [v @ self.data.geom_xmat[i, :3] + self.data.geom_xpos[i, 0] for i,v in self.meshes]
        return (min(float((px-r).min()), *(float(x.min()) for x in mesh_x)),
                max(float((px+r).max()), *(float(x.max()) for x in mesh_x)))


def prepare(out, scene):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    body, _ = approach.prepare(out, "nominal")
    reference = body.plant
    front = Footprint(body).bounds(body.d.qpos)[1]
    tip = head_tip(body)
    y = body.d.xpos[body.plant.body("Head").id, 1]
    boxes = [[float(tip[0]+dx),float(y+dy),.50,.045,w,.50] for dx,dy,w in scene["boxes"]]
    pit = scene["pit"]
    pit_info = None
    if not boxes and pit is None:
        return body, dict(boxes=boxes,pit=None,front=front)
    spec = _load_mjcf(BODY.PAD_XML)
    for n, (x,y,z,hx,hy,hz) in enumerate(boxes):
        spec.worldbody.add_geom(name=f"course_box_{n}",type=mj.mjtGeom.mjGEOM_BOX,
                                pos=[x,y,z],size=[hx,hy,hz],rgba=[.03,.9,.08,1.])
    if pit:
        near = front + pit["ahead_of_front"]
        far = near + pit["width"]
        extent = 30.
        plane_id, = np.flatnonzero(body.planes)
        plane = spec.geom(reference.geom(plane_id).name)
        z = float(reference.geom_pos[plane_id,2])
        plane.pos[2] = z-pit["depth"]
        for name, lo, hi in (("course_land_near",-extent,near),("course_land_far",far,extent)):
            spec.worldbody.add_geom(name=name,type=mj.mjtGeom.mjGEOM_BOX,
                                    pos=[(lo+hi)/2,0.,z-.2],size=[(hi-lo)/2,extent,.2],
                                    rgba=[.52,.52,.52,1.], contype=int(reference.geom_contype[plane_id]),
                                    conaffinity=int(reference.geom_conaffinity[plane_id]),
                                    friction=reference.geom_friction[plane_id])
        pit_info = dict(near=near,far=far,depth=pit["depth"],top_z=z,
                        floor=reference.geom(plane_id).name)
    model = spec.compile()
    for key in ("jnt_qposadr","jnt_dofadr","jnt_range","actuator_trnid","actuator_gear","dof_damping"):
        np.testing.assert_array_equal(getattr(model,key),getattr(reference,key))
    for i in range(reference.ngeom):
        model.geom(reference.geom(i).name).rgba[:] = reference.geom(i).rgba
    model.mat_rgba[:reference.nmat] = reference.mat_rgba
    loader = BODY.act_mod.load_full_model
    (out/"terrain_identity").mkdir()
    BODY.act_mod.load_full_model = lambda *_a, **_kw: model
    try:
        bench = turn.stand.Bench(out/"terrain_identity")
    finally:
        BODY.act_mod.load_full_model = loader
    changed = bench.body
    if pit:
        changed.planes[[model.geom(k).id for k in ("course_land_near","course_land_far")]] = True
    changed.physics_order = BODY.rc.SENS_IDX
    turn.load_initial(bench, START, sha(START))
    np.testing.assert_array_equal(body.d.qpos,changed.d.qpos)
    np.testing.assert_array_equal(body.d.qvel,changed.d.qvel)
    np.testing.assert_array_equal(body.d.ctrl,changed.d.ctrl)
    assert not any(model.geom(int(i)).name.startswith("course_box") for c in changed.d.contact for i in c.geom)
    delta = turn.stand.observe(changed)-turn.stand.observe(body)
    return changed, dict(boxes=boxes,pit=pit_info,front=front,initial_observation_delta=float(abs(delta).max()))


class Forces:
    """Native contact force by physical obstacle and pit floor, plus lateral afferents."""
    def __init__(self, body, geometry):
        library = ctypes.CDLL(str(LIB))
        self.inspect = library.inspect_contact
        self.inspect.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
        self.inspect.restype = None
        self.ids = [body.plant.geom(f"course_box_{n}").id for n in range(len(geometry["boxes"]))]
        pit = geometry["pit"]
        self.nbox = len(self.ids)
        self.has_pit = bool(pit)
        if pit:
            self.ids.append(body.plant.geom(pit["floor"]).id)
        self.env = np.ascontiguousarray(body.planes, dtype=np.int32)
        self.env[self.ids] = 1
        names = [body.plant.geom(i).name for i in range(body.plant.ngeom)]
        self.leg = np.asarray([0 if n.startswith("LFTarsus") else 1 if n.startswith("RFTarsus") else -1
                               for n in names], dtype=np.int32)
        self.buffer = np.empty(6)
        self.sides = np.zeros(2)
        self.peak = np.zeros(len(self.ids))
        self.counts = np.zeros(len(self.ids),dtype=np.int64)
        self.first_hit = np.full(len(self.ids),-1,dtype=np.int64)
        self.frame = np.zeros(len(self.ids))
        self.count = 0

    def before(self, body):
        return None

    def solved(self, body, _):
        first_probe = self.ids[0] if self.ids else -1
        self.check(body,first_probe)
        self.sides += self.buffer[:2]
        for i, probe in enumerate(self.ids):
            if i:
                self.check(body,probe)
            force = float(self.buffer[2:5].sum())
            if force>0.:
                self.frame[i] = max(self.frame[i],force)
                self.counts[i] += 1
                if self.first_hit[i] < 0:
                    self.first_hit[i] = self.count
        self.count += 1

    def check(self, body, probe):
        self.inspect(body.plant._address,body.d._address,self.leg.ctypes.data,self.env.ctypes.data,
                     probe,self.buffer.ctypes.data)

    def after(self, body, before, solved):
        pass

    def reset_ms(self):
        self.sides.fill(0.)
        self.frame.fill(0.)


def score(trace, feet, goal, geometry, scene, policy, decisions, rec, brain):
    head = trace["head"]
    distance = np.linalg.norm(head[:,:2]-goal[:2],axis=1)
    bearing = np.degrees(np.arctan2(goal[1]-head[:,1],goal[0]-head[:,0])-trace["yaw"])
    bearing = (bearing+180.)%360.-180.
    initial = float(distance[START_MS])
    final = float(distance[-1])
    com = trace["com"]
    end_stand = stand_window(trace,len(com)-1)
    start_stand = stand_window(trace,START_MS)
    min_com = float(com[:,2].min())
    max_nonfoot = float(trace["nonfoot"].max()/W)
    peaks = rec.peak/W
    moved = (decisions[:,3]!=0) | (decisions[:,4]!=0)
    checks = dict(initial_stable=start_stand["passed"], no_nonfoot_ground=max_nonfoot<=GATES["nonfoot_force_max_W"],
                  no_low_com=min_com>=GATES["min_com_z"])
    nav = dict(approaches=initial-final>=scene["min_progress"], aligned=abs(bearing[-1])<=GATES["heading_error_max_deg"],
               enters_goal_zone=scene["target_range"][0]<final<scene["target_range"][1],
               stable_at_goal=end_stand["passed"], autonomously_stops=not bool(moved[-40:].any()))
    if policy!="neck_probe":
        checks.update(nav)
    margins = [(float(com[-1,0]-x-hx), float(feet[-1,0]-x-hx)) for x,_,_,hx,_,_ in geometry["boxes"]]
    if rec.nbox:
        checks["no_box_contact"] = bool(np.all(peaks[:rec.nbox]<=GATES["obstacle_force_max_W"]))
        checks["whole_body_passes_boxes"] = all(full>=GATES["all_pads_far_margin"] for _,full in margins)
    pit = geometry["pit"]
    pit_stats = None
    if pit:
        far = pit["far"]
        first_whole = np.flatnonzero(feet[:,0]>far+GATES["all_pads_far_margin"])
        above = feet[-1,0]>far+GATES["all_pads_far_margin"]
        checks["whole_body_crosses_trench"] = bool(above)
        checks["no_trench_bottom_contact"] = bool(peaks[-1]==0.)
        pit_stats = dict(final_full_body_margin=float(feet[-1,0]-far), first_whole_ms=int(first_whole[0]*20) if len(first_whole) else None,
                         final_com_margin=float(com[-1,0]-far), bottom_peak_W=float(peaks[-1]),
                         front_furthest_x=float(feet[:,1].max()), near=float(pit["near"]), far=float(far))
    checks = {k:bool(v) for k,v in checks.items()}
    movement_ms = decisions[moved,0]
    return dict(passed=all(checks.values()),checks=checks,navigation_measurements={k:bool(v) for k,v in nav.items()},
                native_steps=rec.count,initial_range=initial, final_range=final, minimum_range=float(distance.min()),
                initial_bearing_deg=float(bearing[START_MS]), final_bearing_deg=float(bearing[-1]),
                signed_yaw_deg=float(np.degrees(np.unwrap(trace["yaw"])[-1]-trace["yaw"][START_MS])),
                initial_stand=start_stand, final_stand=end_stand,min_com=min_com, max_nonfoot_W=max_nonfoot,
                first_motion_ms=int(movement_ms[0]) if len(movement_ms) else None,
                last_motion_ms=int(movement_ms[-1]) if len(movement_ms) else None,
                lateral_max=float(np.max(np.abs(com[:,1]-com[0,1]))),box_margins=margins,pit_stats=pit_stats,
                contact_peak_W=peaks.tolist(),contact_native_counts=rec.counts.tolist(),first_contact_step=rec.first_hit.tolist(),
                detour_finished=getattr(brain,"finished_detour",None))


def run(scene_name, policy, out, preview):
    torch.set_num_threads(1)
    config = json.loads(SCENES.read_text())
    scene = config["scenes"][scene_name]
    assert policy in scene["policies"]
    body, geometry = prepare(out,scene)
    rec = Forces(body,geometry)
    body.recorder = rec
    eyes = Eyes(body)
    target = eyes.place(scene["target_angle"],scene["target_distance"])
    background = (eyes.pose()[0].copy(),.13,.27)
    brain = None
    if policy == "joint":
        brain = SharedBrain().eval()
        brain.load_state_dict(torch.load(JOINT_WEIGHTS,map_location="cpu",weights_only=True)["brain"])
        neck = SeparatedLink()
        motor = load_body_policy()
    else:
        neck = ApproachLink()
        motor = Motion(body)
        if policy == "detour":
            approach.STOP = 1.46
            brain = WiderDetour()
        elif policy == "approach":
            brain = approach.Policy()
    source = neck.transport if policy=="joint" else neck.link
    files = [Path(__file__),SCENES,START,BODY.PAD_XML,LIB,APPROACH/"APPROACH_BINDING.json",
             APPROACH/"approach_runtime.py",FAR/"visual_rejoin.py",REJOIN/"wider.py",REJOIN/"rejoin.py",
             FAR/"far_navigation.py",FAR/"far_optics.py",FAR/"FAR_EYES.pt",AN_BINDING,JOINT_WEIGHTS,
             JOINT/"shared_brain.py",APPROACH_WEIGHTS]
    files = [Path(p) for p in files if Path(p).is_file()]
    pinned = {str(p):sha(p) for p in files}
    assert pinned[str(START)] == turn.read(turn.STARTS/"BANK.json")["nominal"]["sha256"]
    turn.stand.put(out/"RUN_IDENTITY.json",dict(scene=scene_name,parameters=scene,policy=policy,geometry=geometry,inputs=pinned))
    print(scene_name,policy,geometry,flush=True)
    if preview:
        image = eyes.image(target,WIDTH,COLOR,background)
        assert image.shape[-1]==3
        np.save(out/"INITIAL_EYE.npy",image)
        eyes.close()
        return
    footprint = Footprint(body)
    start = time.monotonic()
    measured = body.measure()
    trace, visual_pose, decisions, views, native_contact = [],[],[],[],[]
    duration = scene["ms"]
    memory, copy, visual = None,0,-4.
    intended, estimate = (0.,0),(0.,0.,0.)
    active_last = 0
    action = np.zeros(42)
    afferents = np.zeros(2)
    under = 0
    state_tick = 0
    with torch.inference_mode():
        try:
            for ms in range(duration):
                if ms%20==0:
                    if policy != "neck_probe":
                        sample = eyes.capture(ms,target if ms>=START_MS else None,WIDTH,COLOR,background)
                        if sample is not None:
                            seen, image = sample
                            views.append((seen,image))
                            if policy == "joint":
                                visual = float(brain.visual_feature(torch.as_tensor(image[None])).item())
                                estimate = (0.,0.,visual)
                            else:
                                intended,estimate = brain.act(image)
                    else:
                        intended = (.08318,0) if ms>=START_MS else (0.,0)
                    if policy != "joint":
                        neck.send(float(intended[0]),int(intended[1]))
                    visual_pose.append((ms,body.d.qpos.copy(),footprint.bounds(body.d.qpos)))
                if ms%5==0:
                    if policy == "joint":
                        signals,valid = neck.observe_packet()
                        x = torch.tensor(np.r_[visual,signals,float(valid),copy],dtype=torch.float32)
                        logits,_,memory = brain.advance(x[None,None],memory)
                        copy = int(logits.item()>0.)
                        neck.send(copy)
                        intended = (copy,0)
                        v,w = neck.action(),0
                        action = motor(turn.features(body,measured),v,w)
                    else:
                        v,w = neck.current()
                        action = motor.action(turn.features(body,measured),v,w)
                    decisions.append((ms,*intended,v,w,*estimate))
                    if v or w:
                        active_last = ms
                if policy == "joint":
                    normal = np.clip(measured["loads"][FRONT]/(SCALE*W),0.,1.)
                    neck.sense(np.r_[afferents,normal].tolist())
                trace.append(dict(com=body.d.subtree_com[body.th].copy(),
                            head=eyes.pose()[0].copy(),yaw=float(np.arctan2(body.d.xmat[body.th,3],body.d.xmat[body.th,0])),
                            loads=measured["loads"].copy(),speed=measured["speed"].copy(),foot=measured["foot"].copy(),nonfoot=float(measured["nonfoot"]),time=float(body.d.time),
                            upright=float(body.d.xmat[body.th,8])))
                rec.reset_ms()
                measured = body.tick(action)
                neck.advance()
                rec.peak = np.maximum(rec.peak,rec.frame)
                native_contact.append(rec.frame.copy())
                afferents = np.clip(rec.sides/(80*W*.02),0.,1.)
                assert not (body.d.qfrc_applied.any() or body.d.xfrc_applied.any())
                state_tick = ms+1
                under = under+1 if body.d.subtree_com[body.th,2]<.85 or measured["nonfoot"]>.5*W else 0
                if under>=100 or (getattr(brain,"phase",None)=="done" and ms-active_last>=1600):
                    break
                if state_tick%10000==0:
                    print(f"physical {state_tick*.001:.0f}s; wall {time.monotonic()-start:.1f}s",flush=True)
        finally:
            eyes.close()
    trace.append(dict(com=body.d.subtree_com[body.th].copy(),head=eyes.pose()[0].copy(),
                 yaw=float(np.arctan2(body.d.xmat[body.th,3],body.d.xmat[body.th,0])),
                 loads=measured["loads"].copy(),speed=measured["speed"].copy(),foot=measured["foot"].copy(),nonfoot=float(measured["nonfoot"]),time=float(body.d.time),
                 upright=float(body.d.xmat[body.th,8])))
    assert state_tick*80==rec.count
    visual_pose.append((state_tick,body.d.qpos.copy(),footprint.bounds(body.d.qpos)))
    trace = {k:np.asarray([row[k] for row in trace]) for k in trace[0]}
    decision_array = np.asarray(decisions)
    stamps, poses, feet = map(np.asarray,zip(*visual_pose))
    result = score(trace,feet,target,geometry,scene,policy,decision_array,rec,brain)
    result.update(geometry=geometry,scene=scene_name,policy=policy,actual_ms=state_tick,wall_s=time.monotonic()-start,
                  max_duration_ms=duration,unsafe_termination=under>=100,source=source.verify_final())
    np.savez_compressed(out/"TRACE.npz", **trace, decisions=decision_array, goal=target, stamps=stamps, qpos=poses,
                       feet=feet, native_contact_max_by_ms=native_contact,
                       eye_stamps=[v[0] for v in views], eyes=[v[1] for v in views])
    turn.stand.put(out/"TRANSPORT.json",source.journal())
    for p, digest in pinned.items():
        assert sha(Path(p))==digest
    result["artifacts"] = {name:sha(out/name) for name in ("TRACE.npz","RUN_IDENTITY.json","TRANSPORT.json")}
    turn.stand.put(out/"RESULT.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("policy")
    parser.add_argument("out",type=Path)
    parser.add_argument("--preview",action="store_true")
    args = parser.parse_args()
    run(args.scene,args.policy,args.out.resolve(),args.preview)
