"""Check score ordering on archived physical trajectories and adversarial traces."""
import json
from pathlib import Path

import numpy as np

from reward import (ACTOR_S, CONTACT_W, GAMMA, PROGRESS_MAX, PROGRESS_PER_UNIT,
                    Measurement, Reward, Task)
from stall import Motion

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"20261116T235500Z-navigation-benchmark"
DATA=json.loads((BASE/"BENCHMARK.json").read_text())
SCENES=json.loads((BASE/"SCENES.json").read_text())["scenes"]
UNIFIED=HERE.parent/"20261117T010100Z-unified-neck-final"


def total(task,start,trace):
    reward=Reward(task)
    discount=1.
    score=undiscounted=elapsed=0.
    sums={k:0. for k in ("event","time","potential","toe","stall","progress")}
    stall_first={}
    contacts=0
    first_contact=None
    for measurement,dt,end in trace:
        step=reward.step(start,measurement,dt=dt,deadline=end)
        score+=discount*step["reward"]
        undiscounted+=step["reward"]
        for k,value in step["terms"].items():
            sums[k]+=discount*value
        elapsed+=dt
        if step["pit_bottom_contact"]:
            contacts+=1
            if first_contact is None:
                first_contact=round(elapsed,5)
        if step["stall_kind"]:
            stall_first.setdefault(step["stall_kind"],round(elapsed,5))
        if step["terminated"]:
            break
        discount*=GAMMA**(dt/ACTOR_S)
        start=measurement.distance
    return dict(reason=step["reason"] or "archive_ended_before_task_terminal",discounted=score,
                undiscounted=undiscounted,seconds=round(elapsed,5),discounted_terms=sums,
                stall_first_s=stall_first,bottom_contact_steps=contacts,first_bottom_contact_s=first_contact,
                raw_toe_cost=reward.toe_spent,raw_progress=reward.progress_paid)


def from_archive(tag,case,may_step=False):
    path=BASE/"RUNS"/tag/"TRACE.npz"
    scene=SCENES[case["scene"]]
    assert case["actual_ms"]<=scene["ms"]
    with np.load(path,allow_pickle=False) as z:
        tracks={key:z[key] for key in ("head","goal","yaw","com","loads","nonfoot","upright","decisions",
                                       "stamps","feet","native_contact_max_by_ms")}
    # Recover the archived project's W from the same floating calibration.
    from touch_runtime import W as native_weight
    W=native_weight
    target=tracks["goal"][:2]-tracks["head"][:,:2]
    distance=np.linalg.norm(target,axis=1)
    azimuth=np.degrees(np.arctan2(target[:,1],target[:,0])-tracks["yaw"])
    bearing=(azimuth+180.)%360.-180.
    boxes=case["geometry"]["boxes"]
    pit=case["geometry"]["pit"]
    edges=[x+hx for x,_,_,hx,_,_ in boxes]
    if pit:
        edges.append(pit["far"])
    task=Task(float(distance[500]),tuple(scene["target_range"]),bool(edges),may_step,require_motion=True)
    forces=tracks["native_contact_max_by_ms"]/W
    com=tracks["com"]
    yaw=np.unwrap(tracks["yaw"])
    ds=tracks["decisions"]
    end=case["actual_ms"]
    body_stamps=tracks["stamps"]
    feet=tracks["feet"]
    def transitions():
        last_pit_touch=-1_000
        for a in range(500,end,20):
            b=min(a+20,end)
            step=(b-a)*.001
            f=forces[a:b].max(0)
            cmd=ds[(ds[:,0]>=a)&(ds[:,0]<b),3:5]
            bound=feet[np.searchsorted(body_stamps,b,side="right")-1,0]
            force=tracks["loads"][a+1:b+1]/W
            stamp=np.searchsorted(body_stamps,b,side="right")-1
            x,y=map(float,com[b,:2])
            bottom=float(f[-1]) if pit else 0.
            pit_id=pit_progress=None
            if pit:
                if bottom>CONTACT_W:
                    last_pit_touch=b
                rear,front=feet[stamp]
                overlaps=bool(front>=pit["near"] and rear<=pit["far"])
                if pit["near"]<=x<=pit["far"] or (overlaps and b-last_pit_touch<=600):
                    pit_id=case["scene"]
                    pit_progress=x-pit["far"]
            speed=float(np.sqrt(np.square(np.diff(com[a:b+1,:2],axis=0)*1000.).sum(1).mean()))
            yaw_speed=float(np.sqrt(np.square(np.degrees(np.diff(yaw[a:b+1]))*1000.).mean()))
            m=Measurement(float(distance[b]), float(bearing[b]),speed,
                          float(tracks["upright"][a+1:b+1].min()),float(com[a+1:b+1,2].min()),
                          float(force.sum(1).mean()),int((force>=.02).sum(1).min()),
                          yaw_speed_deg_s=yaw_speed,nonfoot_W=float(tracks["nonfoot"][a+1:b+1].max()/W),
                          obstacle_peak_W=float(f[:len(boxes)].max()) if boxes else 0.,
                          pit_bottom_peak_W=bottom,
                          cleared=bool(not edges or bound>=max(edges)+.03),neutral_command=not bool(cmd.any()),
                          motion=Motion((x,y),float(np.degrees(yaw[b])),pit_id,pit_progress))
            # Archived telemetry has total pit-floor force. Native torso-vs-toe
            # tagging will be enabled in the SAC environment. Never backfill it.
            yield m,step,b==end and end==scene["ms"]
    return total(task,float(distance[500]),transitions())


def regress_navigation(name,meta):
    """Monitor verified 5 ms control captures without imposing a new target band."""
    from stall import Stall
    path=UNIFIED/"RUNS"/name/"TRACE.npz"
    from touch_runtime import sha
    assert sha(path)==meta["hashes"]["TRACE.npz"]
    with np.load(path,allow_pickle=False) as z:
        com,yaw=z["com"],np.degrees(z["yaw"])
    scene=meta["scene"]
    s=Stall()
    found=set()
    for b in range(20,scene["end"]+1,20):
        # The regression fixture marks the visible task interval. It also
        # supplies the known stop when the verified controller reaches its band.
        active=scene["on"]<=b<scene["off"]
        if name in ("left","right","pace") and meta["stop_ms"] is not None and b>=meta["stop_ms"]:
            active=False
        kind,_,stuck=s.update(Motion(tuple(com[b,:2]),float(yaw[b])),.02,active=active)
        if kind is not None:
            found.add(kind)
        assert not stuck, name
    return sorted(found)


def examples():
    task=Task(5.,(1.65,2.10),True,require_motion=True)

    def row(d, *, x=None, y=0., yaw=0., pit=False, **kw):
        defaults=dict(bearing_deg=5.,com_xy_speed=.08,upright=1.,com_z=1.275,
                      support_W=1.,supporting_feet=4)
        travel=5.-d if x is None else x
        pose=Motion((travel,y),yaw,"pit_a" if pit else None,travel if pit else None)
        return Measurement(d,motion=pose,**(defaults|kw))

    def generate(kind):
        steps=4500 if kind=="late_collision" else 3000
        for i in range(1,steps+1):
            t=i*.02
            if kind in ("idle","blank"):
                m=row(5.,com_xy_speed=0.,neutral_command=True,navigation_active=kind=="idle")
            elif kind=="oscillating":
                m=row(5.-.25*(1.-np.cos(i*np.pi/25.)))
            elif kind=="spin":
                m=row(5.,com_xy_speed=0.,yaw=(t*8.+180.)%360.-180.,yaw_speed_deg_s=8.)
            elif kind=="rocking_spin":
                m=row(5.,com_xy_speed=0.,yaw=30.*np.sin(t*np.pi),yaw_speed_deg_s=40.)
            elif kind=="pit_stuck":
                m=row(4.6,x=.4,y=.06*np.sin(t*5),pit=True,pit_bottom_peak_W=.20)
            elif kind=="pit_rocking":
                m=row(4.6,x=.40+.11*np.sin(t*4.),y=.03*np.cos(t*4.),pit=True,pit_bottom_peak_W=.20)
            elif kind=="switch_pits":
                id_="a" if int(t/3.)%2==0 else "b"
                m=Measurement(5.,0.,0.,1.,1.275,1.,4,motion=Motion((0.,0.),0.,id_,0.))
            elif kind=="early_stop":
                m=row(max(3.1,5.-.08*t),com_xy_speed=0. if t>24 else .08,neutral_command=t>24)
            elif kind in ("success","toe_then_success","pause_then_success","pivot_then_success"):
                delay=max(0.,min(t-5.,3.)) if kind=="pause_then_success" else 0.
                if kind=="pivot_then_success":
                    delay=min(t,9.6)
                effective=t-delay
                distance=max(1.93,5.-.085*effective)
                waiting=(kind=="pause_then_success" and 5.<t<8.) or (kind=="pivot_then_success" and t<=9.6)
                yaw=min(t,9.6)*7.5 if kind=="pivot_then_success" else 0.
                toes=kind=="toe_then_success" and 6.<t<20.
                m=row(distance,pit=toes,yaw=yaw,pit_bottom_peak_W=.31 if toes else 0.,
                      cleared=effective>=26.,com_xy_speed=0. if waiting or effective>37. else .085,
                      neutral_command=waiting or effective>37.)
            elif kind=="safe_detour":
                distance=5.+.05*t if t<=5. else max(1.93,5.25-.091*(t-5.))
                m=row(distance,cleared=t>=32.,com_xy_speed=0. if t>42. else .08,neutral_command=t>42.)
            elif kind=="overshoot":
                m=row(max(.1,5.-.1*t),cleared=t>25.)
            elif kind=="late_collision":
                m=row(5.-.20*(1.-np.cos(t*np.pi/2.)),obstacle_peak_W=.001 if i==4400 else 0.)
            elif kind=="contact_shortcut":
                m=row(max(1.87,5.-.085*t),cleared=t>30.,obstacle_peak_W=.001 if i==1700 else 0.)
            elif kind=="pit_fall":
                m=row(max(1.83,5.-.08*t),pit_body_peak_W=.03 if i==1650 else 0.,
                      pit_bottom_peak_W=.31 if i==1650 else 0.)
            elif kind=="arrival_before_course_complete":
                m=row(max(1.93,5.-.08*t),com_xy_speed=0. if t>39 else .08,neutral_command=t>39.)
            elif kind=="rollout_recovery":
                travel=max(0.,t-6.)*.09
                d=max(1.93,5.-travel)
                m=row(d,cleared=t>=33.,com_xy_speed=0. if t<=6. or t>42. else .09,
                      neutral_command=t<=6. or t>42.)
            else:
                raise ValueError(kind)
            yield m,.02,i==steps

    names=("idle","blank","oscillating","spin","rocking_spin","pit_stuck","pit_rocking","switch_pits","early_stop","success",
           "toe_then_success","pause_then_success","pivot_then_success","overshoot","contact_shortcut","pit_fall",
           "arrival_before_course_complete","safe_detour","rollout_recovery","late_collision")
    out={name:total(task,5.,generate(name)) for name in names}
    for key in ("success","toe_then_success","pause_then_success","pivot_then_success","safe_detour","rollout_recovery"):
        assert out[key]["reason"]=="success" and out[key]["discounted"]>0., (key,out[key])
    for key in ("pause_then_success","pivot_then_success","safe_detour","blank","switch_pits"):
        assert out[key]["stall_first_s"]=={}, (key,out[key])
    assert out["rollout_recovery"]["stall_first_s"].keys()=={"idle"}
    assert out["toe_then_success"]["raw_toe_cost"]==3.
    expected={"idle":"stalled_idle","spin":"stalled_spin","rocking_spin":"stalled_spin","pit_stuck":"stalled_pit",
              "pit_rocking":"stalled_pit","early_stop":"stalled_idle","overshoot":"overshoot","contact_shortcut":"obstacle_contact",
              "pit_fall":"fall","arrival_before_course_complete":"stalled_idle","late_collision":"obstacle_contact"}
    for kind,label in expected.items():
        assert out[kind]["reason"]==label and out[kind]["discounted"]<0., (kind,out[kind])
    assert out["pit_fall"]["raw_toe_cost"]==0.
    for key in ("idle","spin","rocking_spin","pit_stuck"):
        assert out[key]["discounted"]>out["late_collision"]["discounted"], (key,out[key],out["late_collision"])
    assert out["idle"]["seconds"]>12. and out["pit_stuck"]["seconds"]>13.
    assert out["blank"]["reason"]==out["oscillating"]["reason"]=="deadline"
    assert np.isclose(out["oscillating"]["raw_progress"],PROGRESS_PER_UNIT*.5)
    assert out["blank"]["raw_progress"]==out["idle"]["raw_progress"]==0.
    np.testing.assert_allclose(out["oscillating"]["discounted"]-out["blank"]["discounted"],
                               out["oscillating"]["discounted_terms"]["progress"],rtol=0.,atol=1e-10)

    repeat=Reward(task)
    first_credit=repeat.step(5.,row(4.6))["terms"]["progress"]
    for _ in range(9):
        for before,after in ((4.6,5.),(5.,4.6)):
            step=repeat.step(before,row(after))
            assert step["terms"]["progress"]==0.
    assert np.isclose(first_credit,PROGRESS_PER_UNIT*.4)

    long=Reward(Task(15.,(1.65,2.10),False,require_motion=True))
    for before in np.arange(15.,2.24,-.02):
        step=long.step(float(before),row(float(before-.02)))
        assert not step["terminated"]
    assert long.progress_paid==PROGRESS_MAX
    on_contact=Reward(task).step(5.,row(4.9,obstacle_peak_W=.002))
    assert on_contact["reason"]=="obstacle_contact" and on_contact["terms"]["progress"]==0.
    first_valid=Reward(task)
    no_goal=first_valid.step(5.,row(4.8,navigation_active=False))
    goal_back=first_valid.step(4.8,row(4.7))
    assert no_goal["terms"]["progress"]==0. and np.isclose(goal_back["terms"]["progress"],.4)

    probe=Reward(task)
    for i in range(29):
        assert not probe.step(1.93,row(1.93,com_xy_speed=0.,cleared=True,neutral_command=True))["terminated"]
    conflict=probe.step(1.93,row(1.93,com_xy_speed=0.,cleared=True,neutral_command=True,obstacle_peak_W=.002))
    assert conflict["reason"]=="obstacle_contact" and conflict["terms"]["event"]==-80.

    toe=Reward(task).step(5.,row(4.98,pit=True,pit_bottom_peak_W=.35))
    assert not toe["terminated"] and toe["pit_bottom_contact"] and np.isclose(toe["terms"]["toe"],-.015)
    partial=Reward(task).step(5.,row(4.9),truncated=True)
    assert partial["truncated"] and not partial["terminated"] and partial["terms"]["event"]==0.
    without_motion=Measurement(4.99,0.,.08,1.,1.275,1.,4)
    try:
        Reward(task).step(5.,without_motion)
    except ValueError as exc:
        assert "physical motion telemetry" in str(exc)
    else:
        raise AssertionError("New SAC configuration accepted an active step without telemetry")
    legacy=Reward(Task(5.,(1.65,2.10),True)).step(5.,without_motion)
    assert legacy["stall_kind"] is None and np.isfinite(legacy["reward"])
    return out


if __name__=="__main__":
    # Load board modules through the same project path initialization.
    import sys
    sys.path.append(str(UNIFIED))
    import unified
    from touch_runtime import sha
    physical={}
    for tag,record in DATA["cases"].items():
        source=BASE/"RUNS"/tag/"TRACE.npz"
        assert sha(source)==record["artifacts"]["TRACE.npz"]
        physical[tag]=from_archive(tag,record)
    for tag in ("narrow_trench__neck_probe","wide_trench__neck_probe"):
        physical[tag+"__shallow_rules"]=from_archive(tag,DATA["cases"][tag],may_step=True)
    synthetic=examples()
    assert physical["staggered_boxes__detour"]["reason"]=="obstacle_contact"
    assert physical["wide_trench__neck_probe"]["reason"]=="stalled_pit", physical["wide_trench__neck_probe"]
    assert physical["wide_trench__joint"]["reason"]=="stalled_pit", physical["wide_trench__joint"]
    assert not physical["narrow_trench__neck_probe"]["stall_first_s"], physical["narrow_trench__neck_probe"]
    assert physical["narrow_trench__neck_probe"]["reason"]=="overshoot"
    assert physical["flat_paired__joint"]["reason"]=="stalled_idle"
    assert physical["near_left__approach"]["reason"]=="success"
    unified_cases=json.loads((UNIFIED/"ACCEPTANCE.json").read_text())["cases"]
    regression={name:regress_navigation(name,record) for name,record in unified_cases.items()}
    assert all(not flags for flags in regression.values()), regression
    result=dict(passed=True,synthetic=synthetic,physical=physical,unified_navigation_regression=regression,
                notes="toe-vs-torso native labels absent in benchmark; physical fall outcome not assumed")
    (HERE/"REWARD_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    for category in ("synthetic","physical"):
        for name,score in result[category].items():
            print(category,name,score["reason"],round(score["discounted"],4))
