"""Check score ordering on archived physical trajectories and adversarial traces."""
import json
from pathlib import Path

import numpy as np

from reward import ACTOR_S, GAMMA, Measurement, Reward, Task

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"20261116T235500Z-navigation-benchmark"
DATA=json.loads((BASE/"BENCHMARK.json").read_text())
SCENES=json.loads((BASE/"SCENES.json").read_text())["scenes"]


def total(task,start,trace):
    reward=Reward(task)
    discount=1.
    score=undiscounted=0.
    sums={k:0. for k in ("event","time","potential","toe")}
    for measurement,dt,end in trace:
        step=reward.step(start,measurement,dt=dt,deadline=end)
        score+=discount*step["reward"]
        undiscounted+=step["reward"]
        for k,value in step["terms"].items():
            sums[k]+=discount*value
        if step["terminated"]:
            return dict(reason=step["reason"],discounted=score,undiscounted=undiscounted,
                        seconds=round(-np.log(discount)/(-np.log(GAMMA))*ACTOR_S+dt,5),
                        discounted_terms=sums)
        discount*=GAMMA**(dt/ACTOR_S)
        start=measurement.distance
    return dict(reason="archive_ended_before_task_terminal",discounted=score,undiscounted=undiscounted,
                discounted_terms=sums)


def from_archive(tag,case,may_step=False):
    assert case["policy"]!="neck_probe" or case["pit_stats"]
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
    task=Task(float(distance[500]),tuple(scene["target_range"]),bool(edges),may_step)
    forces=tracks["native_contact_max_by_ms"]/W
    com=tracks["com"]
    ds=tracks["decisions"]
    end=case["actual_ms"]
    body_stamps=tracks["stamps"]
    feet=tracks["feet"]
    def transitions():
        for a in range(500,end,20):
            b=min(a+20,end)
            step=(b-a)*.001
            f=forces[a:b].max(0)
            cmd=ds[(ds[:,0]>=a)&(ds[:,0]<b),3:5]
            bound=feet[np.searchsorted(body_stamps,b,side="right")-1,0]
            force=tracks["loads"][a+1:b+1]/W
            m=Measurement(float(distance[b]), float(bearing[b]),float(np.linalg.norm(com[b,:2]-com[a,:2])/step),
                          float(tracks["upright"][a+1:b+1].min()),float(com[a+1:b+1,2].min()),
                          float(force.sum(1).mean()),int((force>=.02).sum(1).min()),
                          nonfoot_W=float(tracks["nonfoot"][a+1:b+1].max()/W),
                          obstacle_peak_W=float(f[:len(boxes)].max()) if boxes else 0.,
                          pit_bottom_peak_W=float(f[-1]) if pit else 0.,
                          cleared=bool(not edges or bound>=max(edges)+.03),neutral_command=not bool(cmd.any()))
            # Archived telemetry has total pit-floor force. Native torso-vs-toe
            # tagging will be enabled in the SAC environment. Never backfill it.
            yield m,step,b==end and end==scene["ms"]
    return total(task,float(distance[500]),transitions())


def examples():
    task=Task(5.,(1.65,2.10),True)
    def generate(kind):
        def row(d,**kw):
            defaults=dict(bearing_deg=5.,com_xy_speed=.08,upright=1.,com_z=1.275,
                          support_W=1.,supporting_feet=4)
            return Measurement(d,**(defaults|kw))
        for i in range(1,3001):
            t=i*.02
            if kind=="idle":
                m=row(5.,com_xy_speed=0.,neutral_command=True)
            elif kind=="oscillating":
                m=row(5.-.25*(1.-np.cos(i*np.pi/25.)))
            elif kind=="approach_without_stop":
                m=row(max(1.90,5.-.08*t),cleared=t>=30.)
            elif kind=="early_stop":
                distance=max(3.1,5.-.08*t)
                m=row(distance,com_xy_speed=0. if t>24 else .08,neutral_command=t>24)
            elif kind=="success":
                distance=max(1.93,5.-.08*t)
                m=row(distance,cleared=t>=28.,com_xy_speed=0. if t>39. else .08,neutral_command=t>39.)
            elif kind=="overshoot":
                m=row(max(.1,5.-.1*t),cleared=t>25.)
            elif kind=="contact_shortcut":
                m=row(max(1.87,5.-.085*t),cleared=t>30.,obstacle_peak_W=.001 if i==1700 else 0.)
            elif kind=="pit_bottom":
                m=row(max(1.83,5.-.08*t),pit_bottom_peak_W=.31 if i==1650 else 0.)
            elif kind=="pit_fall":
                m=row(max(1.83,5.-.08*t),pit_body_peak_W=.03 if i==1650 else 0.)
            elif kind=="arrival_before_course_complete":
                m=row(max(1.93,5.-.08*t),com_xy_speed=0. if t>39. else .08,neutral_command=t>39.)
            else:
                raise ValueError(kind)
            yield m,.02,i==3000
    names=("idle","oscillating","approach_without_stop","early_stop","success","overshoot","contact_shortcut",
           "pit_bottom","pit_fall","arrival_before_course_complete")
    out={name:total(task,5.,generate(name)) for name in names}
    assert out["success"]["reason"]=="success" and out["success"]["discounted"]>0.
    assert out["pit_fall"]["reason"]=="fall"
    assert out["contact_shortcut"]["reason"]=="obstacle_contact"
    assert out["pit_bottom"]["reason"]=="forbidden_pit_bottom"
    assert out["overshoot"]["reason"]=="overshoot"
    for key in ("idle","oscillating","early_stop","approach_without_stop","arrival_before_course_complete"):
        assert out[key]["reason"]=="deadline" and out[key]["discounted"]<0.
    assert abs(out["oscillating"]["discounted"]-out["idle"]["discounted"])<1.e-10
    assert out["success"]["discounted"]>out["early_stop"]["discounted"]>out["idle"]["discounted"]
    assert out["idle"]["discounted"]>out["contact_shortcut"]["discounted"]>out["pit_fall"]["discounted"]
    partial=Reward(task).step(5.,Measurement(4.9,0.,.08,1.,1.275,1.,4),truncated=True)
    assert partial["truncated"] and not partial["terminated"] and partial["terms"]["event"]==0.
    return out


if __name__=="__main__":
    # Load board modules through the same project path initialization.
    import unified
    from touch_runtime import sha
    physical={}
    for tag,record in DATA["cases"].items():
        if record["policy"]=="neck_probe" and not record["pit_stats"]:
            continue
        source=BASE/"RUNS"/tag/"TRACE.npz"
        assert sha(source)==record["artifacts"]["TRACE.npz"]
        physical[tag]=from_archive(tag,record)
    for tag in ("narrow_trench__neck_probe","wide_trench__neck_probe"):
        physical[tag+"__shallow_rules"]=from_archive(tag,DATA["cases"][tag],may_step=True)
    synthetic=examples()
    assert physical["staggered_boxes__detour"]["reason"]=="obstacle_contact"
    assert physical["wide_trench__neck_probe"]["reason"]=="forbidden_pit_bottom"
    assert physical["narrow_trench__neck_probe"]["reason"]=="forbidden_pit_bottom"
    assert physical["near_left__approach"]["reason"]=="success"
    result=dict(passed=True,synthetic=synthetic,physical=physical,
                notes="toe-vs-torso native labels absent in benchmark; physical fall outcome not assumed")
    (HERE/"REWARD_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    for category in ("synthetic","physical"):
        for name,score in result[category].items():
            print(category,name,score["reason"],round(score["discounted"],4))
