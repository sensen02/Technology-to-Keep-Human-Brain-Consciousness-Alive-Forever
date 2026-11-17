"""Identify the actual left-hind/floor-tile contact in the X-layout failure."""
from pathlib import Path

import mujoco as mj
import numpy as np

import crossing as q


here=Path(__file__).resolve().parent
run=here/"ANCHOR_COURSE"
repeat=here/"ANCHOR_COURSE_REPEAT"
as_record=q.read(run/"RESULT.json")
compare=("TRACE.npz","DECISIONS.npz","NATIVE.npz","GEOMETRY_NATIVE.npz")
assert all(q.stan.sha(run/k)==q.stan.sha(repeat/k) for k in compare)
initial=q.bench.STARTS/"nominal.npz"
c,_=q.build_scene(as_record["scenario"],initial,q.read(q.bench.STARTS/"BANK.json")["nominal"]["sha256"])
leg=q.bench.LEGS.index("LH")
opening=as_record["scenario"]["opening"]
with np.load(run/"TRACE.npz",allow_pickle=False) as z:
    x=z["qpos"]
    v=z["qvel"]
    u=z["ctrl"]
    com=z["com"]
    lh=z["all_pad_xyz"][:,leg]
    zbottom=z["padz"][:,leg]
    behavior=z["nonfoot"]
    below=np.flatnonzero((zbottom<-.045)&(np.arange(len(x))>q.RELEASE))
    j=int(below[0])
    first=np.flatnonzero((behavior>.01*q.W)&(np.arange(len(x))>q.RELEASE))
    interior=lambda a: bool(opening[0]<a[0]<opening[1] and opening[2]<a[1]<opening[3])
    def replay(ms):
        c.d.qpos[:]=x[ms]
        c.d.qvel[:]=v[ms]
        c.d.ctrl[:]=u[ms]
        mj.mj_forward(c.plant,c.d)
        hits=[]
        for contact in c.d.contact:
            p,r=int(contact.geom1),int(contact.geom2)
            if (c.geom_leg[p]==leg and c.planes[r]) or (c.geom_leg[r]==leg and c.planes[p]):
                hits.append(dict(pair=[c.plant.geom(p).name,c.plant.geom(r).name],
                        dist=float(contact.dist),contact_xyz=contact.pos.tolist(),
                        native_normal=contact.frame[:3].tolist(),side_abs_normal_z=float(abs(contact.frame[2]))))
        return dict(ms=ms,LH_pad_xyz=lh[ms].tolist(),LH_bottom_z=float(zbottom[ms]),
                    COM=com[ms].tolist(),all_body_nonfoot_W=float(behavior[ms]/q.W),geometric_replay_contacts=hits)
    ms_rows=sorted(set([int(first[0]),j-20,j-5,j,j+3,j+10,j+35,j+75,len(x)-1]))
    records=[replay(ms) for ms in ms_rows]
    p_cross=np.flatnonzero(com[:,0]>opening[1])
    result=dict(scope="shortened actual X-first course, left hind tile seam",
        true_gap=opening,actual_native_integrations=as_record["native_iterations"],
        independently_exact=True,first_COM_x_crossing_ms=int(p_cross[0]) if len(p_cross) else None,
        restored_trace_forward_tests_only=True,
        LH_center_inside_real_gap_at_depth_event=interior(lh[j]),
        first_LH_depression_below_045_ms=j,first_nonfoot_above_01W_ms=int(first[0]),
        last_COM=as_record["COM_native_end"],last_pads=as_record["tail_native_end"],contact_geometry=records)
q.stan.put(here/"X_SEAM_DIAGNOSIS.json",result)
print(dict(LH_drop=j,first_nonfoot=int(first[0]),COM_past=result["first_COM_x_crossing_ms"],
           old_left_right_tile_oblique_side_at_drop=[r for r in records if any(c["side_abs_normal_z"]<.5 for c in r["geometric_replay_contacts"])]))
