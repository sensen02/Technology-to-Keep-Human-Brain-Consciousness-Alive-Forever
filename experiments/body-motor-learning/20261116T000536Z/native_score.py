"""Recompute physical engineering gates using all native samples in each window."""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from walking_bench import W, stand, read


def inspect(run):
    result=read(run/"RESULT.json")
    if not result["raw_native_contacts"]:
        raise ValueError("Native contact recording required")
    walk=int(result["walk_ms"])
    with np.load(run/"TRACE.npz",allow_pickle=False) as sample:
        assert len(sample["time"]) == result["actual_ms"]+1
        assert stand.sha(run/"TRACE.npz")==result["trace_sha256"]
        start=sample["time"][0]
        walk_before=sample["com"][500].copy()
        walk_after=sample["com"][500+walk].copy()
        quiet_speed=[]
        for a,b in ((300,500),(500+walk+1000,500+walk+1200)):
            s=np.diff(sample["com"][a:b+1,:2],axis=0)/(.001*stand.LENGTH)
            quiet_speed.append(float(np.sqrt(np.square(s).sum(1).mean())))
        pose=sample["upright"].copy()
        millisecond_pos=sample["qpos"].copy()
        millisecond_vel=sample["qvel"].copy()
    spans={"walking":(500,500+walk),"initial_stand":(300,500),"terminal_stand":(500+walk+1000,500+walk+1200)}
    rows={p:dict(count=0, load=np.zeros(6),slip_load=np.zeros(6),nonfoot=0.,com_min=np.inf,
                three_legs=0,min_pitch_dot=1.) for p in spans}
    previous=0
    dt=stand.physics.NATIVE_DT
    inputs={}
    feet_hist=[]
    foot_support=[]
    signed_residual=[]
    free_fall=0
    alpha=float(-np.expm1(-dt/.001))
    filter_tail=None
    native_gravity=abs(read(run/"RUN_IDENTITY.json")["physics"]["gravity"][2])
    for p in sorted((run/"native").glob("native_*.npz")):
        inputs[p.name]=stand.sha(p)
        with np.load(p,allow_pickle=False) as data:
            n=len(data["time_before_s"])
            idx=np.arange(previous,previous+n)
            np.testing.assert_allclose(data["time_before_s"],start+idx*dt,atol=5e-8,rtol=0.)
            ms_end=idx%80 == 79
            ms_indices=(idx[ms_end]+1)//80
            np.testing.assert_array_equal(data["qpos_after"][ms_end],millisecond_pos[ms_indices])
            np.testing.assert_array_equal(data["qvel_after"][ms_end],millisecond_vel[ms_indices])
            end_quat=millisecond_pos[ms_indices,3:7]
            np.testing.assert_allclose(1.-2.*(end_quat[:,1]**2+end_quat[:,2]**2),pose[ms_indices],
                                       atol=2e-12,rtol=0)
            for v in ("external_body_wrench_native","external_generalized_force_native","adhesion_body_wrench_native"):
                if np.any(data[v]):
                    raise AssertionError(f"Extra wrench found in {v}")
            loads=data["loads_native"]
            speed=data["contact_speed_native_per_s"]
            nonfoot=data["nonfoot_force_native"]
            com=np.minimum(data["com_after"][:,2],data["com_before"][:,2])
            quat=data["qpos_before"][:,3:7]
            up=1.-2.*(quat[:,1]**2+quat[:,2]**2)
            deriv=(data["com_velocity_after"][:,2]-data["com_velocity_before"][:,2])/(
                data["time_after_s"]-data["time_before_s"])/native_gravity
            raw_residual=data["force_balance_residual_native"][:,2]/W
            rates=np.stack((deriv,raw_residual))
            if filter_tail is None:
                filter_tail=(1.-alpha)*rates[:,:1]
            filtered,filter_tail=lfilter([alpha],[1.,-(1.-alpha)],rates,axis=1,zi=filter_tail)
            walking=(idx>=500*80)&(idx<(500+walk)*80)
            support=loads[walking].sum(1)/W
            foot_support.append(support)
            force_total=data["ground_force_native"][:,2]/W
            free_fall+=int(np.sum(walking&(force_total<.2)&(filtered[0]<=-.5)))
            signed_residual.append(filtered[1,walking])
            feet_hist.append(np.count_nonzero(loads[walking]>=.10*W,axis=1))
            for phase,(a,b) in spans.items():
                mask=(idx>=a*80)&(idx<b*80)
                if not mask.any():
                    continue
                r=rows[phase]
                q=loads[mask]
                r["count"]+=int(mask.sum())
                r["load"]+=q.sum(0)
                r["slip_load"]+=(q*speed[mask]).sum(0)
                r["nonfoot"]=max(r["nonfoot"],float(nonfoot[mask].max()))
                r["com_min"]=min(r["com_min"],float(com[mask].min()))
                r["min_pitch_dot"]=min(r["min_pitch_dot"],float(up[mask].min()))
                r["three_legs"]+=int(np.sum((q>=.02*W).sum(1)>=3))
            previous+=n
    assert previous==result["actual_ms"]*80
    output={}
    for key, values in rows.items():
        n=values["count"]
        assert n==(spans[key][1]-spans[key][0])*80
        feet=float(values["load"].sum()/(W*n))
        r=dict(samples=n, foot_support_mean_w=feet, nonfoot_peak_w=values["nonfoot"]/W,
               com_min=values["com_min"], min_upright_dot=values["min_pitch_dot"],
               three_foot_fraction=values["three_legs"]/n)
        if key=="walking":
            speed=np.linalg.norm((walk_after-walk_before)[:2])/(walk*.001)
            vt=values["slip_load"]/values["load"]
            signed=float((walk_after[0]-walk_before[0])/(walk*.001))
            assert np.isfinite(vt).all()
            r.update(forward_signed=signed,slip_max=float((vt/speed).max()),
                     per_leg_slip=(vt/speed).tolist(),absolute_contact_speed=vt.tolist(),
                     cycles=result["cycles"],contact_force_zero=values["nonfoot"]==0.)
            r["gates"]={"support":feet>=.9,"nonfoot":values["nonfoot"]<=.2*W,"com":r["com_min"]>=1.2,
                        "forward":signed>=.05,"slip":r["slip_max"]<=.5,"cycles":all(result["cycles"].values())}
        else:
            r["rms_vel"]=quiet_speed[0 if key=="initial_stand" else 1]
            r["gates"]={"support":feet>=.9,"nonfoot":r["nonfoot_peak_w"]<.01,"com":r["com_min"]>=1.2,
                        "three_legs":r["three_foot_fraction"]>=.95,"rms_vel":r["rms_vel"]<=.01,
                        "posture":bool(r["min_upright_dot"]>=np.cos(np.deg2rad(20.)))}
        r["passed"]=bool(all(r["gates"].values()))
        output[key]=r
    r=output["walking"]
    p10=float(np.quantile(np.concatenate(foot_support),.10))
    force=np.concatenate(signed_residual)
    p95=float(np.quantile(force,.95))
    p95_abs=float(np.quantile(np.abs(force),.95))
    median_legs=float(np.median(np.concatenate(feet_hist)))
    # Historical D3/D5 require the same trial to include an auxiliary walking phase.
    d=dict(D1=dict(passed=r["contact_force_zero"],peak_nonfoot_w=r["nonfoot_peak_w"]),
           D2=dict(passed=free_fall==0,free_fall_samples=free_fall),
           D3=dict(status="UNKNOWN",reason="no same-trial auxiliary walking"),
           D4=dict(passed=p10>=.5,foot_w_p10=p10),
           D4b=dict(passed=True,nonfoot_fraction=0.) if r["contact_force_zero"] else dict(status="UNKNOWN"),
           D5=dict(status="UNKNOWN",reason="no same-trial auxiliary walking"),
           D6=dict(passed=p95<=.15,signed_p95_w=p95,absolute_p95_w=p95_abs,causal_filter_ms=1.),
           D7=dict(passed=median_legs>=3,median_legs_at_point_one_w=median_legs),
           D8=dict(passed=r["com_min"]>=.8*1.2072528224791905,minimum_com=r["com_min"]))
    report=dict(windows=output,native_samples=previous,physical_pass=all(r["passed"] for r in output.values()),
                chunk_sha256=inputs,full_D1_D8=False,partial_D1_D8=d,zero_extra_force=True)
    stand.put(run/"NATIVE_AUDIT.json",report)
    print({k:v for k,v in report.items() if k!="chunk_sha256"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    inspect(p.parse_args().run.resolve())
