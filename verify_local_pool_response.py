"""Offline NumPy verifier for the local motor-pool response experiment.

This module only reads runner artefacts.  It deliberately does not import MuJoCo,
run a neural model, or rerun physics.  It verifies the declared pool perturbation,
independent FIR replay, sparse actuator-moment decoding and saved dynamics balance.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

from verify_neural_initialized_startup import (
    ROOT, MANIFEST, BINS, DT, require, equal, sha, load_json, load_npz,
    fir_recompute, support_metrics,
)

RUNNER = ROOT / "run_local_pool_response.py"
ARMS = ("baseline", "down", "up")
PRE, TICKS, SAMPLES = 80, 20, 101
POOL = np.array([3], dtype=int)
BINS_LOCAL = np.array([1, 2, 3], dtype=int)
DT_BODY = 1e-4

def pick(a, *names, required=True):
    for n in names:
        if n in a: return a[n]
    if required: raise ValueError("Missing field: " + "/".join(names))
    return None

def finite(a, label):
    require(np.isfinite(a).all(), label + " contains nonfinite values")

def sparse_dense(a):
    """Decode MuJoCo actuator_moment CSR rows independently for each sample."""
    val = np.asarray(pick(a, "actuator_moment_raw", "moment_raw"))
    nnz = np.asarray(pick(a, "moment_rownnz")); adr = np.asarray(pick(a, "moment_rowadr")); ind = np.asarray(pick(a, "moment_colind"))
    if nnz.ndim == 1: nnz = np.broadcast_to(nnz, (SAMPLES, nnz.size))
    if adr.ndim == 1: adr = np.broadcast_to(adr, (SAMPLES, adr.size))
    if ind.ndim == 1: ind = np.broadcast_to(ind, (SAMPLES, ind.size))
    if val.ndim == 1: val = np.broadcast_to(val, (SAMPLES, val.size))
    require(nnz.shape == (SAMPLES, 90) and adr.shape == (SAMPLES, 90), "moment row shape")
    require(ind.shape == val.shape and ind.shape[0] == SAMPLES, "moment sparse shape")
    out = np.zeros((SAMPLES, 90, 48), float)
    for t in range(SAMPLES):
        require(np.all(nnz[t] >= 0) and np.all(adr[t] >= 0), "negative sparse pointer")
        for u in range(90):
            lo, hi = int(adr[t,u]), int(adr[t,u] + nnz[t,u])
            require(hi <= ind.shape[1], "sparse row exceeds colind")
            cols = ind[t,lo:hi].astype(int)
            require(np.all((cols >= 0) & (cols < 48)), "moment colind bounds")
            out[t,u,cols] = val[t,lo:hi]
    return out

def check_dynamics(a, label):
    dense = np.asarray(pick(a, "actuator_moment"))
    require(dense.shape == (SAMPLES,90,48), label+" dense moment shape")
    decoded = sparse_dense(a)
    equal(dense, decoded, label+" sparse->dense moment", 0)
    force = np.asarray(pick(a, "actuator_force", "force", "actual_actuator_force"))
    require(force.shape == (SAMPLES,90), label+" actuator force shape")
    torque = np.einsum("tud,tu->td", dense, force)
    qfrc = np.asarray(pick(a, "qfrc_actuator"))
    require(qfrc.shape == (SAMPLES,48), label+" qfrc_actuator shape")
    equal(torque, qfrc, label+" moment.T@actuator_force", 2e-12)
    Ma = np.asarray(pick(a, "Ma", "qfrc_Ma")); bias=np.asarray(pick(a,"qfrc_bias","bias"))
    passive=np.asarray(pick(a,"qfrc_passive","passive")); constraint=np.asarray(pick(a,"qfrc_constraint","constraint"))
    require(Ma.shape == bias.shape == passive.shape == constraint.shape == (SAMPLES,48), label+" dynamics shapes")
    residual = Ma + bias - passive - qfrc - constraint
    applied = pick(a,"qfrc_applied",required=False)
    if applied is not None: residual -= np.asarray(applied)
    finite(residual,label+" residual")
    scale=np.maximum.reduce([np.abs(Ma),np.abs(bias)+np.abs(passive)+np.abs(qfrc)+np.abs(constraint),np.full_like(residual,1e-12)])
    normalized=np.abs(residual)/scale
    require(float(np.max(np.abs(residual))) <= 1e-10,label+" dynamics residual exceeds tolerance")
    require(float(np.max(normalized)) <= 1e-8,label+" normalized dynamics residual exceeds tolerance")
    result=dict(moment_max_abs=float(np.max(np.abs(torque-qfrc))),
                dynamics_residual_max_abs=float(np.max(np.abs(residual))),
                dynamics_residual_normalized_max=float(np.max(normalized)))
    if "actuator_velocity" in a:
        vel=np.asarray(a["actuator_velocity"]); qvel=np.asarray(a["qvel"])
        power_force=np.sum(np.asarray(force)*vel,axis=1)
        power_qfrc=np.sum(np.asarray(qfrc)*qvel,axis=1)
        per=np.maximum.reduce([np.abs(power_force),np.abs(power_qfrc),np.full(power_force.shape,1e-12)])
        require(float(np.max(np.abs(power_force-power_qfrc))) <= 1e-10,label+" power identity")
        result.update(power_error_max_abs=float(np.max(np.abs(power_force-power_qfrc))),power_error_normalized_max=float(np.max(np.abs(power_force-power_qfrc)/per)))
    return result

def metrics(a, weight):
    t=np.asarray(a["time"]); q=np.asarray(a["qpos"]); v=np.asarray(a["qvel"])
    h=q[:,2]/q[0,2]; n=np.linalg.norm(q[:,3:7],axis=1)
    ang=2*np.arccos(np.clip(np.abs(q[:,3:7]@q[0,3:7])/(n*n[0]),0,1))
    contact=np.asarray(pick(a,"contacts")); legs=(contact>0).sum(1)
    load=np.asarray(pick(a,"foot_force"))[:,:,2].sum(1)/weight
    depth=np.asarray(pick(a,"nonfoot_depth")); lim=np.asarray(pick(a,"limit_excess"))
    gates=np.column_stack((h>=.85,ang<=.35,legs>=3,load>=.8,depth<=1e-12,lim<=1e-6))
    total,longest,cum=support_metrics(gates.all(1),t)
    early=slice(0, min(51,SAMPLES))
    def trap(x): return float(np.trapezoid(x,t)*1000)
    return dict(first_failures_ms={k:(float(t[np.flatnonzero(~gates[:,j])[0]]*1000) if (~gates[:,j]).any() else None) for j,k in enumerate(("height","angle","legs","load","nonfoot","limit"))},
      all_gates_support_ms=total,longest_all_gates_true_endpoint_span_ms=longest,
      gate_support_ms={k:support_metrics(gates[:,j],t)[0] for j,k in enumerate(("height","angle","legs","load","nonfoot","limit"))},
      early_support_ms=support_metrics(gates[early].all(1),t[early])[0],
      load_curve_integral_native_ms=trap(load), load_deficit_integral_native_ms=trap(np.maximum(.8-load,0)),
      limit_excess_integral_rad_ms=trap(np.maximum(lim-1e-6,0)),
      min_height_ratio=float(h.min()),max_downward_velocity_native=float(np.maximum(-v[:,2],0).max()),
      max_angle_rad=float(ang.max()),max_nonfoot_depth_native=float(depth.max()),max_limit_excess_rad=float(lim.max()),
      min_load_ratio=float(load.min()),min_contact_legs=int(legs.min()), target_qpos9_max=float(q[:,9].max()),
      target_qvel8_max=float(v[:,8].max()),
      aggregate_joint8_torque_first_01ms=float(np.asarray(a['qfrc_actuator'])[1,8]),
      aggregate_joint8_torque_16ms=float(np.asarray(a['qfrc_actuator'])[16,8]),
      muscle5_joint8_torque_first_01ms=float(np.asarray(a['actuator_moment'])[1,5,8]*np.asarray(a['actuator_force'])[1,5]),
      muscle5_joint8_torque_16ms=float(np.asarray(a['actuator_moment'])[16,5,8]*np.asarray(a['actuator_force'])[16,5]))

def verify(out):
    out=Path(out).resolve(); require("mujoco" not in sys.modules,"physics imported")
    require(load_json(out/"COMPLETE.json").get("status")=="COMPLETE","runner incomplete")
    require(not (out/"FAILED.json").exists(),"FAILED.json present")
    pre=load_json(out/"PREREG.json"); require(pre.get("initialization_ticks",PRE)==PRE,"initialization ticks")
    require(pre.get("physical_ticks",TICKS)==TICKS and pre.get("physical_samples",SAMPLES)==SAMPLES,"timing")
    hashes=pre.get("source_hashes",{}); require(str(RUNNER) in hashes and str(MANIFEST) in hashes,"missing source hashes")
    for p,d in hashes.items(): require(sha(p)==d,"source hash mismatch: "+p)
    results=load_json(out/"RESULTS.json"); require([r["label"] for r in results]==list(ARMS),"arm order")
    ledger=load_json(out/"CALL_LEDGER.json"); require(ledger.get("completed")==list(ARMS),"ledger completion"); require(ledger.get("active") is None,"ledger active")
    require(ledger.get("model_load")==3 and ledger.get("step")==300 and ledger.get("forward")==303 and ledger.get("neural_ticks")==300,"ledger counts")
    direction=load_npz(out/"DIRECTION.npz"); w0=np.asarray(pick(direction,"baseline","weights"))
    require(w0.shape==(66,5),"weights shape")
    expected={"baseline":w0.copy(),"down":w0.copy(),"up":w0.copy()}
    expected["down"][3,BINS_LOCAL]*=.8; expected["up"][3,BINS_LOCAL]*=1.2
    for label in ARMS:
        if label in direction: equal(direction[label],expected[label],label+" exact perturbation")
        other=np.ones(w0.shape,dtype=bool); other[3,BINS_LOCAL]=False
        require(np.array_equal(expected[label][other],w0[other]),label+" other weights changed")
    from engine.embodied.motor_translation import FiniteMotorTranslator
    mapping=FiniteMotorTranslator(load_json(MANIFEST),w0)
    checks=[]; plotted=[]; base_init=None
    for r in results:
        label=r["label"]; d=out/label; a=load_npz(d/"TRAJECTORY.npz"); init=load_npz(d/"INITIALIZATION.npz")
        for name,key in (("TRAJECTORY.npz","trajectory_sha256"),("INITIALIZATION.npz","initialization_sha256")):
            require(sha(d/name)==r[key],label+" archive hash")
        require(r["direction_sha256"]==sha(out/"DIRECTION.npz"),label+" direction hash")
        for p,digest in r.get("hashes",{}).items(): require(sha(p)==digest,label+" artifact hash "+p)
        schedule=load_json(d/"SCHEDULE.json"); require(len(schedule)==TICKS,label+" schedule")
        for k,row in enumerate(schedule):
            require(row["tick"]==k and row["consume_id"]==PRE+k,label+" schedule IDs")
            require(abs(row["source_start_s"]-(PRE+k)*DT)<=1e-12 and abs(row["source_end_s"]-(PRE+k+1)*DT)<=1e-12,label+" source clock")
            require(abs(row["physical_start_s"]-k*DT)<=1e-12 and abs(row["physical_end_s"]-(k+1)*DT)<=1e-12,label+" physical clock")
        require(a["time"].shape==(SAMPLES,),label+" samples"); equal(a["time"],np.arange(SAMPLES)*DT_BODY,label+" clock",1e-12)
        equal(a["qpos"][0],init["body_qpos"],label+" initial qpos")
        if base_init is not None: equal(a["qpos"][0],base_init["body_qpos"],label+" matched initial qpos")
        require(a["spikes"].shape[0]==TICKS and a["features"].shape[0]==TICKS,label+" neural ticks")
        require(np.asarray(init["spikes"]).shape==(PRE,17099),label+" warm spikes")
        if base_init is None: base_init=init
        for k in ("current","spikes","features","history","release_features","body_qpos","body_qvel","body_act","body_ctrl"):
            require(k in init,label+" missing initialization "+k)
            equal(init[k],base_init[k],label+" matched "+k)
        for name in ("NEURAL_COLD.npz","CONNECTIVITY.npz","NEURAL_COLD_META.json","NEURAL_RELEASE_META.json","INITIAL_OBSERVATION.json"):
            require((d/name).exists(),label+" missing "+name)
        if label==ARMS[0]:
            cold=load_npz(d/"NEURAL_COLD.npz"); csr=load_npz(d/"CONNECTIVITY.npz"); meta=load_json(d/"NEURAL_COLD_META.json"); obs=load_json(d/"INITIAL_OBSERVATION.json")
            base_cold,base_csr,base_meta,base_obs=cold,csr,meta,obs
        else:
            for k,v in base_cold.items(): equal(load_npz(d/"NEURAL_COLD.npz")[k],v,label+" cold "+k)
            for k,v in base_csr.items(): equal(load_npz(d/"CONNECTIVITY.npz")[k],v,label+" csr "+k)
            require(load_json(d/"NEURAL_COLD_META.json")==base_meta,label+" cold metadata")
            require(load_json(d/"INITIAL_OBSERVATION.json")==base_obs,label+" initial observation")
        wf,wu,hist=fir_recompute(np.concatenate((init["spikes"],a["spikes"])),mapping.pools,mapping.actuator_pool,expected[label])
        equal(a["features"],wf[PRE:],label+" FIR",1e-14); equal(a["applied"],wu[PRE:],label+" commands",1e-14)
        ctrl=np.asarray(a["ctrl"]); require(ctrl.shape[0]==SAMPLES,label+" ctrl samples"); equal(ctrl[1:],np.repeat(a["applied"],5,axis=0),label+" 5-step hold",1e-14)
        dyn=check_dynamics(a,label)
        af=np.asarray(a["actuator_force"]); mom=np.asarray(a["actuator_moment"]); qfrc=np.asarray(a["qfrc_actuator"])
        proj=np.einsum("tud,tu->td",mom,af)
        equal(proj,qfrc,label+" all-muscle projection",2e-12)
        load=np.asarray(a["foot_force"])[:,:,2].sum(1)/float(r["weight_native"]); lim=np.asarray(a["limit_excess"])
        jload=float(np.trapezoid((np.maximum(.8-load,0)/.8)**2,dx=DT_BODY)/.01)
        jlimit=float(np.trapezoid((np.maximum(lim-1e-6,0)/.001)**2,dx=DT_BODY)/.01)
        require(abs(jload-float(r["Jload"]))<=1e-12,label+" Jload")
        require(abs(jlimit-float(r["Jlimit"]))<=1e-12,label+" Jlimit")
        m=metrics(a,float(r["weight_native"])); m.update(label=label,passed=True,weight_native=float(r["weight_native"]),Jload=jload,Jlimit=jlimit,**dyn,actuator_projection_max_abs=float(np.max(np.abs(proj))))
        m["early_first_differences_source"]="PREFAILURE_DIFFERENCES.json (runner-side saved comparison)"
        checks.append(m); plotted.append((label,a,m))
    return dict(passed=True,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,checks=checks,source_hashes=hashes,physical_units="NOT_ESTABLISHED",behavior_claim="NONE",decision="DESCRIPTIVE_PARETO_ONLY_NO_DEPLOYMENT",caveats=["Saved neural and physical arrays are not rerun.","Moment and dynamics checks are algebraic archive checks; units remain NOT_ESTABLISHED.","No baseline override or arm selection is performed."]),plotted

def plot(out, plotted):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(4,1,figsize=(12,12),sharex=True)
    for label,a,m in plotted:
        t=a["time"]*1000; qf=a["qfrc_actuator"][:,8]; load=np.asarray(a["foot_force"])[:,:,2].sum(1)/m.get("weight_native",1) if "weight_native" in m else np.asarray(a["foot_force"])[:,:,2].sum(1)
        ax[0].plot(t,qf,label=label+" target torque"); ax[1].plot(t,load,label=label+" load"); ax[2].plot(t,a["qpos"][:,2]/a["qpos"][0,2],label=label+" height")
        ax[3].plot(t,a["limit_excess"],label=label+" limit")
    ax[1].axhline(.8,color='black',ls=':'); ax[2].axhline(.85,color='black',ls=':'); ax[3].axhline(1e-6,color='black',ls=':')
    for x in ax: x.grid(alpha=.25); x.legend(fontsize=8)
    ax[0].set_ylabel("qfrc_actuator DOF8"); ax[1].set_ylabel("load / native weight"); ax[2].set_ylabel("height ratio"); ax[3].set_ylabel("limit excess (rad)"); ax[-1].set_xlabel("physical time (ms)")
    fig.suptitle("Local pool response: target torque, load and limit/height\nOffline saved-array verification; descriptive Pareto only")
    fig.tight_layout(); fig.savefig(Path(out)/"LOCAL_RESPONSE_COMPARISON.png",dpi=150); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,required=True); o=p.parse_args().output.resolve(); report,plotted=verify(o); plot(o,plotted); report["comparison_png"]=str(o/"LOCAL_RESPONSE_COMPARISON.png"); (o/"INDEPENDENT_CHECK.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); print(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__": main()
