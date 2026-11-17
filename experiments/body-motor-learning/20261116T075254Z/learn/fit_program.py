"""Fit an instantaneous body controller only with pinned physical approvals."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from course import CENTER_SPEED, UNITS
from physical import BASE, HERE, bench, sha, write
from policy import LocalBank, PARAMETERS, Residual, RESIDUAL_BOUND, YAW_SCALE, base_hashes, feature_rows


def checked_data(directory):
    result_file=directory/"RESULT.json"
    native_path=directory/"NATIVE_AUDIT.json"
    if not native_path.is_file():
        native_path=directory/"NATIVE.json"
    if not native_path.is_file():
        native_path=result_file
    native_raw=directory/"NATIVE.npz"
    decisions=directory/"DECISIONS.npz"
    trace=directory/"TRACE.npz"
    result=bench.read(result_file)
    native=bench.read(native_path)
    if not any(native.get(key) is True for key in ("native_joint_pass","joint","qualified","passed")):
        raise ValueError(f"No qualifying all-native behavior at {directory}")
    hashes={k:sha(p) for k,p in dict(trace=trace,decision=decisions,raw=native_raw,audit=native_path).items()}
    if hashes["trace"]!=result.get("trace_sha256"):
        raise ValueError(f"Physical trajectory hash mismatch at {directory}")
    if hashes["decision"] != result.get("decision_sha256",result.get("decisions_sha256")):
        raise ValueError(f"Decision hash mismatch at {directory}")
    if hashes["raw"] != native.get("native_sha256",native.get("native_arrays_sha256")):
        raise ValueError(f"Raw force archive absent from the approval at {directory}")
    if "native_audit_sha256" in result and hashes["audit"]!=result["native_audit_sha256"]:
        raise ValueError(f"Native audit hash mismatch at {directory}")
    qualification=directory/"TEACHER_ACCEPTANCE.json"
    if qualification.is_file() and bench.read(qualification).get("passed") is not True:
        raise ValueError(f"Teacher approval failed at {directory}")
    steps=native.get("native_integrations",native.get("native_steps",native.get("native_samples")))
    with np.load(trace,allow_pickle=False) as a:
        horizon=len(a["time"])-1
    if steps!=horizon*80 or result.get("actual_ms",horizon)!=horizon:
        raise ValueError(f"Incomplete or inconsistent motion evidence at {directory}")
    with np.load(native_raw,allow_pickle=False) as archive:
        pose=archive["pose"]
        force=archive["forces"]
        if pose.shape!=(steps,8) or force.shape!=(steps,13) or not np.isfinite(pose).all():
            raise ValueError(f"Incomplete native contact and pose arrays at {directory}")
        if pose[:,7].any() or not np.isfinite(force).all():
            raise ValueError(f"Unexpected applied force or invalid contact data at {directory}")
    with np.load(decisions,allow_pickle=False) as a:
        kinds=("x","command","action","ms")
        x,u,y,ms=(a[k] for k in kinds)
    n=(horizon+4)//5
    if (x.shape!=(n,bench.INPUTS) or u.shape!=(n,2) or y.shape!=(n,42) or
        not all(np.isfinite(a).all() for a in (x,u,y)) or np.abs(y).max()>1 or (u[:,0]<0).any()):
        raise ValueError(f"Expected finite 213 input, [v in native/s, signed yaw in rad/s], 42 action at {directory}")
    np.testing.assert_array_equal(ms,np.arange(n)*5)
    names=[result_file,native_path,native_raw,decisions,trace]
    if qualification.is_file():
        names.append(qualification)
    return (x,u,y,ms),{str(path):sha(path) for path in names}


def fit(out, entries, anchor, iterations, seed):
    all_x=[]
    all_u=[]
    all_y=[]
    motion_data=[]
    provenance={}
    for source in (anchor,*entries):
        records, pin=checked_data(source)
        x,u,y,ms=records
        is_stock=source==anchor
        keep=u[:,0]>0
        if keep.sum() < 100:
            raise ValueError(f"Too few actual movement controls at {source}")
        if is_stock:
            if not (np.abs(u[keep,0]-CENTER_SPEED)<1e-12).all() or (u[:,1]!=0).any():
                raise ValueError("Stock anchor is not the exact straight nominal course")
        elif not np.any(keep & ((u[:,1]!=0) | (np.abs(u[:,0]-CENTER_SPEED)>1e-8))):
            raise ValueError(f"{source} has no different speed or steering action")
        all_x.append(x[keep])
        all_u.append(u[keep])
        all_y.append(y[keep])
        motion_data.append((str(source),int(keep.sum())))
        provenance.update(pin)
    x=np.concatenate(all_x)
    u=np.concatenate(all_u)
    y=np.concatenate(all_y)
    labels=np.repeat(np.arange(len(motion_data)),[m for _,m in motion_data])
    anchors=labels==0
    keys=feature_rows(x,u)
    with np.load(BASE/"PROTOTYPES.npz",allow_pickle=False) as z:
        half=z["halfspan"]
    bank=LocalBank(keys,u,y,half,anchors,labels)
    out.mkdir()
    library=out/"LIBRARY.npz"
    np.savez_compressed(library,keys=keys,commands=u,actions=y,half=half,anchors=anchors,labels=labels)
    ident=base_hashes()
    provenance.update({str(Path(__file__)):sha(__file__)})
    lib_digest=sha(library)
    write(out/"LIBRARY.json",dict(format=PARAMETERS,base=ident,archive_sha256=lib_digest,
                                  sources=provenance,moving_windows=motion_data,
                                  command_units=list(UNITS),preserve_stop=True))
    idx=np.arange(len(x))
    distance,near=bank.indices(keys,exclude=idx)
    pred=bank.predict_rows(keys,near)
    before=float(np.abs(pred-y).mean())
    if iterations<1:
        write(out/"FIT_RESULT.json",dict(format=PARAMETERS,bank=lib_digest,updates=0,
                                         leave_one_sample_out_mae=before,rows=len(x)))
        return
    z=np.concatenate((x,u/[CENTER_SPEED,YAW_SCALE]),axis=1)
    reference=torch.as_tensor(pred,dtype=torch.float32)
    output=torch.as_tensor(y,dtype=torch.float32)
    weight=torch.as_tensor(np.clip(distance/.02,0.,1.),dtype=torch.float32).view(-1,1)
    vals=torch.as_tensor(z,dtype=torch.float32)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    m=Residual(np.median(z,axis=0),np.maximum(np.std(z,axis=0),.015))
    optim=torch.optim.AdamW(m.parameters(),lr=.00065,weight_decay=.005)
    anchors_by_run=[np.flatnonzero(labels==i) for i in range(len(motion_data))]
    def take_batch():
        return np.concatenate([rng.choice(sub,size=96,replace=True) for sub in anchors_by_run])
    milestones={0,iterations//3,2*iterations//3,iterations}
    with (out/"UPDATE_LOG.jsonl").open("x") as f:
        for step in range(iterations+1):
            info=dict(update=step)
            if step:
                select=take_batch()
                delta=m(vals[select])
                acted=(reference[select]+weight[select]*delta).clamp(-1.,1.)
                err=F.smooth_l1_loss(acted,output[select],beta=.08)
                loss=err+.012*(delta/RESIDUAL_BOUND).square().mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),.8,error_if_nonfinite=True)
                optim.step()
                info.update(loss=float(loss.detach()),huber=float(err.detach()))
            if step in milestones:
                path=out/f"step_{step:04d}.pt"
                torch.save(dict(state=m.state_dict(),optimizer=optim.state_dict(),base=ident,
                                library_sha256=lib_digest,update=step),path)
                info["checkpoint_sha256"]=sha(path)
            f.write(json.dumps(info)+"\n")
            f.flush()
    m.eval()
    with torch.inference_mode():
        estimates=(reference+weight*m(vals)).clamp(-1.,1.)
        after=float((estimates-output).abs().mean())
    all_correct=all(sha(path)==digest for path,digest in provenance.items())
    if not all_correct:
        raise ValueError("Training inputs changed during the fit")
    write(out/"FIT_RESULT.json",dict(format=PARAMETERS,bank=lib_digest,updates=iterations,
               leave_one_sample_out_mae=before,training_mae_after=after,rows=len(x),
               applied_residual_bound=RESIDUAL_BOUND,all_sources_reverified=True,
               checkpoint_sha256=sha(out/f"step_{iterations:04d}.pt")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--teacher",type=Path,action="append",required=True)
    parser.add_argument("--anchor",type=Path,default=HERE/"STOCK_PHYSICS")
    parser.add_argument("--updates",type=int,default=240)
    parser.add_argument("--seed",type=int,default=3483617)
    args=parser.parse_args()
    torch.set_num_threads(1)
    entries=[source.resolve() for source in args.teacher]
    if len(set(entries))!=len(entries) or args.anchor.resolve() in entries:
        raise ValueError("Repeated source paths hide dataset weighting")
    fit(args.output.resolve(),entries,args.anchor.resolve(),args.updates,args.seed)
