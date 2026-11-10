"""Independent archived-model/rate replay with explicit finite actuator equation."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import mujoco as mj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);p=ap.parse_args().folder
    record=json.loads((p/'RESULT.json').read_text());protocol=json.loads((p/'PROTOCOL.json').read_text())
    for name,digest in record['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
    def load(name):
        with np.load(p/name) as z:return {k:z[k] for k in z.files}
    initial=load('INITIAL.npz');body=load('BODY.npz');native=load('NATIVE.npz')['rows'];commands=load('COMMANDS.npz')['rows']
    m=mj.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mj.MjData(m);assert mj.__version__=='3.9.0'
    mj.mj_setState(m,d,initial['state'],mj.mjtState.mjSTATE_INTEGRATION);mj.mj_forward(m,d)
    target=initial['target'].copy();qa=initial['qa'];va=initial['va'];lo,hi=initial['joint_bounds'].T
    thorax=m.body('Thorax').id;legs=['LF','LM','LH','RF','RM','RH'];planes=m.geom_type==mj.mjtGeom.mjGEOM_PLANE
    feet=np.array([next((i for i,l in enumerate(legs) if (m.geom(g).name or '').startswith(l) and 'Tars' in (m.geom(g).name or '')),-1) for g in range(m.ngeom)])
    errors={k:0. for k in ['time','qpos','qvel','ctrl','qacc_warmstart','com','target','native']};started=time.monotonic()
    def compare(key,a,b):
        err=float(np.abs(a-b).max()) if np.size(a) else 0.
        assert np.isfinite(err);errors[key]=max(errors[key],err)
    def state(i):
        for key in ['qpos','qvel','ctrl','qacc_warmstart']:compare(key,getattr(d,key),body[key][i])
        compare('time',np.asarray(d.time),body['time'][i]);compare('com',d.subtree_com[thorax],body['com'][i])
    state(0);saturated=0
    for k,row in enumerate(commands):
        target=np.clip(target+np.clip(row[1:43],-30,30)*.001,lo,hi);compare('target',target,row[43:85])
        for sub in range(80):
            d.ctrl[:]=np.clip(initial['kp']*(target-d.qpos[qa])-initial['kd']*d.qvel[va],-initial['limits'],initial['limits'])
            saturated+=int(np.count_nonzero(np.abs(d.ctrl)>=initial['limits']-1e-12))
            assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mj.mj_step(m,d);mj.mj_forward(m,d)
            # Independent force ledger, verified at every native step.
            foot=np.zeros(6);nonfoot=0.;signed=0.;count=0
            for i,c in enumerate(d.contact):
                a,b=int(c.geom1),int(c.geom2)
                if not(planes[a] or planes[b]):continue
                f=np.zeros(6);mj.mj_contactForce(m,d,i,f)
                z=float((c.frame.reshape(3,3).T@f[:3])[2])*(1 if planes[a] else -1);leg=feet[b if planes[a] else a]
                if leg<0:nonfoot+=max(0,z);signed+=z;count+=1
                else:foot[leg]+=z
            actual=np.r_[d.time,foot,nonfoot,signed,count,d.subtree_com[thorax,2],d.qpos[:7],np.mean(d.ctrl**2),np.max(np.abs(d.qfrc_applied[:6]))]
            compare('native',actual,native[k*80+sub])
        state(k+1)
        if (k+1)%500==0:print(json.dumps(dict(replay_s=float(d.time),wall_s=time.monotonic()-started)),flush=True)
    passed=max(errors.values())<1e-9
    out=dict(status='REPLAY_PASS' if passed else 'REPLAY_FAIL',recovered_saturated_entries=saturated,warning_counts=d.warning.number.tolist(),max_errors=errors,native_steps=len(commands)*80,wall_s=time.monotonic()-started,physical_acceptance=record['physical_gate']['passed'],neural_model_run=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (p/'VERIFICATION.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
