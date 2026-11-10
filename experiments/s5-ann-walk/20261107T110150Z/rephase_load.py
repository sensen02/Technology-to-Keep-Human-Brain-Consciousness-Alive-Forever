"""One measured-load scheduling hypothesis; each leg's accepted path is unchanged.
RH's high-utilization middle stance previously overlapped LH/RM swings. Move
contralateral hind swing to earlier RH stance, where measured leverage margins
were better. Pairwise rear-to-front alternating wave, not a biological claim.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0]=[str(BASE),str(ROOT),str(ROOT/'tools')]
import mujoco as mj
from plant_probe import Plant,sha
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T110150Z')
LEGS=('LF','LM','LH','RF','RM','RH')
ORDER=('RH','LH','RM','LM','RF','LF')
DUTY=5/6


def dump(path,value):
    if path.exists():path.read_bytes()
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')


def interp(phase,q):
    return PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],
        np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)


def exact_rates(f):
    maximum=np.zeros(f.c.shape[2])
    for k,(left,right) in enumerate(zip(f.x[:-1],f.x[1:])):
        if left<0 or right>2*np.pi:continue
        a,b,c=f.c[:3,k];h=right-left
        peak=np.maximum(abs(c),abs(3*a*h*h+2*b*h+c))
        v=np.divide(-b,3*a,out=np.full_like(a,np.nan),where=abs(a)>1e-20)
        peak=np.where((v>0)&(v<h),np.maximum(peak,abs(3*a*v*v+2*b*v+c)),peak)
        maximum=np.maximum(maximum,peak)
    return maximum*2*np.pi


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior rephasing')
    with np.load(SOURCE/'GAIT.npz') as z:
        phase=z['phase'].copy();source=z['qpos_hinges'].copy();qa=z['qpos_addresses'].copy()
        names=z['joint_names'].astype(str);meta=json.loads(str(z['metadata_json'].item()))
    old=np.asarray(meta['phase_offsets']);new=np.array([(DUTY-ORDER.index(leg)/6)%1 for leg in LEGS])
    files=[Path(__file__),SOURCE/'GAIT.npz',SOURCE/'METADATA.json',BASE/'incremental_actuator.py',BASE/'plant_probe.py']
    dump(OUT/'PREREGISTRATION.json',dict(importance='核心必需',stage='S5 one load-aware chronological phase scheduling diagnostic',
        hypothesis='RHnativecapsinmiddlestanceu.35-.66 overlapped oldLH/RMswings. MoveLHswing toearlyRHstance; quasistatic sampledRHutilization loweratearlyu.14-.32. No promisefulldynamicimprovement.',
        old_offsets=old,new_offsets=new,chronological_liftoff_order=ORDER,
        unchanged='Eachleg101114fulljointgeometry/duty5/6/clearance/1Hz/nativebounds/originalbody+forcecaps; onlyrelativelegphase changes',
        operations='periodicPCHIP rephasebyname, source513knots thenrebuildperiodicPCHIP. Dense4097footpose+truewholefootsurface comparedtoshiftedsourcepath; exportonlyifpass.',
        thresholds=dict(position_error=.005,orientation_error=.03,joint_bound_excess=1e-9,rate_1hz=30.,sole_median_spread=.005),
        source_hashes={str(p):sha(p) for p in files},physics_steps=0,neural_steps=0,
        limitation='Engineeringphasecoordination hypothesis, notrawanimalrecord/neuralsuccess. Actualmaterialslip/support mustbemeasured separately.'))
    f=interp(phase,source);q=np.empty_like(source);ids=[]
    for j,leg in enumerate(LEGS):
        ix=np.array([i for i,name in enumerate(names) if name.startswith('joint_'+leg)])
        assert len(ix)==7;ids.append(ix)
        query=(phase+2*np.pi*(new[j]-old[j]))%(2*np.pi)
        q[:,ix]=f(query)[:,ix]
    q[-1]=q[0];g=interp(phase,q)
    dense=np.linspace(0,2*np.pi,4097);joint=g(dense)
    p=Plant();m=p.m;assert np.array_equal(p.qa,qa)
    actual=mj.MjData(m);expected=mj.MjData(m);q0=m.key_qpos[0].copy()
    floor=int(np.flatnonzero(p.planes)[0]);geoms=[np.flatnonzero(p.feet==j) for j in range(6)]
    foot=np.empty((len(dense),6,3));target=np.empty_like(foot);sole=np.empty((len(dense),6));pos=np.empty_like(sole);ori=np.empty_like(sole)
    segment=np.zeros(6)
    for i,t in enumerate(dense):
        actual.qpos[:]=q0;actual.qpos[qa]=joint[i];mj.mj_forward(m,actual)
        expected.qpos[:]=q0
        for j in range(6):expected.qpos[qa[ids[j]]]=f((t+2*np.pi*(new[j]-old[j]))%(2*np.pi))[ids[j]]
        mj.mj_forward(m,expected)
        foot[i]=actual.xpos[p.bids];target[i]=expected.xpos[p.bids]
        pos[i]=np.linalg.norm(foot[i]-target[i],axis=1)
        for j,bid in enumerate(p.bids):
            R=actual.xmat[bid].reshape(3,3);R0=expected.xmat[bid].reshape(3,3)
            ori[i,j]=np.linalg.norm(Rotation.from_matrix(R0.T@R).as_rotvec())
            sole[i,j]=min(mj.mj_geomDistance(m,actual,int(geom),floor,100.,segment) for geom in geoms[j])
    stance=((dense[:,None]/(2*np.pi)+new)%1)<DUTY
    med=np.array([np.median(sole[stance[:,j],j]) for j in range(6)])
    lo=m.jnt_range[p.jids,0];hi=m.jnt_range[p.jids,1]
    excess=float(max(np.max(lo-joint),np.max(joint-hi),0.));rates=exact_rates(g)
    seam=float(abs(g(0,1)-g(2*np.pi,1)).max())
    checks=dict(position=float(pos.max())<=.005,orientation=float(ori.max())<=.03,
                bounds=excess<=1e-9,rate=float(rates.max())<=30.,sole_plane=float(np.ptp(med))<=.005,seam=seam<=1e-10)
    meta.update(version='broad-lift-paired-wave-load-hypothesis',phase_offsets=new.tolist(),offset_order=list(ORDER),
                fk_feasible=bool(all(checks.values())),checks=checks,teacher_only=True,
                source_gait=str(SOURCE/'GAIT.npz'),source_sha256=sha(SOURCE/'GAIT.npz'))
    report=dict(passed=bool(all(checks.values())),checks=checks,max_position_error=float(pos.max()),
        max_orientation_error=float(ori.max()),joint_bound_excess=excess,max_rate_at_1hz=float(rates.max()),
        sole_median_spread=float(np.ptp(med)),sole_medians=med,seam=seam,
        old_offsets=old,new_offsets=new,chronological_liftoff_order=ORDER,physics_steps=0)
    if all(checks.values()):
        np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=q,qpos_addresses=qa,joint_names=names,metadata_json=np.array(json.dumps(meta)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=joint,foot=foot,target=target,position_error=pos,orientation_error=ori,sole_distance=sole,stance=stance)
    dump(OUT/'METADATA.json',meta);dump(OUT/'VERIFICATION.json',report)
    print(json.dumps(report,default=lambda x:np.asarray(x).tolist()),flush=True)
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
