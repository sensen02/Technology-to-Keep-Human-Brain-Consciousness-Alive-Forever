"""One fixed timing control: source paths, duty.9, frequency.8Hz, no physics.

Offsets shift bynewduty-oldduty, preserving global liftoff phases. Perleg source
phase mapping changes stance/swing durations only; no IK, clipping or filtering.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path.insert(0,str(BASE))
import mujoco as mj
from plant_probe import Plant
from incremental_actuator import DEFAULT_MODEL_PATH
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z/GAIT.npz')
SOURCE_SHA256='8a0d0bf2a9910963e45e40d87db6f35617270646762f9348ffae3adbfdcba739'
OUT=SOURCE.parent.parent/'20261107T122951Z'
OLD_DUTY,NEW_DUTY,FREQUENCY=5/6,.9,.8
LEGS=('LF','LM','LH','RF','RM','RH')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def periodic(phase,q):
    return PchipInterpolator(np.r_[phase[-3:-1]-2*np.pi,phase,phase[1:3]+2*np.pi],
        np.concatenate([q[-3:-1],q,q[1:3]]),axis=0,extrapolate=False)


def mapped_phase(phase,oldoffset):
    newoffset=(oldoffset+NEW_DUTY-OLD_DUTY)%1
    local=(np.asarray(phase)[...,None]/(2*np.pi)+newoffset)%1
    oldlocal=np.where(local<NEW_DUTY,local*OLD_DUTY/NEW_DUTY,
        OLD_DUTY+(local-NEW_DUTY)*(1-OLD_DUTY)/(1-NEW_DUTY))
    global_source=((oldlocal-oldoffset)%1)*2*np.pi
    return global_source,local,oldlocal,newoffset


def exact_rates(curve):
    maximum=np.zeros(42);locations=np.zeros(42)
    for segment in range(len(curve.x)-1):
        left,right=curve.x[segment:segment+2]
        if left < -1e-12 or right > 2*np.pi+1e-12:continue
        a,b,c=curve.c[:3,segment];h=right-left
        for j in range(42):
            candidates=[0.,h]
            if abs(a[j])>1e-20:
                v=-b[j]/(3*a[j])
                if 0<v<h:candidates.append(v)
            for v in candidates:
                rate=abs(3*a[j]*v*v+2*b[j]*v+c[j])*2*np.pi*FREQUENCY
                if rate>maximum[j]:maximum[j]=rate;locations[j]=left+v
    return maximum,locations


def main():
    def forbidden(*a,**kw):raise AssertionError('No mj_step in timing-only asset preparation')
    mj.mj_step=forbidden
    assert sha(SOURCE)==SOURCE_SHA256,'Accepted original.25 reference changed'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior timing candidate')
    with np.load(SOURCE) as z:G={k:z[k].copy() for k in z.files}
    meta=json.loads(str(G['metadata_json'].item()));offset=np.asarray(meta['phase_offsets'])
    assert meta['parameters']['duty']==OLD_DUTY and meta['parameters']['clearance']==.25
    files=[Path(__file__),SOURCE,BASE/'plant_probe.py',BASE/'incremental_actuator.py',DEFAULT_MODEL_PATH,
        ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py']
    hashes={str(p):sha(p) for p in files}
    dump(OUT/'PREREGISTRATION.json',dict(stage='S5 single fixed support-dwell timing geometry control',source_hashes=hashes,
        old_duty=OLD_DUTY,new_duty=NEW_DUTY,frequency_hz=FREQUENCY,old_phase_offsets=offset,
        new_phase_offsets=(offset+NEW_DUTY-OLD_DUTY)%1,
        inverse_mapping='newlocal=(newglobalcycle+newoffset)%1; oldlocal=newlocal*(5/6)/.9 innewstance,else5/6+(newlocal-.9)*(1/6)/.1;sourceglobal=(oldlocal-oldoffset)%1.',
        phase_order='newoffset=oldoffset+(.9-5/6) preserveschronologicalglobalLIFTOFFphases, notfixedoldtouchdown. No leg-ordersearch.',
        purpose='Measuredslip mostlyinsidecone, switchwindowsamplify butonly25/32%hindslipnumerator; all-sixnominalstance dwell testsloadtransitionhypothesiswithoutclaimprimarycure.',
        changes='Timingonly: duty.9/freq.8 toretainoriginal30rad/scap,513knotsdirectsourceperiodicPCHIPresampling. NoIK/filter/angleclip. Sourcefootposepath/XY/.25heightretainedwithininterpolationgates.',
        validation=dict(dense=4097,position=.005,orientation=.03,sole_median_spread=.005,
            plateau_actual_wholefoot_clearance=.1,native_bound_tolerance=1e-9,seam=1e-10,exact_rate_at_0p8Hz=30.,
            stance_count='4096uniqueposes,only5or6nominalstancelegs; allsixfractionexpected.4 (gridquantizationreported)'),
        physical='NOT_RUN: lateractualrootfree/original71mesh/gravity/contact/friction/solref/solimp/noslip0/nativecapsunchanged. Nominaldwell doesnotproveactualloadfollowstiming.',
        offsets='No+.02RHjointoffsetbaked; parentmaypreservemeasuredselectedphysicalreferenceoffsetseparately.',
        export='OnlyGAITifALLgatespass; anyresamplingratefailurestopswithoutclipping/refiltering',physics_steps=0,neural_steps=0))
    start=time.monotonic();p=Plant();m=p.m;d=mj.MjData(m);calls=0
    assert np.array_equal(p.qa,G['qpos_addresses'])
    names=G['joint_names'].astype(str);ids=[np.flatnonzero(np.char.startswith(names,'joint_'+leg)) for leg in LEGS]
    assert all(len(ix)==7 for ix in ids)
    phase=G['phase'];oldcurve=periodic(phase,G['qpos_hinges'])
    def sourceq(phases):
        mapped,local,oldlocal,newoffset=mapped_phase(phases,offset)
        q=np.empty((len(np.atleast_1d(phases)),42))
        for li,ix in enumerate(ids):q[:,ix]=oldcurve(mapped[:,li])[:,ix]
        return q,local,oldlocal,newoffset,mapped
    qknots,_,_,newoffset,knot_mapping=sourceq(phase);qknots[-1]=qknots[0]
    curve=periodic(phase,qknots);dense=np.linspace(0,2*np.pi,4097)
    desiredq,local,oldlocal,_,mapping=sourceq(dense);actualq=curve(dense)
    desiredpos=np.empty((4097,6,3));actualpos=np.empty_like(desiredpos)
    desiredrot=np.empty((4097,6,3,3));actualrot=np.empty_like(desiredrot)
    source_soles=np.empty((4097,6));actual_soles=np.empty_like(source_soles)
    plane=int(np.flatnonzero(p.planes)[0]);geoms=[np.flatnonzero(p.feet==j) for j in range(6)];ft=np.zeros(6)
    def fk(q):
        nonlocal calls
        d.qpos[:]=p.q0;d.qpos[p.qa]=q;mj.mj_forward(m,d);calls+=1
        assert d.time==0 and np.array_equal(d.qpos[:7],p.q0[:7]) and not d.warning.number.any()
        sole=np.array([min(mj.mj_geomDistance(m,d,int(g),plane,1000.,ft) for g in gg) for gg in geoms])
        return d.xpos[p.bids].copy(),d.xmat[p.bids].reshape(6,3,3).copy(),sole
    for i in range(4097):
        desiredpos[i],desiredrot[i],source_soles[i]=fk(desiredq[i])
        actualpos[i],actualrot[i],actual_soles[i]=fk(actualq[i])
    pe=np.linalg.norm(actualpos-desiredpos,axis=2);oe=np.empty((4097,6))
    for li in range(6):oe[:,li]=np.linalg.norm(Rotation.from_matrix(desiredrot[:,li].transpose(0,2,1)@actualrot[:,li]).as_rotvec(),axis=1)
    stance=local<NEW_DUTY;oldv=(oldlocal-OLD_DUTY)/(1-OLD_DUTY)
    plateau=(~stance)&(oldv>=.25-1e-10)&(oldv<=.75+1e-10)
    medians=np.array([np.median(source_soles[stance[:,li],li]) for li in range(6)])
    common=float(medians.max());actualmed=np.array([np.median(actual_soles[stance[:,li],li]) for li in range(6)])
    clear=actual_soles-common;rate,locations=exact_rates(curve)
    lo,hi=m.jnt_range[p.jids].T;excess=float(max(np.max(lo-actualq),np.max(actualq-hi),0.))
    seam=float(abs(curve(0,1)-curve(2*np.pi,1)).max());counts=stance[:-1].sum(1)
    oldlift=(OLD_DUTY-offset)%1;newlift=(NEW_DUTY-newoffset)%1
    lifterror=float(np.max(abs((oldlift-newlift+.5)%1-.5)))
    checks=dict(position=bool(pe.max()<=.005),orientation=bool(oe.max()<=.03),
        true_stance_sole_median_spread=bool(np.ptp(actualmed)<=.005),
        true_stance_sole_point_deviation=bool(abs(actual_soles[stance]-source_soles[stance]).max()<=.005),
        plateau_actual_wholefoot_clearance=bool(clear[plateau].min()>=.1),nativebounds=excess<=1e-9,
        seam=seam<=1e-10,exact_rate_0p8Hz=bool(rate.max()<=30),endpoint=bool(np.array_equal(qknots[0],qknots[-1])),
        nominal_stance_5or6=bool(np.isin(counts,[5,6]).all()),all6fraction=abs(float((counts==6).mean())-.4)<=6/4096,
        liftoff_global_phase_unchanged=lifterror<=1e-12)
    info=dict(status='FK_ACCEPTED' if all(checks.values()) else 'TIMING_RESAMPLING_FAILED',passed=all(checks.values()),checks=checks,
        max_position_error=float(pe.max()),max_orientation_error=float(oe.max()),per_leg_position_max=pe.max(0),
        per_leg_orientation_max=oe.max(0),stance_sole_medians=actualmed,stance_sole_spread=float(np.ptp(actualmed)),
        source_common_stance_plane=common,min_plateau_wholefoot_clearance=float(clear[plateau].min()),
        per_leg_plateau_min_clearance=[float(clear[plateau[:,j],j].min()) for j in range(6)],
        bounds_excess=excess,derivative_seam=seam,max_rate_rad_s=float(rate.max()),frequency_hz=FREQUENCY,rates=rate,
        worst_rate_joint=str(names[rate.argmax()]),worst_rate_globalphase=float(locations[rate.argmax()]),
        nominal_stance_counts=dict(five=int((counts==5).sum()),six=int((counts==6).sum()),all6fraction=float((counts==6).mean())),
        old_liftoff_globalcycle=oldlift,new_liftoff_globalcycle=newlift,liftoff_phase_error=lifterror,
        FK_calls=calls,physics_steps=0,neural_steps=0,wall_s=time.monotonic()-start,source_hashes_unchanged={f:sha(f)==h for f,h in hashes.items()})
    meta.update(version='broadlift-fixed-duty0p9-frequency0p8-dwell',source=str(SOURCE),source_sha256=SOURCE_SHA256,
        parameters=dict(meta['parameters'],duty=NEW_DUTY),phase_offsets=newoffset.tolist(),frequency_hz=FREQUENCY,
        max_rate_at_0p8hz=float(rate.max()),max_rate_at_1hz=float(rate.max()/FREQUENCY),
        checks=checks,fk_feasible=bool(all(checks.values())),teacher_only=True,physics_steps=0,
        timing_transformation='piecewiseoldlocaltimewarp; liftoffchronologyunchanged;sourcefullposepathpreserved',joint_offset_baked=False)
    if info['passed']:
        np.savez_compressed(OUT/'GAIT.npz',phase=phase,qpos_hinges=qknots,qpos_addresses=p.qa,joint_names=names,metadata_json=np.array(json.dumps(meta)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',phase=dense,qpos_hinges=actualq,mapped_source_qpos_hinges=desiredq,
        source_phase_per_leg=mapping,nominal_stance=stance,plateau=plateau,newlocal=local,oldlocal=oldlocal,
        source_position_world=desiredpos,achieved_world=actualpos,source_orientation_world=desiredrot,achieved_orientation_world=actualrot,
        position_error=pe,orientation_error_rad=oe,source_wholefoot_sole=source_soles,actual_wholefoot_sole=actual_soles,
        clearance_above_common_plane=clear,old_phase_offsets=offset,new_phase_offsets=newoffset)
    dump(OUT/'VERIFICATION.json',info);dump(OUT/'STATUS.json',info);dump(OUT/'METADATA.json',meta)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    axes[0].step(dense[:-1]/(2*np.pi),counts,where='post');axes[0].set(ylim=(4.8,6.2),ylabel='Nominal stance legs',title='Duty.9 / frequency.8 timing control, actual loads NOT established')
    for j,leg in enumerate(LEGS):axes[1].plot(dense/(2*np.pi),clear[:,j],label=leg)
    axes[1].axhline(.1,color='k',ls='--');axes[1].set(xlabel='New global cycle',ylabel='Actual FK wholefoot clearance from source stance plane');axes[1].legend(ncol=6)
    fig.suptitle('Source.25posepaths retained; nominal all-six fraction '+format((counts==6).mean(),'.5f')+'; exact rate '+format(rate.max(),'.5f')+'rad/s; 0physics')
    fig.savefig(OUT/'DWELL.png',dpi=150);plt.close(fig)
    print(json.dumps(info,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),flush=True)
    return 0 if info['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
