"""Resolve saved wave contact slip by nominal stance/swing, with FK only.

Scratch MjData receives saved qpos/qvel/ctrl/time, then mj_forward; NEVER mj_step.
Frozen verifier material-point velocities include its independent Jacobian check.
Reconstructed contact forces are checked against native[399::400]; unavailable
integration warmstarts are disclosed. Native phase impulses use ALL release rows.
Counterfactual integrated-target foot penetration is geometric overlap, not force.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
VERIFY=ROOT/'experiments/s5-ann-walk/20261107T073258Z/verify_walk.py'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T081009Z')
DEFAULT=OUT/'neural/teacher_gain8_wave_4s'
LEGS=('LF','LM','LH','RF','RM','RH')
DT,CDT,PDT=.0000125,.001,.005


def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def divide(a,b):
    return float(a/b) if b>0 else None


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,default=DEFAULT)
    ap.add_argument('--force-tolerance',type=float,default=1e-6)
    args=ap.parse_args(argv)
    sys.path[:0]=[str(BASE),str(ROOT/'tools')]
    import mujoco as mj
    from plant_probe import Plant,sha
    import dynamic_support_metrics as dsm
    spec=importlib.util.spec_from_file_location('frozen_walk_verifier',VERIFY)
    verified=importlib.util.module_from_spec(spec);spec.loader.exec_module(verified)
    folder=args.folder
    files=[folder/n for n in ('BODY.npz','NATIVE.npz','COMMANDS.npz','RESULT.json')]+[OUT/'GAIT.npz',OUT/'METADATA.json',VERIFY,Path(__file__)]
    before={str(f):sha(f) for f in files}
    body=load(folder/'BODY.npz');native=load(folder/'NATIVE.npz')['rows'];command=load(folder/'COMMANDS.npz')['rows']
    metadata=json.loads((OUT/'METADATA.json').read_text())
    offsets=np.asarray(metadata['phase_offsets']);duty=float(metadata['parameters']['duty'])
    assert tuple(metadata['legs'])==LEGS and len(offsets)==6
    t=body['time'];n=len(t);assert len(native)==n*400 and len(command)==n*5
    assert np.allclose(t,native[399::400,0],rtol=0,atol=1e-9)
    keep=t-PDT>=1.-1e-9
    rawkeep=native[:,0]>=1.
    # Teacher warmup ramp0.5: phase=2pi*f*(t-ramp/2), f=1Hz.
    local=((t[:,None]-.25)+offsets)%1.
    stance=local<duty
    rawlocal=((native[:,0,None]-.25)+offsets)%1.
    rawstance=rawlocal<duty
    p=Plant();m=p.m;scratch=mj.MjData(m);target_data=mj.MjData(m)
    rows=np.empty((n,20));weighted=np.empty((n,6));squared=np.empty_like(weighted);weight=np.empty_like(weighted)
    target_penetration=np.zeros((n,6));actual_penetration=np.zeros_like(target_penetration)
    target_z=np.empty((n,6));actual_z=np.empty_like(target_z)
    low=m.jnt_range[p.jids,0];high=m.jnt_range[p.jids,1]
    posttarget=np.clip(command[:,1:43]+CDT*np.clip(command[:,43:85]+command[:,85:127],-30,30),low,high)
    consistency=float(np.abs(posttarget[:-1]-command[1:,1:43]).max());assert consistency<1e-10
    ticks=np.arange(4,len(command),5)
    assert np.allclose(command[ticks,0]+CDT,t,atol=1e-9,rtol=0)
    for i in range(n):
        scratch.qpos[:]=body['qpos'][i];scratch.qvel[:]=body['qvel'][i]
        scratch.ctrl[:]=body['ctrl'][i];scratch.time=float(t[i])
        scratch.qacc_warmstart[:]=0.  # Full integration warmstart is not in BODY.
        assert not scratch.qfrc_applied.any() and not scratch.xfrc_applied.any()
        mj.mj_forward(m,scratch)
        rows[i],weighted[i],squared[i],weight[i]=verified.contact_row_and_material_velocity(mj,p,scratch)
        actual_z[i]=scratch.xpos[p.bids,2]
        target_data.qpos[:]=body['qpos'][i];target_data.qpos[p.qa]=posttarget[ticks[i]]
        target_data.time=float(t[i]);mj.mj_forward(m,target_data)
        assert np.array_equal(target_data.qpos[:7],body['qpos'][i,:7])
        target_z[i]=target_data.xpos[p.bids,2]
        for data,depth in ((scratch,actual_penetration),(target_data,target_penetration)):
            for contact in data.contact[:data.ncon]:
                ga,gb=int(contact.geom1),int(contact.geom2)
                if not (p.planes[ga] or p.planes[gb]):continue
                leg=p.feet[gb if p.planes[ga] else ga]
                if leg>=0:depth[i,leg]=max(depth[i,leg],max(0.,-float(contact.dist)))
    expected=native[399::400]
    delta=np.abs(rows-expected)
    force_error=float(delta[keep,1:9].max())
    exact_enough=force_error<=args.force_tolerance
    status='MATCH_WITHIN_TOLERANCE' if exact_enough else 'RECONSTRUCTED_FORCE_WEIGHTS_UNCERTAIN'
    native_force=np.maximum(native[:,1:7],0)
    frame_force=np.maximum(expected[:,1:7],0)
    mean_frame_force=native[:,1:7].reshape(n,400,6).mean(1)
    loaded_threshold=.01*dsm.WEIGHT
    bins=36;bin_slip=np.full((6,bins),np.nan);bin_load=np.full_like(bin_slip,np.nan)
    bin_loaded=np.full_like(bin_slip,np.nan);bin_penetration=np.full_like(bin_slip,np.nan)
    perleg=[]
    for j,leg in enumerate(LEGS):
        normal_impulse=float(native_force[rawkeep,j].sum()*DT)
        numerator=float(weighted[keep,j].sum()*PDT)
        denominator=float(weight[keep,j].sum()*PDT)
        parts={}
        for label,isstance in [('stance',True),('swing',False)]:
            sample_mask=keep & (stance[:,j]==isstance)
            native_mask=rawkeep & (rawstance[:,j]==isstance)
            ni=float(native_force[native_mask,j].sum()*DT)
            wi=float(weight[sample_mask,j].sum()*PDT)
            vi=float(weighted[sample_mask,j].sum()*PDT)
            vi2=float(squared[sample_mask,j].sum()*PDT)
            speed=np.divide(weighted[sample_mask,j],weight[sample_mask,j],out=np.zeros(sample_mask.sum()),where=weight[sample_mask,j]>0)
            recorded_weight=frame_force[sample_mask,j]
            alternate_mean=divide(float((speed*recorded_weight).sum()),float(recorded_weight.sum()))
            parts[label]=dict(sampled_frames=int(sample_mask.sum()),native_samples=int(native_mask.sum()),
                native_seconds=float(native_mask.sum()*DT),
                reconstructed_normal_weighted_mean_material_speed=divide(vi,wi),
                reconstructed_normal_weighted_rms_material_speed=float(np.sqrt(vi2/wi)) if wi>0 else None,
                normal_weighted_slip_numerator_native=vi,
                slip_numerator_fraction_of_leg=divide(vi,numerator),
                native_normal_impulse=ni,native_impulse_fraction_of_leg=divide(ni,normal_impulse),
                native_loaded_fraction_above_001W=float((native_force[native_mask,j]>loaded_threshold).mean()),
                sampled_frame_mean_loaded_fraction_above_001W=float((mean_frame_force[sample_mask,j]>loaded_threshold).mean()),
                alternate_recorded_native_weights_mean_material_speed=alternate_mean,
                recorded_loaded_but_reconstructed_weight_zero_frames=int(np.sum((recorded_weight>loaded_threshold)&(weight[sample_mask,j]<=0))),
                target_FK_ground_penetration_max=float(target_penetration[sample_mask,j].max()),
                target_FK_ground_penetration_fraction=float((target_penetration[sample_mask,j]>1e-9).mean()),
                actual_FK_ground_penetration_max=float(actual_penetration[sample_mask,j].max()),
                target_origin_world_z_max=float(target_z[sample_mask,j].max()),
                actual_origin_world_z_max=float(actual_z[sample_mask,j].max()))
        perleg.append(dict(leg=leg,whole_release_mean_material_speed=divide(numerator,denominator),
            reconstructed_impulse=denominator,native_impulse=normal_impulse,
            reconstructed_weighted_slip_numerator=numerator,
            stance=parts['stance'],swing=parts['swing']))
        for b in range(bins):
            sm=keep & (np.floor(local[:,j]*bins).astype(int)==b)
            nm=rawkeep & (np.floor(rawlocal[:,j]*bins).astype(int)==b)
            if sm.any():
                wsum=weight[sm,j].sum()
                if wsum>0:bin_slip[j,b]=weighted[sm,j].sum()/wsum
                bin_penetration[j,b]=target_penetration[sm,j].max()
            if nm.any():
                bin_load[j,b]=native_force[nm,j].mean()/dsm.WEIGHT
                bin_loaded[j,b]=(native_force[nm,j]>loaded_threshold).mean()
    aggregate={}
    for label in ('stance','swing'):
        numerator=sum(x[label]['normal_weighted_slip_numerator_native'] for x in perleg)
        impulse=sum(x[label]['native_normal_impulse'] for x in perleg)
        reconstructed=sum(float(weight[keep & (stance[:,j]==(label=='stance')),j].sum()*PDT) for j in range(6))
        aggregate[label]=dict(normal_weighted_mean_material_speed=divide(numerator,reconstructed),
            slip_numerator=numerator,native_normal_impulse=impulse)
    total_num=sum(x['slip_numerator'] for x in aggregate.values());total_imp=sum(x['native_normal_impulse'] for x in aggregate.values())
    for x in aggregate.values():
        x['fraction_of_normal_weighted_slip_numerator']=divide(x['slip_numerator'],total_num)
        x['fraction_of_native_normal_impulse']=divide(x['native_normal_impulse'],total_imp)
    yaw,_=verified.attitude(body['qpos'][:,3:7]);positions=body['com'][keep,:2];dyaw=yaw[keep]
    increments=np.diff(positions,axis=0);heading=(dyaw[1:]+dyaw[:-1])/2
    forward=float(np.sum(increments[:,0]*np.cos(heading)+increments[:,1]*np.sin(heading)))
    comvel=np.gradient(body['com'][:,:2],t,axis=0)
    after={str(f):sha(f) for f in files};assert before==after
    outputs=[OUT/'SLIP_PHASE.json',OUT/'SLIP_PHASE.png']
    previous={str(f):sha(f) for f in outputs if f.exists()}  # read before overwrite
    clean=lambda values:[[float(x) if np.isfinite(x) else None for x in row] for row in values]
    report=dict(stage='S5 phase-resolved saved contact-slip diagnostic, FK only',folder=str(folder),
        force_reconstruction_status=status,force_weights_within_tolerance=exact_enough,
        comparison=dict(tolerance=args.force_tolerance,release_native_row_max=float(delta[keep].max()),
            release_contact_force_max=force_error,release_contact_force_rms=float(np.sqrt(np.mean(delta[keep,1:9]**2))),
            allframes_native_row_max=float(delta.max()),per_column_max=delta[keep].max(axis=0).tolist(),
            force_per_leg_max=delta[keep,1:7].max(axis=0).tolist(),
            contact_count_mismatches=int(np.sum(rows[keep,9]!=expected[keep,9]))),
        phase=dict(formula='global_cycle=(BODY.time-0.25)*1Hz; local=(global_cycle+metadata.phase_offsets)%1',
            handover_s=1.,ramp_s=.5,frequency_hz=1.,duty=duty,offsets=offsets.tolist(),
            release_seconds=float(keep.sum()*PDT),sample_dt_s=PDT,native_dt_s=DT,
            note='Nominal teacher phase; no fitted timing shift. Command hold/integration can shift actual target motion around nominal boundaries.'),
        per_leg=perleg,aggregate=aggregate,
        body=dict(signed_forward_travel=forward,forward_speed=forward/(t[keep][-1]-t[keep][0]),
            COM_horizontal_speed_rms=float(np.sqrt(np.mean(np.sum(comvel[keep]**2,axis=1)))),
            heading_change_rad=float(dyaw[-1]-dyaw[0])),
        phase_bins=dict(count=bins,slip_material_speed=clean(bin_slip),native_force_weight=clean(bin_load),
            native_loaded_fraction=clean(bin_loaded),target_FK_penetration_max=clean(bin_penetration)),
        counts=dict(saved_body_frames=n,release_body_frames=int(keep.sum()),native_release_samples=int(rawkeep.sum()),
            FK_forward=2*n,physics_steps=0,neural_steps=0),
        source_hashes=before,source_hashes_unchanged=True,previous_outputs=previous,
        limitations=['Scratch reconstruction lacks saved integration warmstart; force agreement is measured, not assumed. Weighted slip must be treated as uncertain when tolerance fails.',
            'Recorded qpos/qvel/ctrl/time are assigned only to scratch data for mj_forward. No physics integration is run.',
            'Material point velocity uses frozen BODY twist plus contact offset and independent point Jacobian check. It is sampled every5ms, not continuous.',
            'Native normal impulse/loaded fractions use every native release sample and nominal phase, independent of reconstructed force-weight uncertainty.',
            'Normal-weighted slip numerator is a diagnostic proxy, not frictional work: tangential contact force is not used.',
            'Stance versus swing breakdown identifies when slip occurs; it does not isolate a unique root/body/joint cause.',
            'Counterfactual target tarsal penetration is geometric overlap at actual root pose. Foot-origin heights are proxies, not sole clearance.',
            'These are algorithm-teacher diagnostics; no neural or biological locomotion claim.'])
    outputs[0].write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,12),constrained_layout=True)
    x=np.arange(6)
    for offset,label,color in [(-.18,'stance','C0'),(.18,'swing','C1')]:
        values=[v[label]['reconstructed_normal_weighted_mean_material_speed'] or 0 for v in perleg]
        axes[0,0].bar(x+offset,values,.35,label=label,color=color)
    axes[0,0].set(xticks=x,xticklabels=LEGS,ylabel='Material point tangential speed',title='Reconstructed normal-weighted means')
    axes[0,0].axhline(report['body']['forward_speed'],color='k',ls=':',label='signed body forward speed');axes[0,0].legend()
    axes[0,1].bar(x,[v['swing']['native_impulse_fraction_of_leg'] for v in perleg],color='C1')
    axes[0,1].set(xticks=x,xticklabels=LEGS,ylabel='Fraction of native normal impulse',title='Impulse during nominal swing')
    for ax,data,title in [(axes[1,0],bin_slip,'Material slip speed by local phase'),
        (axes[1,1],bin_load,'Native mean foot load / weight'),
        (axes[2,0],bin_penetration,'Target FK ground penetration at actual root'),
        (axes[2,1],bin_loaded,'Native loaded fraction (>0.01 weight)')]:
        image=ax.imshow(data,aspect='auto',origin='upper',extent=[0,1,5.5,-.5],interpolation='nearest')
        ax.axvline(duty,color='r',ls='--');ax.set(yticks=x,yticklabels=LEGS,xlabel='Local phase: stance0..5/6, swing5/6..1',title=title)
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle('Saved wave contact slip by nominal phase; '+status+'\nNo new physics; material velocities sampled5ms, native impulses use all samples')
    fig.savefig(outputs[1],dpi=150);plt.close(fig)
    print(json.dumps(dict(force_status=status,force_error=force_error,aggregate=aggregate,
        legs=[dict(leg=v['leg'],total=v['whole_release_mean_material_speed'],stance=v['stance']['reconstructed_normal_weighted_mean_material_speed'],
            swing=v['swing']['reconstructed_normal_weighted_mean_material_speed'],swing_impulse_fraction=v['swing']['native_impulse_fraction_of_leg'],
            swing_loaded_fraction=v['swing']['native_loaded_fraction_above_001W'],swing_target_penetration=v['swing']['target_FK_ground_penetration_max']) for v in perleg],physics_steps=0)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
