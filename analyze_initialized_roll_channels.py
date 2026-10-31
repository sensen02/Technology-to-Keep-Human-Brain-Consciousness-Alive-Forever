#!/usr/bin/env python3
"""S116: read-only saved-array roll analysis. No simulator/neural modules imported.
All forces/torques are native model units, NOT_ESTABLISHED physically.
Candidate screens are observational options, not causal estimates or W edits.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/home/sensen/Desktop/engine修复验收')
REFS = [BASE/'s113-com-timestep/20261024T060500Z/dt25us',
        BASE/'s115-extended-initialization/20261024T062500Z/init80ms']
LABELS = ['S113_40ms', 'S115_80ms']
BINS = ((0,5),(5,10),(10,20),(20,40),(40,80))
FEET = ('lf','lm','lh','rf','rm','rh')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def loadj(p):
    return json.loads(Path(p).read_text())

def savej(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def csvout(p, rows):
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True,help='must not exist')
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    hashes={}; checked={}
    def bind(p, expected=None):
        p=Path(p); h=sha(p)
        if expected is not None and h!=expected: raise ValueError(f'hash mismatch: {p}')
        hashes[str(p)]=h
        if expected is not None: checked[str(p)]=h
        return p
    bind(Path(__file__).resolve())
    zs=[]; ins=[]; metas=[]; preregs=[]; manifests=[]
    # Validate original source commitments and every RESULT artifact before analysis.
    for ref in REFS:
        pre=loadj(bind(ref.parent/'PREREG.json')); preregs.append(pre)
        for p,h in pre['source_hashes'].items(): bind(p,h)
        bind(ref.parent/'INPUT_HASHES.json')
        r=loadj(bind(ref/'RESULT.json'))
        for p,h in r['hashes'].items(): bind(p,h)
        bind(ref/'TRAJECTORY.npz',r['trajectory_sha256'])
        bind(ref/'INITIALIZATION.npz',r['initialization_sha256'])
        m=loadj(bind(ref/'MOMENT_METADATA.json')); metas.append(m)
        candidates=[p for p in pre['source_hashes'] if p.endswith('/MANIFEST.json') and 's4-endpoints' in p]
        assert len(candidates)==1
        manifests.append(loadj(bind(candidates[0],pre['source_hashes'][candidates[0]])))
        zs.append(dict(np.load(ref/'TRAJECTORY.npz',allow_pickle=False)))
        ins.append(dict(np.load(ref/'INITIALIZATION.npz',allow_pickle=False)))
    assert manifests[0]==manifests[1]
    names=metas[0]['actuator_names']; assert names==metas[1]['actuator_names'] and len(names)==90
    for m in metas:
        j=m['joint_names'].index('joint_LFCoxa_roll')
        assert m['jnt_qposadr'][j]==9 and m['jnt_dofadr'][j]==8
        assert m['jnt_range'][j][1]==.6236 and m['jnt_limited'][j]
    t=zs[0]['time']; close(t,zs[1]['time']); close(t,np.arange(121)*.000025)
    assert len(t)==121
    W=zs[0]['weights']; assert np.array_equal(W,zs[1]['weights'])
    pool=zs[0]['actuator_pool']; assert np.array_equal(pool,zs[1]['actuator_pool'])
    groups=[tuple(sorted(r['local_indices'])) for r in manifests[0]['rows']]
    pools=sorted(set(g for g in groups if g)); lookup={g:i for i,g in enumerate(pools)}
    assert np.array_equal(pool,[lookup.get(g,-1) for g in groups])
    mapped=pool>=0
    relevance=np.any(np.stack([np.abs(z['actuator_moment'][:,:,8])>1e-12 for z in zs]),axis=(0,1))
    commandrows=[]; poolrows=[]; recon=[]; features=[]; rawp=[]; rawacts=[]; clips=[]
    warmchecks=[]
    for label,z,ini in zip(LABELS,zs,ins):
        assert np.array_equal(W,ini['weights'])
        # Reconstruct entire warmup and physical FIR from saved binary spikes.
        history=np.zeros((80,len(pools))); calc=[]
        allsp=np.concatenate((ini['spikes'],z['spikes']))
        for sp in allsp:
            activity=np.array([sp[list(g)].mean()/.0005/100 for g in pools])
            history=np.concatenate((activity[None],history))[:80]
            calc.append(np.stack([history[lo:hi].mean(0) for lo,hi in BINS],axis=1))
        n=len(ini['spikes']); calc=np.array(calc)
        close(calc[:n],ini['features']); close(calc[n:],z['features'])
        close(calc[n-1],ini['release_features'])
        warmraw=(calc[:n]*W).sum(2); warmact=np.zeros((n,90)); warmact[:,mapped]=np.clip(warmraw[:,pool[mapped]],0,1)
        close(warmact,ini['decoded'])
        feat=np.concatenate((ini['release_features'][None],z['features'])); features.append(feat)
        rp=(feat*W).sum(2); rawp.append(rp)
        ra=np.zeros((7,90)); ra[:,mapped]=rp[:,pool[mapped]]; rawacts.append(ra)
        cl=np.clip(ra,0,1); clips.append(cl)
        close(cl[0],ini['release_decoded']); close(cl[1:],z['decoded']); close(cl[1:],z['applied'])
        close(z['ctrl'][0],np.zeros(90))
        for i in range(1,121): close(z['ctrl'][i],cl[1+(i-1)//20])
        warmchecks.append(dict(label=label,warm_ticks=n,all_warm_features_and_decoded_verified=True,release_is_not_applied_to_body=True))
        for stage in range(7):
            for a in range(90):
                p=int(pool[a]); commandrows.append(dict(arm=label,stage='release' if stage==0 else f'tick{stage-1}',physical_start_ms=0 if stage==0 else (stage-1)*.5,actuator=a,name=names[a],pool=p,relevant=bool(relevance[a]),raw_unclipped=float(ra[stage,a]),clipped=float(cl[stage,a]),applied=float(cl[stage,a]) if stage else 0.,saturated=bool(ra[stage,a]>1),pool_identity=json.dumps(pools[p]) if p>=0 else 'unmapped'))
        recon.append(dict(label=label,warmup=warmchecks[-1],saturated=[dict(stage='release' if s==0 else f'tick{s-1}',pool=int(p),raw=float(rp[s,p]),identity=list(pools[p]),actuators=[names[a] for a in np.flatnonzero(pool==p)]) for s,p in zip(*np.where(rp>1))]))
    for s in range(7):
        for p in range(len(pools)):
            ds=(features[1][s,p]-features[0][s,p])*W[p]
            close(ds.sum(),rawp[1][s,p]-rawp[0][s,p])
            for b,(lo,hi) in enumerate(BINS):
                poolrows.append(dict(stage='release' if s==0 else f'tick{s-1}',pool=p,bin=b,bin_lo_tick=lo,bin_hi_tick=hi,weight=float(W[p,b]),feature0=float(features[0][s,p,b]),feature1=float(features[1][s,p,b]),weighted_feature_delta=float(ds[b]),raw_pool_delta=float(ds.sum()),effective_pool_delta=float(np.clip(rawp[1][s,p],0,1)-np.clip(rawp[0][s,p],0,1)),shared_actuators=json.dumps([names[a] for a in np.flatnonzero(pool==p)]),identity=json.dumps(pools[p])))
    F0,F1=[z['actuator_force'] for z in zs]; M0,M1=[z['actuator_moment'][:,:,8] for z in zs]
    tau0,tau1=F0*M0,F1*M1; df=F1-F0; dm=M1-M0
    force=df*M0; moment=F0*dm; cross=df*dm; delta=tau1-tau0
    close(force+moment+cross,delta)
    for z,tau in zip(zs,[tau0,tau1]): close(tau.sum(1),z['qfrc_actuator'][:,8])
    torque=[]; timeline=[]; events=[]
    for ai,(label,z,tau) in enumerate(zip(LABELS,zs,[tau0,tau1])):
        for i in range(121):
            row=dict(arm=label,state=i,time_ms=float(t[i]*1000),q=float(z['qpos'][i,9]),v=float(z['qvel'][i,8]),upper=.6236,upper_excess=float(max(z['qpos'][i,9]-.6236,0)),load_ratio=float(z['load_ratio'][i]),global_limit_excess=float(z['limit_excess'][i]))
            for k in ['qacc','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','qfrc_applied','Ma','RHS','residual']: row[k]=float(z[k][i,8])
            row['signed_tau90_sum']=float(tau[i].sum())
            for k,foot in enumerate(FEET): row[f'{foot}_Fz']=float(z['foot_force'][i,k,2])
            timeline.append(row)
            for a in range(90): torque.append(dict(arm=label,state=i,time_ms=float(t[i]*1000),actuator=a,name=names[a],pool=int(pool[a]),relevant=bool(relevance[a]),force=float(z['actuator_force'][i,a]),moment=float(z['actuator_moment'][i,a,8]),signed_torque=float(tau[i,a]),delta_force_M0=float(force[i,a]),F0_delta_moment=float(moment[i,a]),delta_force_delta_moment=float(cross[i,a]),exact_delta_torque=float(delta[i,a])))
        def first(mask):
            idx=np.flatnonzero(mask); return int(idx[0]) if len(idx) else None
        ev=dict(arm=label,first_load_fail_state=first(z['load_ratio']<.8),first_limit_fail_state=first(z['limit_excess']>1e-6),peak_limit_state=int(np.argmax(z['limit_excess'])),peak_limit_excess=float(np.max(z['limit_excess'])))
        for k in ['first_load_fail','first_limit_fail','peak_limit']: ev[k+'_ms']=None if ev[k+'_state'] is None else float(t[ev[k+'_state']]*1000)
        events.append(ev)
    aggregate=[dict(state=i,time_ms=float(t[i]*1000),delta_tau=float(delta[i].sum()),delta_force_M0=float(force[i].sum()),F0_delta_moment=float(moment[i].sum()),cross=float(cross[i].sum())) for i in range(121)]
    # No linear allocation of clipped commands to bins: only raw drive is additive.
    # Screen each SINGLE W[p,b] against late observed command AND actual torque.
    late=np.flatnonzero((t>=.0025-1e-12)&(t<=.00265+1e-12))
    candidates=[]; screening=[]
    for p in range(len(pools)):
        acts=np.flatnonzero((pool==p)&relevance)
        if not len(acts): continue
        for b in range(5):
            stage=5 # tick4 held through state100; tick5 at state101 onwards
            stages=np.array([1+(i-1)//20 for i in late])
            weighted=(features[1][stages,p,b]-features[0][stages,p,b])*W[p,b]
            effective=np.clip(rawp[1][stages,p],0,1)-np.clip(rawp[0][stages,p],0,1)
            fterm=force[np.ix_(late,acts)].sum(1); mterm=moment[np.ix_(late,acts)].sum(1); xterm=cross[np.ix_(late,acts)].sum(1); dterm=delta[np.ix_(late,acts)].sum(1)
            tests=dict(nonzero_weight=bool(W[p,b]>0),positive_bin_raw_increment=bool(np.all(weighted>0)),new_effective_drive_nonnegative_with_positive_late=bool(np.all(effective>=-1e-12) and np.any(effective>1e-12)),actual_delta_torque_toward_upper=bool(np.all(dterm>0)),force_change_dominant=bool(np.sum(np.abs(fterm))>np.sum(np.abs(mterm))+np.sum(np.abs(xterm))),force_term_toward_upper=bool(np.all(fterm>0)))
            entry=dict(pool=p,bin=b,weight=float(W[p,b]),tests=tests,eligible=all(tests.values()),identity=list(pools[p]),all_shared_actuators=[dict(index=int(a),name=names[a],relevant=bool(relevance[a])) for a in np.flatnonzero(pool==p)],target_actuators=[names[a] for a in acts],late_states=late.tolist(),weighted_raw_delta=weighted.tolist(),effective_command_delta=effective.tolist(),actual_delta_torque=dterm.tolist(),force_term=fterm.tolist(),moment_term=mterm.tolist(),cross_term=xterm.tolist(),saturated_new_states=[int(i) for i,s in zip(late,stages) if rawp[1][s,p]>1],interpretation='single coefficient observational option only; no magnitude or causal response inferred; shared-map coupling retained')
            entry['frozen_feature_sensitivity']=[dict(stage='release' if s==0 else f'tick{s-1}',old_feature=float(features[0][s,p,b]),new_feature=float(features[1][s,p,b]),old_raw=float(rawp[0][s,p]),new_raw=float(rawp[1][s,p]),old_clip=float(np.clip(rawp[0][s,p],0,1)),new_clip=float(np.clip(rawp[1][s,p],0,1)),raw_bin_deltas=((features[1][s,p]-features[0][s,p])*W[p]).tolist()) for s in range(7)]
            entry['coupled_dof_actual_torque_at_2p5ms']=[dict(dof=d,old=float(sum(F0[100,a]*zs[0]['actuator_moment'][100,a,d] for a in acts)),new=float(sum(F1[100,a]*zs[1]['actuator_moment'][100,a,d] for a in acts))) for d in range(48) if any(abs(z['actuator_moment'][100,a,d])>1e-12 for z in zs for a in acts)]
            entry['magnitude_recommendation']=None
            screening.append(entry)
            if entry['eligible']: candidates.append(entry)
    snapshots=[]
    for ms in [0,1.7,1.925,2.5,2.625,2.65,2.825,3.0]:
        i=int(np.argmin(abs(t-ms/1000)))
        snapshots.append(dict(time_ms=ms,state=i,arms=[next(r for r in timeline if r['arm']==l and r['state']==i) for l in LABELS],decomposition=aggregate[i],per_actuator=[dict(index=int(a),name=names[a],tau0=float(tau0[i,a]),tau1=float(tau1[i,a]),force_term=float(force[i,a]),moment_term=float(moment[i,a]),cross=float(cross[i,a])) for a in np.flatnonzero(relevance)]))
    summary=dict(scope='P1/S2-S4 offline observational roll-channel diagnosis',physical_units='NOT_ESTABLISHED',calls=dict(model_load=0,forward=0,step=0,neural=0),behavior_claim='NONE',weights_edited=False,target=dict(joint='joint_LFCoxa_roll',qpos=9,dof=8,upper=.6236),foot_order=list(FEET),samples=121,time_window_ms=[0,3],relevant_actuators=[dict(index=int(a),name=names[a],pool=int(pool[a])) for a in np.flatnonzero(relevance)],events=events,reconstruction=recon,snapshots=snapshots,candidate_options=candidates,no_candidate=not bool(candidates),candidate_screen=screening,screen_rule='all late 2.5..2.65ms states: positive W*feature delta, nonnegative effective pool drive delta with at least one positive late state (zero-net states explicitly retained), positive actual delta torque and force term; sum(abs(force term)) > sum(abs(moment term))+sum(abs(cross)); EACH option is ONE W[pool,bin], no direction/magnitude recommended',limits=['Difference algebra is exact but not a causal command-to-force estimate.','Release decoded is diagnostic, not applied: body t=0 ctrl/act remain zero; tick0 output is first applied.','Only raw bin contributions are additive. Clipped applied command increments are not linearly allocated to bins.','Initialization improves early loading; whole-window new drive is NOT uniformly harmful.','Actual force differences include activation, length, velocity and state; moment term alone is not full state dependence.','Shared actuator maps couple other joints. Candidate option is not deployment or permission to edit W.'],algebra='delta(F*M)=deltaF*M0+F0*deltaM+deltaF*deltaM',max_decomposition_error=float(np.max(abs(force+moment+cross-delta))),max_aggregate_error=float(np.max(abs(delta.sum(1)-(zs[1]['qfrc_actuator'][:,8]-zs[0]['qfrc_actuator'][:,8])))),commitments_checked=len(checked))
    assert not any(k=='mujoco' or k.startswith('engine') for k in sys.modules)
    args.output.mkdir(parents=True)
    savej(args.output/'SOURCE_HASHES.json',dict(source_hashes=hashes,validated_commitments=checked))
    csvout(args.output/'ROLL_TIMELINE.csv',timeline); csvout(args.output/'PER_STATE_PER_ACTUATOR_TORQUE.csv',torque)
    csvout(args.output/'POOL_BIN_COMMAND_DELTA.csv',poolrows); csvout(args.output/'RAW_CLIP90_COMMANDS.csv',commandrows)
    csvout(args.output/'AGGREGATE_TORQUE_DECOMPOSITION.csv',aggregate)
    savej(args.output/'SUMMARY.json',summary)
    fig,ax=plt.subplots(5,1,figsize=(13,16),sharex=True)
    for label,z,tau in zip(LABELS,zs,[tau0,tau1]):
        ax[0].plot(t*1000,z['qpos'][:,9]-.6236,label=label)
        ax[1].plot(t*1000,z['load_ratio'],label=label)
        ax[2].plot(t*1000,tau.sum(1),label=label)
        for a in np.flatnonzero(relevance): ax[3].plot(t*1000,tau[:,a],label=f'{label} act{a}',alpha=.65)
    for value,label in [(delta.sum(1),'actual delta torque'),(force.sum(1),'deltaF*M0'),(moment.sum(1),'F0*deltaM'),(cross.sum(1),'cross')]: ax[4].plot(t*1000,value,label=label)
    ax[0].axhline(1e-6,color='k',ls=':'); ax[1].axhline(.8,color='k',ls=':')
    for a,title in zip(ax,['LFCoxa_roll q - upper (rad)','Actual load ratio','Actual actuator torque DOF8 (native)','Relevant signed actuator torques (native)','Exact three-term torque difference (native)']):
        a.set_ylabel(title); a.grid(alpha=.25); a.legend(fontsize=7,ncol=3)
        for ms in [1.7,1.925,2.5,2.625,2.65,2.825]: a.axvline(ms,alpha=.16,color='k')
    ax[-1].set_xlabel('Physical time (ms); native units NOT_ESTABLISHED')
    fig.suptitle('Saved arrays only: early loading improves, late roll safety regresses; not all new drive is harmful')
    fig.tight_layout(); fig.savefig(args.output/'DRIVE_TORQUE_LIMIT.png',dpi=150); plt.close(fig)
    savej(args.output/'ARTIFACT_HASHES.json',{str(p):sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps(dict(output=str(args.output),events=events,candidate_options=[dict(pool=e['pool'],bin=e['bin']) for e in candidates],calls=summary['calls']),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
