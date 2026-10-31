"""S115/P1/S2-S4 saved-array-only 80ms initialization check.
No runner, simulator, neural engine, training, or physical framework imports.
Reuses S113 raw COM/physical/measure algebra and frozen native criteria.
"""
from pathlib import Path
import argparse
import ast
import csv
import json
import sys
import traceback
import numpy as np
from verify_com_contact_probe import Checks, require, sha, js, npz, serial, write_json, reconstruct_fir
from verify_com_timestep_probe import canonical_checks, reconstruct, physical
from verify_initial_height_probe import raw_feet, balance, differences, LEGS

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us')
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z')


def same(checks, label, a, b):
    checks.add(label+' keys', set(a)==set(b))
    for key in set(a)&set(b): checks.eq(label+' '+key,a[key],b[key])


def source_audit(path, checks):
    tree=ast.parse(path.read_text()); writes=[]; resets=[]
    def qpos(n):
        return isinstance(n,ast.Attribute) and n.attr=='qpos' or isinstance(n,ast.Subscript) and qpos(n.value)
    for n in ast.walk(tree):
        targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
        for t in targets:
            if qpos(t): writes.append(dict(line=n.lineno,source=ast.unparse(n)))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','mj_resetData','mj_resetDataKeyframe'):
            resets.append(dict(line=n.lineno,source=ast.unparse(n)))
    checks.add('hash-bound runner sole initial qpos assignment',len(writes)==1,assignments=writes)
    checks.add('hash-bound runner no explicit reset calls',not resets,calls=resets)
    return dict(qpos_writes=writes,resets=resets,boundary='static hash-bound source evidence, not proof of unrecorded runtime events')


def decision(old,new,effective):
    regress=[]
    for key in ('angle_rad','nonfoot_depth','limit_excess'):
        if new['extrema'][key]['max']>old['extrema'][key]['max']: regress.append(key)
    for key in ('max_downward_root_velocity','max_downward_COM_velocity'):
        if new[key]>old[key]: regress.append(key)
    if new['extrema']['height_ratio']['min']<old['extrema']['height_ratio']['min']: regress.append('height')
    for key,previous in old['first_failures_ms'].items():
        now=new['first_failures_ms'][key]
        if now is not None and (previous is None or now<previous): regress.append('earlier_'+key)
    load=(new['load_deficit_integral0_1p7_ratio_ms']<old['load_deficit_integral0_1p7_ratio_ms'] and new['initial_foot_Fz']>=old['initial_foot_Fz'])
    support=(new['all_gates_support_ms']>old['all_gates_support_ms'] and new['longest_all_gates_true_endpoint_span_ms']>=old['longest_all_gates_true_endpoint_span_ms'])
    candidate=bool(effective and load and support and not regress)
    return dict(effective_output_difference=bool(effective),load_improvement=bool(load),support_improvement=bool(support),safety_regressions=regress,safety_nonregression=not regress,suggest_candidate=candidate,reject=not candidate,retention=False,checkpoint_replacement=False,deployment=False,training_acceptance=False,standing_or_walking_claim=False,rule='candidate only if actual applied output differs AND load deficit decreases without lower initial load AND support improves AND all safety nonregression; otherwise no retention; 3ms is diagnostic, not training acceptance')


def verify(out):
    require(not any(x in sys.modules for x in ('mujoco','torch')),'physics/neural loaded')
    checks=Checks(); observed={}; pre=js(out/'PREREG.json'); sources=pre['source_hashes']
    canonical_checks(checks,sources,'source',observed)
    canonical={str(Path(p).resolve()):d for p,d in sources.items()}
    runner=ROOT/'run_extended_initialization_probe.py'
    checks.add('runner committed',canonical.get(str(runner.resolve()))==sha(runner))
    audit=source_audit(runner,checks)
    checks.add('INPUT_HASHES exact',js(out/'INPUT_HASHES.json')==sources)
    deps=js(out/'MODEL_DEPENDENCIES.json'); canonical_checks(checks,deps,'model dependencies',observed)
    checks.add('dependencies committed',all(canonical.get(str(Path(p).resolve()))==d for p,d in deps.items()))
    checks.add('complete without failure',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    require(len(pre['arms'])==1,'exactly one arm'); arm=pre['arms'][0]
    checks.add('exact init80ms arm label',arm=='init80ms')
    checks.add('80ms actual warm160 ticks',pre['initialization_ticks']==160 and pre['neural_dt_ms']==.5 and pre['initialization_ms']==80)
    checks.eq('continuous source offset80ms',pre['source_offset_s'],.08,1e-12)
    checks.add('3ms six neural ticks',pre['physical_duration_ms']==3 and pre['physical_neural_ticks_per_arm']==6)
    checks.add('25us body dt',pre['timesteps_s']=={arm:.000025} and pre['substeps']=={arm:20})
    checks.add('121 states120steps',pre['physical_samples_per_arm']=={arm:121} and pre['physical_steps_per_arm']=={arm:120})
    ledger=js(out/'CALL_LEDGER.json'); count=dict(model_load=1,forward=121,step=120,neural_ticks=166)
    checks.add('single arm exact ledger',ledger==dict(count,active=None,completed=[arm],per_arm={arm:count}),observed=ledger)
    results=js(out/'RESULTS.json'); require(len(results)==1,'one result'); res=js(out/arm/'RESULT.json')
    checks.add('result agreement',results==[res] and res['label']==arm)
    oldres=js(REFERENCE/'RESULT.json')
    canonical_checks(checks,oldres['hashes'],'reference artifacts',observed)
    canonical_checks(checks,res['hashes'],'new artifacts',observed)
    for name in ('TRAJECTORY.npz','INITIALIZATION.npz','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json'):
        checks.add('reference committed '+name,canonical.get(str((REFERENCE/name).resolve()))==sha(REFERENCE/name))
    if (out/'REFERENCE_ARTIFACTS.json').exists():
        canonical_checks(checks,js(out/'REFERENCE_ARTIFACTS.json')['artifacts'],'retained reference',observed)
    xmls=[Path(p) for p in sources if Path(p).suffix=='.xml']; require(len(xmls)==1,'one XML')
    manifests=[Path(p) for p in sources if Path(p).name=='MANIFEST.json']; require(len(manifests)==1,'one mapping manifest')
    rows=sorted(js(manifests[0])['rows'],key=lambda r:r['actuator_index'])
    pools=sorted({tuple(sorted(r['local_indices'])) for r in rows if r['local_indices']}); lookup={p:i for i,p in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in rows])
    checks.add('mapping dimensions',len(pools)==66 and len(rows)==90)
    w=npz(pre['same_weights'])['weights']; oldpre=js(REFERENCE.parent/'PREREG.json')
    checks.eq('original W same S113',w,npz(oldpre['same_weights'])['weights'])
    direction=npz(out/'DIRECTION.npz'); checks.eq('DIRECTION W',direction['weights'],w); checks.eq('DIRECTION arm W',direction[arm],w)
    items=[]
    for label,path,result,warm in [('dt25us_init40ms',REFERENCE,oldres,80),(arm,out/arm,res,160)]:
        a=npz(path/'TRAJECTORY.npz'); init=npz(path/'INITIALIZATION.npz'); meta=js(path/'COM_CONTACT_METADATA.json')
        committed={str(Path(p).resolve()) for p in result['hashes']}
        for name in ('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz','NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json'):
            checks.add(label+' committed '+name,str((path/name).resolve()) in committed)
        for key,value in a.items(): checks.add(label+' finite '+key,np.isfinite(value).all())
        checks.eq(label+' clock',a['time'],np.arange(121)*.000025,1e-12)
        for key,shape in dict(qpos=(121,49),qvel=(121,48),spikes=(6,17099),features=(6,66,5),applied=(6,90)).items(): checks.add(label+' shape '+key,a[key].shape==shape)
        for key,shape in dict(spikes=(warm,17099),features=(warm,66,5),decoded=(warm,90),history=(80,66)).items(): checks.add(label+' warm shape '+key,init[key].shape==shape)
        for alias,primary in [('warmup_spikes','spikes'),('warmup_features','features'),('warmup_decoded','decoded')]:
            if alias in init: checks.eq(label+' warm alias '+alias,init[alias],init[primary])
        checks.eq(label+' W',a['weights'],w); checks.eq(label+' warm W',init['weights'],w)
        checks.eq(label+' mapping',a['actuator_pool'],mapping)
        checks.eq(label+' q init',a['qpos'][0],init['body_qpos'])
        for key in ('qvel','act','ctrl'): checks.add(label+' zero initial '+key,not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' zero externals',not a['qfrc_applied'].any() and not a['xfrc_applied'].any())
        checks.add(label+' no warnings',not a['warnings'].any())
        checks.eq(label+' ctrl hold20',a['ctrl'][1:],np.repeat(a['applied'],20,axis=0))
        checks.eq(label+' fixed sensory current',a['current'],np.repeat(init['current'][None],6,axis=0))
        fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,w)
        _,_,warmhist=reconstruct_fir(init['spikes'],pools,mapping,w)
        # Warm160 records are distinct from the fixed recent80 history.
        recent=np.stack([[np.mean(s[list(p)])/.0005/100 for p in pools] for s in init['spikes'][-80:][::-1]])
        for name,x,y in [('warm features',init['features'],fs[:warm]),('warm decoded',init['decoded'],us[:warm]),('release features',init['release_features'],fs[warm-1]),('release decoded',init['release_decoded'],us[warm-1]),('warm history',init['history'],warmhist),('real recent80 spikes history',init['history'],recent),('runtime features',a['features'],fs[warm:]),('runtime applied',a['applied'],us[warm:]),('final FIR',npz(path/'NEURAL_FINAL.npz')['FIR_history'],hist)]: checks.eq(label+' FIR '+name,x,y,1e-14)
        schedule=js(path/'SCHEDULE.json'); checks.add(label+' six schedule rows',len(schedule)==6)
        for k,row in enumerate(schedule):
            checks.add(label+' consume id '+str(k),row['tick']==k and row['consume_id']==warm+k)
            for key,value in dict(source_start_s=(warm+k)*.0005,source_end_s=(warm+k+1)*.0005,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=warm*.0005).items(): checks.eq(label+' schedule '+str(k)+' '+key,row[key],value,1e-12)
        for name,steps in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',warm),('NEURAL_FINAL_META.json',warm+6)]:
            scalars=js(path/name)['neural_scalars']; checks.add(label+' clock '+name,scalars['steps']==steps and abs(scalars['t_ms']-steps*.5)<=1e-10)
        feet=raw_feet(a,meta)
        for key in ('foot_force','contacts','nonfoot_depth'): checks.eq(label+' independent raw '+key,a[key],feet[key],2e-12)
        if 'foot_order' in meta: checks.add(label+' explicit raw confirmed foot order',meta['foot_order']==list(LEGS))
        moment=js(path/'MOMENT_METADATA.json')
        checks.add(label+' manifest actuator names', [r['actuator'] for r in rows]==moment['actuator_names'])
        sparse=np.zeros_like(a['actuator_moment'])
        for ti in range(121):
            for u in range(90):
                start=int(a['moment_rowadr'][ti,u]); size=int(a['moment_rownnz'][ti,u]); cols=a['moment_colind'][ti,start:start+size]
                require(len(set(cols.tolist()))==size and np.all((cols>=0)&(cols<48)),'valid unique sparse moment columns')
                sparse[ti,u,cols]=a['moment_raw'][ti,start:start+size]
        checks.eq(label+' raw sparse CSR moment reconstruction',a['actuator_moment'],sparse)
        torque=a['actuator_force'][:,:,None]*sparse
        checks.eq(label+' actual signed torque all DOFs',torque.sum(1),a['qfrc_actuator'],2e-12)
        checks.add(label+' actual42limited joints',sum(moment['jnt_limited'])==42)
        limit_rows=[]
        for j,limited in enumerate(moment['jnt_limited']):
            if not limited: continue
            adr=moment['jnt_qposadr'][j]; lo,hi=moment['jnt_range'][j]; q=a['qpos'][:,adr]
            excess=np.maximum(np.maximum(lo-q,q-hi),0)
            limit_rows.append(dict(joint=moment['joint_names'][j],qpos_index=adr,dof_index=moment['jnt_dofadr'][j],range=[lo,hi],min_q=float(q.min()),max_q=float(q.max()),max_excess=float(excess.max()),max_excess_at_ms=float(a['time'][np.argmax(excess)]*1000)))
        rec=reconstruct(a,meta,checks,label); native=balance(a,rec); coarse=balance(a,rec,4)
        for key in ('instantaneous','impulse','momentum_FD'): checks.add(label+' native '+key,native[key]['passed'],details=native[key])
        weight=rec['mass']*abs(meta['gravity_world'][2]); checks.eq(label+' weight',result['weight_native'],weight,1e-12)
        metrics=physical(a,rec,checks,label,weight,xmls[0]); ix=np.flatnonzero(a['time']<=.0017+1e-12)
        metrics.update(initial_foot_Fz=float(feet['foot_force'][0,:,2].sum()),initial_load_ratio=float(metrics['load_ratio'][0]),load_deficit_integral0_1p7_ratio_ms=float(np.trapezoid(np.maximum(.8-metrics['load_ratio'][ix],0),x=a['time'][ix])*1000),max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),max_downward_COM_velocity=float(np.maximum(-rec['velocity'][:,2],0).max()),COM_momentum_samples={str(ms):rec['momentum'][int(round(ms/.025))] for ms in (0,1.6,1.7,3)})
        metrics['all42_joint_limits']=limit_rows
        metrics['warm_evidence']=dict(actual_warm_tick_count=len(init['spikes']),all_warm_real_spikes=int(init['spikes'].sum()),first80_real_spikes=int(init['spikes'][:80].sum()),recent80_real_spikes=int(init['spikes'][-80:].sum()),recent40_real_spikes=int(init['spikes'][-40:].sum()),release_nonzero_feature_elements=int(np.count_nonzero(init['release_features'])),release_active_actuator_count=int(np.count_nonzero(init['release_decoded'])),release_control_min=float(init['release_decoded'].min()),release_control_max=float(init['release_decoded'].max()))
        items.append(dict(label=label,path=str(path),arrays=a,initialization=init,derived=rec,feet=feet,physical=metrics,native=native,downsample100us_diagnostic_only=coarse,moment=moment))
    old,new=items
    checks.eq('exact original qpose',new['arrays']['qpos'][0],old['arrays']['qpos'][0])
    checks.add('exact all model numeric config',js(REFERENCE/'MODEL_PROVENANCE.json')['after']==js(out/arm/'MODEL_PROVENANCE.json')['after'])
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'): same(checks,'same '+name,npz(out/arm/name),npz(REFERENCE/name))
    for name in ('NEURAL_COLD_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json'): checks.add('same '+name,js(out/arm/name)==js(REFERENCE/name))
    checks.eq('same actual sensory current',new['initialization']['current'],old['initialization']['current'])
    for key in ('spikes','features','decoded'): checks.eq('deterministic first80 '+key,new['initialization'][key][:80],old['initialization'][key])
    changes={}
    for key in ('history','release_features','release_decoded','v','ge','gi','adaptation','_ring','refractory_until_ms','spike_count'):
        if key in new['initialization'] and key in old['initialization']: changes[key]=differences({key:new['initialization'][key]},{key:old['initialization'][key]})[key]
    changes['runtime']=differences({k:new['arrays'][k] for k in ('spikes','features','applied')},{k:old['arrays'][k] for k in ('spikes','features','applied')})
    changes['release_comparison_policy']='Actual networks after80 vs160 ticks may differ; no release/runtime equality forced. First80 warm spikes/features must match deterministic identical input.'
    effective=not np.array_equal(new['arrays']['applied'],old['arrays']['applied'])
    return checks,items,dict(ledger=ledger,observed_hashes=observed,source_audit=audit,neural_differences=changes,decision=decision(old['physical'],new['physical'],effective),mapping_rows=rows)


def outputs(out,items,checks,rows):
    def save(name,header,data):
        path=out/name
        with path.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(data)
        return str(path)
    poolrows=[]; actrows=[]; dofrows=[]; curves=[]; drive_summary={}
    for item in items:
        a=item['arrays']; init=item['initialization']; label=item['label']; meta=item['moment']; mapping=a['actuator_pool']; w=a['weights']
        names=meta['actuator_names']; raw=np.sum(init['release_features']*w,axis=1)
        stages=[('release',init['release_features'],init['release_decoded'])]+[('physical_tick'+str(k),a['features'][k],a['applied'][k]) for k in range(4)]
        drive_summary[label]={}
        for stage,features,control in stages:
            drive=(features*w).sum(1)
            drive_summary[label][stage]=dict(raw_min=float(drive.min()),raw_max=float(drive.max()),raw_nonzero_count=int(np.count_nonzero(drive)),clip_low_count=int(np.count_nonzero(drive<=0)),clip_high_count=int(np.count_nonzero(drive>=1)),control_min=float(control.min()),control_max=float(control.max()),active_actuators=int(np.count_nonzero(control)),active_pools=int(np.count_nonzero(np.clip(drive,0,1))))
            for p in range(len(w)):
                ids=np.flatnonzero(mapping==p); poolrows.append([label,stage,p,';'.join(map(str,ids)),';'.join(names[i] for i in ids),*features[p],*w[p],drive[p],np.clip(drive[p],0,1),int(drive[p]<=0),int(drive[p]>=1)])
        item['release_vs_first_applied']=dict(release_source_end_ms=len(init['spikes'])*.5,first_applied_source_start_ms=len(init['spikes'])*.5,first_applied_source_end_ms=(len(init['spikes'])+1)*.5,release_decoded=init['release_decoded'],first_applied=a['applied'][0],delta=a['applied'][0]-init['release_decoded'],convention='features are decoded AFTER consuming each source interval END; physical tick0 consumes next real source interval, not release command replay')
        first=item['physical']['first_failures_ms']['load']; end=.0017 if first is None else first/1000
        for i,t in enumerate(a['time']):
            if t>=end-1e-12: continue
            torque=a['actuator_force'][i,:,None]*a['actuator_moment'][i]
            for d in range(48):
                joint=meta['joint_names'][meta['dof_jntid'][d]]
                dofrows.append([label,t*1000,d,joint,a['qfrc_actuator'][i,d],json.dumps(torque[:,d].tolist()),json.dumps(a['actuator_force'][i].tolist()),int(d==8)])
        for ms in (0,.025,1.6):
            i=int(round(ms/.025))
            for u,name in enumerate(names):
                torque=a['actuator_force'][i,u]*a['actuator_moment'][i,u]
                actrows.append([label,ms,u,name,int(mapping[u]),a['ctrl'][i,u],a['act'][i,u],a['actuator_force'][i,u],a['actuator_moment'][i,u,8],torque[8],json.dumps(torque.tolist()),a['qfrc_actuator'][i,8]])
        for i,t in enumerate(a['time']): curves.append([label,t*1000,item['physical']['load_ratio'][i],*item['feet']['foot_force'][i,:,2],*item['derived']['momentum'][i],item['derived']['velocity'][i,2],a['limit_excess'][i],item['physical']['extrema']['angle_rad']['max'],a['nonfoot_depth'][i]])
    historyrows=[]; limitrows=[]
    for item in items:
        for age,row in enumerate(item['initialization']['history']):
            for pool,value in enumerate(row): historyrows.append([item['label'],age,(age+1)*.5,pool,value])
        for row in item['physical']['all42_joint_limits']:
            limitrows.append([item['label'],row['joint'],row['qpos_index'],row['dof_index'],*row['range'],row['min_q'],row['max_q'],row['max_excess'],row['max_excess_at_ms']])
    paths=[save('EXTENDED_INIT_RECENT80_REAL_SPIKE_HISTORY.csv',['arm','age_tick_newest0','age_bin_end_ms','pool','real_normalized_pool_spikes'],historyrows),save('EXTENDED_INIT_ALL42_LIMITS.csv',['arm','actual_joint','qpos_index','DOF','lower','upper','min_q','max_q','max_excess','max_excess_at_ms'],limitrows),save('EXTENDED_INIT_POOL_RELEASE.csv',['arm','stage','pool','actuator_ids','actual_actuator_names',*[f'feature_bin{k}' for k in range(5)],*[f'W_bin{k}' for k in range(5)],'raw_weighted_drive','clipped_control','clip_low','clip_high'],poolrows),save('EXTENDED_INIT_PERACTUATOR_EARLY_RESPONSE.csv',['arm','physical_ms','actuator','actual_name','pool','ctrl','activation','actual_signed_force','moment_DOF8','actual_signed_torque_DOF8','actual_signed_torque_all48_json','aggregate_DOF8'],actrows),save('EXTENDED_INIT_PREFAILURE_ALL_DOF.csv',['arm','physical_ms','DOF','actual_joint','aggregate_signed_actuator_torque','peractuator_signed_torque90_json','actual_signed_force90_json','target_DOF8'],dofrows),save('EXTENDED_INIT_COMPARISON.csv',['arm','physical_ms','load_ratio',*[l+'_Fz' for l in LEGS],'COM_px','COM_py','COM_pz','COM_vz','all42_limit_excess','max_angle_fullwindow','nonfoot_depth'],curves),save('EXTENDED_INIT_ALL_CHECKS.csv',['name','passed','details_json'],[[r['name'],r['passed'],json.dumps(r,default=serial,allow_nan=False)] for r in checks.rows])]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    for item in items:
        a=item['arrays']; t=a['time']*1000; p=item['physical']; label=item['label']
        axes[0,0].plot(t,p['load_ratio'],label=label)
        axes[0,1].plot(t,item['derived']['velocity'][:,2],label=label)
        axes[1,0].plot(t,a['limit_excess'],label=label)
        q=a['qpos'][:,3:7]; angle=2*np.arccos(np.clip(np.abs(q@q[0])/(np.linalg.norm(q,axis=1)*np.linalg.norm(q[0])),0,1))
        axes[1,1].plot(t,angle,label=label)
        axes[2,0].plot(t,np.max(np.abs(item['derived']['force_residual']),axis=1),label=label)
        ids=item['native']['momentum_FD']['sample_indices']; axes[2,1].plot(t[ids],np.max(np.abs(item['native']['fd_residual'][ids]),axis=1),label=label)
    for ax,title in zip(axes.flat,['Load: actual six-foot Fz / weight','COM vertical velocity','All42 joint maximum limit excess','Root orientation angle','Raw instantaneous COM residual','Native original centered momentum FD residual']):
        ax.set_title(title); ax.set_xlabel('physical ms'); ax.legend(fontsize=7); ax.grid(alpha=.25); ax.axvline(1.7,color='grey',ls=':')
    axes[0,0].axhline(.8,color='black',ls=':'); axes[2,0].set_yscale('symlog',linthresh=1e-15)
    fig.suptitle('S115 40ms reference vs80ms actual neural initialization, same W/body\nOffline native25us evidence, 3ms diagnostic only; NOT_ESTABLISHED units; no training/walking acceptance')
    fig.tight_layout(); png=out/'EXTENDED_INIT_COMPARISON.png'; fig.savefig(png,dpi=150); plt.close(fig)
    return dict(comparison_png=str(png),tables=paths,drive_summary=drive_summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT)
    out=parser.parse_args().output.resolve()
    report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE',historical_coarse_failures='Preserved unchanged in S113. Downsample100us here diagnostic only; not promoted to native gates.')
    try:
        checks,items,details=verify(out)
        generated=outputs(out,items,checks,details.pop('mapping_rows'))
        report.update(details,**generated,passed=checks.passed,checks=checks.rows,per_arm={i['label']:{k:i[k] for k in ('path','physical','native','downsample100us_diagnostic_only','release_vs_first_applied')} for i in items})
        require(not any(x in sys.modules for x in ('mujoco','torch')),'physics/neural imported')
    except Exception as exc: report.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'EXTENDED_INIT_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),report=str(out/'EXTENDED_INIT_INDEPENDENT_CHECK.json')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
