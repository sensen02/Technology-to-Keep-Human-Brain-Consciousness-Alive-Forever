"""S117/P1/S2-S4 saved-array-only single-weight attenuation audit.
No runner, physics framework or neural engine imports; no simulation calls.
Native original-scope COM criteria and six physical gates remain unchanged.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import traceback
import numpy as np
from verify_com_contact_probe import Checks, require, sha, js, npz, serial, write_json, reconstruct_fir
from verify_com_timestep_probe import canonical_checks, reconstruct, physical
from verify_initial_height_probe import raw_feet, balance, differences, LEGS
from verify_extended_initialization_probe import same, source_audit

ROOT = Path(__file__).resolve().parent
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s117-single-weight-attenuation/20261024T064500Z')
S115 = Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z/init80ms')
S113 = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us')
ARM = 'w36b4_down'
OLD_VALUE = .18265069824315935
NEW_VALUE = .1461205585945275


def mapped(features, weights, mapping):
    pooled = np.clip(np.sum(features * weights, axis=-1), 0, 1)
    result = np.zeros(features.shape[:-2] + (len(mapping),))
    valid = mapping >= 0
    result[..., valid] = pooled[..., mapping[valid]]
    return result


def decision(s113, s115, new, effective):
    """Strict numerical <=/>= only; no epsilon or retrospective tradeoff."""
    regress = []
    for key in ('limit_excess', 'angle_rad', 'nonfoot_depth'):
        if new['extrema'][key]['max'] > s113['extrema'][key]['max']:
            regress.append(key)
    if new['extrema']['height_ratio']['min'] < s113['extrema']['height_ratio']['min']:
        regress.append('height_ratio')
    for key in ('max_downward_root_velocity','max_downward_COM_velocity'):
        if new[key] > s113[key]: regress.append(key)
    for gate in ('height', 'angle', 'legs', 'nonfoot', 'limit'):
        old = s113['first_failures_ms'][gate]; now = new['first_failures_ms'][gate]
        if now is not None and (old is None or now < old): regress.append('earlier_' + gate)
    oldload = s115['first_failures_ms']['load']; newload = new['first_failures_ms']['load']
    retained = dict(initial_load=new['initial_load_ratio'] >= s115['initial_load_ratio'],
                    deficit=new['load_deficit_integral0_1p7_ratio_ms'] <= s115['load_deficit_integral0_1p7_ratio_ms'],
                    first_load_failure=newload is None or (oldload is not None and newload >= oldload),
                    support=new['all_gates_support_ms'] >= s115['all_gates_support_ms'],
                    longest=new['longest_all_gates_true_endpoint_span_ms'] >= s115['longest_all_gates_true_endpoint_span_ms'])
    candidate = bool(effective and all(retained.values()) and not regress)
    return dict(reject=not candidate, suggest_candidate=candidate, actual_output_changed=bool(effective),
                S115_support_benefits_retained=retained, safety_regressions_against_S113=regress,
                safety_nonregression=not regress, automatic_checkpoint=False, checkpoint_replacement=False,
                deployment=False, standing_or_walking_claim=False, training_acceptance=False,
                rule='All S115 support benefits retained (deficit<=, load failure>=, support>=, longest>=), AND S113 safety no worse (all42 maximum limit excess/angle/nonfoot/height and each safety first-failure), plus initial load>=S115 and downward root/COM velocity<=S113 as committed by runner preregistration; otherwise reject.')


def verify(out):
    require(not any(k in sys.modules for k in ('mujoco', 'torch')), 'physics/neural imported')
    checks = Checks(); observed = {}; pre = js(out/'PREREG.json'); sources = pre['source_hashes']
    canonical_checks(checks, sources, 'source', observed)
    canonical = {str(Path(p).resolve()): h for p, h in sources.items()}
    runner = ROOT/'run_single_weight_attenuation.py'
    checks.add('runner source committed', canonical.get(str(runner.resolve())) == sha(runner))
    audit = source_audit(runner, checks)
    checks.add('INPUT_HASHES exact', js(out/'INPUT_HASHES.json') == sources)
    dependencies = js(out/'MODEL_DEPENDENCIES.json')
    canonical_checks(checks, dependencies, 'dependencies', observed)
    checks.add('dependencies committed', all(canonical.get(str(Path(p).resolve())) == h for p,h in dependencies.items()))
    checks.add('complete no failure', js(out/'COMPLETE.json')['status'] == 'COMPLETE' and not (out/'FAILED.json').exists())
    checks.add('exact arm', pre['arms'] == [ARM])
    checks.add('warm160 actual80ms', pre['initialization_ticks'] == 160 and pre['initialization_ms'] == 80 and pre['neural_dt_ms'] == .5)
    checks.add('physical3ms six ticks', pre['physical_duration_ms'] == 3 and pre['physical_neural_ticks_per_arm'] == 6)
    checks.eq('source offset80ms', pre['source_offset_s'], .08, 1e-12)
    checks.add('dt25us20substeps', pre['timesteps_s'] == {ARM:.000025} and pre['substeps'] == {ARM:20})
    checks.add('121states120steps', pre['physical_samples_per_arm'] == {ARM:121} and pre['physical_steps_per_arm'] == {ARM:120})
    count = dict(model_load=1, forward=121, step=120, neural_ticks=166)
    ledger = js(out/'CALL_LEDGER.json')
    checks.add('exact call ledger', ledger == dict(count, active=None, completed=[ARM], per_arm={ARM:count}), observed=ledger)
    results = js(out/'RESULTS.json'); result = js(out/ARM/'RESULT.json')
    checks.add('single result', results == [result] and result['label'] == ARM)
    manifests = [Path(p) for p in sources if Path(p).name == 'MANIFEST.json']
    xmls = [Path(p) for p in sources if Path(p).suffix == '.xml']
    require(len(manifests) == len(xmls) == 1, 'unique mapping and XML')
    rows = sorted(js(manifests[0])['rows'], key=lambda r:r['actuator_index'])
    pools = sorted({tuple(sorted(r['local_indices'])) for r in rows if r['local_indices']})
    lookup = {p:i for i,p in enumerate(pools)}
    mapping = np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in rows])
    checks.add('66pools90act and pool36 only act2', len(pools)==66 and len(mapping)==90 and np.array_equal(np.flatnonzero(mapping==36), [2]))
    oldw = npz(S115/'TRAJECTORY.npz')['weights']
    candidate = oldw.copy(); candidate[36,4] = NEW_VALUE
    checks.add('source W shape66x5', oldw.shape == (66,5))
    checks.eq('old element exact', oldw[36,4], OLD_VALUE)
    direction = npz(out/'DIRECTION.npz')
    checks.eq('DIRECTION source W', direction['weights'], oldw)
    checks.eq('DIRECTION candidate W', direction[ARM], candidate)
    mask = candidate != oldw; expected_mask = np.zeros_like(mask); expected_mask[36,4] = True
    checks.eq('independent sole element mask', mask, expected_mask)
    checks.eq('DIRECTION old_weights',direction['old_weights'],oldw)
    checks.eq('DIRECTION new_weights',direction['new_weights'],candidate)
    checks.add('zero mask preserved',np.array_equal(oldw==0,candidate==0))
    checks.add('weight bounds',bool(np.all((candidate>=0)&(candidate<=4))))
    weight_hashes=js(out/'WEIGHT_HASHES.json')
    for name,weights in [('old',oldw),('new',candidate)]:
        digest=hashlib.sha256(np.ascontiguousarray(weights,dtype=np.float64).tobytes()).hexdigest()
        checks.add(name+' weight independent array byte SHA',weight_hashes[name+'_W_sha256']==digest and pre[name+'_W_sha256']==digest and str(direction[name+'_W_sha256'])==digest,observed_sha256=digest)
    checks.add('source weight file committed',canonical.get(str(Path(pre['same_weights']).resolve()))==sha(pre['same_weights']))
    checks.eq('source weight file exact',npz(pre['same_weights'])['weights'],oldw)
    # Raw NPZ identities are source hashes, array masks are independently derived.
    weight_sources = []
    for p,h in sources.items():
        if Path(p).suffix == '.npz' and 'weight' in Path(p).name.lower():
            weight_sources.append(dict(path=str(Path(p).resolve()), sha256=h))
    items = []
    required = ('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz','NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json')
    for label,path,warm,w,res in [('S113_dt25us_init40ms',S113,80,oldw,js(S113/'RESULT.json')), ('S115_init80ms',S115,160,oldw,js(S115/'RESULT.json')), (ARM,out/ARM,160,candidate,result)]:
        canonical_checks(checks, res['hashes'], label+' artifacts', observed)
        committed = {str(Path(p).resolve()) for p in res['hashes']}
        for name in required:
            checks.add(label+' committed '+name, str((path/name).resolve()) in committed)
        if label != ARM:
            # A preregistered RESULT hash transitively commits every artifact; no
            # requirement that runners redundantly list all artifact paths.
            result_bound = canonical.get(str((path/'RESULT.json').resolve())) == sha(path/'RESULT.json')
            direct_bound = all(canonical.get(str((path/name).resolve())) == sha(path/name) for name in required)
            checks.add(label+' prereg reference commitment', result_bound or direct_bound)
        a=npz(path/'TRAJECTORY.npz'); init=npz(path/'INITIALIZATION.npz'); meta=js(path/'COM_CONTACT_METADATA.json'); moment=js(path/'MOMENT_METADATA.json')
        for key,value in a.items(): checks.add(label+' finite '+key, np.isfinite(value).all())
        for key,value in init.items(): checks.add(label+' finite warm '+key, np.isfinite(value).all())
        checks.eq(label+' time', a['time'], np.arange(121)*.000025, 1e-12)
        for key,shape in dict(qpos=(121,49),qvel=(121,48),spikes=(6,17099),features=(6,66,5),applied=(6,90)).items(): checks.add(label+' shape '+key, a[key].shape==shape)
        for key,shape in dict(spikes=(warm,17099),features=(warm,66,5),decoded=(warm,90),history=(80,66)).items(): checks.add(label+' warm shape '+key, init[key].shape==shape)
        checks.eq(label+' runtime W', a['weights'], w); checks.eq(label+' W used from cold', init['weights'], w)
        checks.eq(label+' mapping', a['actuator_pool'], mapping)
        checks.eq(label+' qpos release', a['qpos'][0], init['body_qpos'])
        for key in ('qvel','act','ctrl'): checks.add(label+' initial zero '+key, not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' zero externals warnings', not a['qfrc_applied'].any() and not a['xfrc_applied'].any() and not a['warnings'].any())
        checks.eq(label+' hold20', a['ctrl'][1:], np.repeat(a['applied'],20,axis=0))
        checks.eq(label+' fixed actual current', a['current'], np.repeat(init['current'][None],6,axis=0))
        fs,us,hist=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,w)
        _,_,warmhist=reconstruct_fir(init['spikes'],pools,mapping,w)
        recent=np.stack([[np.mean(s[list(p)])/.0005/100 for p in pools] for s in init['spikes'][-80:][::-1]])
        for key,x,y in [('warm features',init['features'],fs[:warm]),('warm commands',init['decoded'],us[:warm]),('release features',init['release_features'],fs[warm-1]),('release decoded',init['release_decoded'],us[warm-1]),('warm history',init['history'],warmhist),('recent80 real spike history',init['history'],recent),('runtime features',a['features'],fs[warm:]),('runtime commands',a['applied'],us[warm:]),('final FIR',npz(path/'NEURAL_FINAL.npz')['FIR_history'],hist)]: checks.eq(label+' reconstruct '+key,x,y,1e-14)
        schedule=js(path/'SCHEDULE.json'); checks.add(label+' six schedule',len(schedule)==6)
        for k,row in enumerate(schedule):
            checks.add(label+' consume'+str(k),row['tick']==k and row['consume_id']==warm+k)
            for key,value in dict(source_start_s=(warm+k)*.0005,source_end_s=(warm+k+1)*.0005,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=warm*.0005).items(): checks.eq(label+' schedule '+str(k)+' '+key,row[key],value,1e-12)
        for name,n in [('NEURAL_COLD_META.json',0),('NEURAL_RELEASE_META.json',warm),('NEURAL_FINAL_META.json',warm+6)]:
            scalars=js(path/name)['neural_scalars']; checks.add(label+' clock '+name,scalars['steps']==n and abs(scalars['t_ms']-.5*n)<=1e-10)
        feet=raw_feet(a,meta)
        for key in ('foot_force','contacts','nonfoot_depth'): checks.eq(label+' raw foot '+key,a[key],feet[key],2e-12)
        if 'foot_order' in meta: checks.add(label+' raw foot identity order',meta['foot_order']==list(LEGS))
        checks.add(label+' manifest actual names',[r['actuator'] for r in rows]==moment['actuator_names'])
        sparse=np.zeros_like(a['actuator_moment'])
        for ti in range(121):
            for u in range(90):
                start=int(a['moment_rowadr'][ti,u]); size=int(a['moment_rownnz'][ti,u]); cols=a['moment_colind'][ti,start:start+size]
                require(len(set(cols.tolist()))==size and np.all((cols>=0)&(cols<48)),'unique valid moment CSR')
                sparse[ti,u,cols]=a['moment_raw'][ti,start:start+size]
        checks.eq(label+' independent CSR moment',a['actuator_moment'],sparse)
        checks.eq(label+' all90 signed torque sum',np.einsum('tu,tud->td',a['actuator_force'],sparse),a['qfrc_actuator'],2e-12)
        checks.add(label+' all42limited',sum(moment['jnt_limited'])==42)
        limitrows=[]
        for j,limited in enumerate(moment['jnt_limited']):
            if not limited: continue
            adr=moment['jnt_qposadr'][j]; lo,hi=moment['jnt_range'][j]; q=a['qpos'][:,adr]; excess=np.maximum(np.maximum(lo-q,q-hi),0)
            limitrows.append(dict(joint=moment['joint_names'][j],qpos_index=adr,dof_index=moment['jnt_dofadr'][j],range=[lo,hi],min_q=float(q.min()),max_q=float(q.max()),max_excess=float(excess.max()),max_excess_at_ms=float(a['time'][excess.argmax()]*1000)))
        rec=reconstruct(a,meta,checks,label); native=balance(a,rec)
        for key in ('instantaneous','impulse','momentum_FD'): checks.add(label+' native '+key,native[key]['passed'],details=native[key])
        weight=rec['mass']*abs(meta['gravity_world'][2]); checks.eq(label+' native weight',res['weight_native'],weight,1e-12)
        metrics=physical(a,rec,checks,label,weight,xmls[0]); ix=np.flatnonzero(a['time']<=.0017+1e-12)
        metrics.update(initial_foot_Fz=float(feet['foot_force'][0,:,2].sum()),initial_load_ratio=float(metrics['load_ratio'][0]),load_deficit_integral0_1p7_ratio_ms=float(np.trapezoid(np.maximum(.8-metrics['load_ratio'][ix],0),x=a['time'][ix])*1000),max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),max_downward_COM_velocity=float(np.maximum(-rec['velocity'][:,2],0).max()),COM_momentum_samples={str(ms):rec['momentum'][int(round(ms/.025))] for ms in (0,1.6,1.7,3)},all42_joint_limits=limitrows,
                       Jload=float(np.trapezoid((np.maximum(.8-metrics['load_ratio'],0)/.8)**2,x=a['time'])/.003),
                       Jlimit=float(np.trapezoid((np.maximum(a['limit_excess']-1e-6,0)/.001)**2,x=a['time'])/.003))
        items.append(dict(label=label,path=str(path),arrays=a,initialization=init,derived=rec,feet=feet,physical=metrics,native=native,moment=moment))
    old,base,new=items
    for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'): same(checks,'exact S115 '+name,npz(Path(new['path'])/name),npz(S115/name))
    for name in ('NEURAL_COLD_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json'): checks.add('exact S115 '+name,js(Path(new['path'])/name)==js(S115/name))
    newprov=js(Path(new['path'])/'MODEL_PROVENANCE.json'); oldprov=js(S115/'MODEL_PROVENANCE.json')
    for key in ('before','after'):
        checks.add('all model numeric '+key+' exact S115',newprov[key]==oldprov[key])
    for key in ('body_qpos','body_qvel','body_act','body_ctrl','current'): checks.eq('exact S115 initial '+key,new['initialization'][key],base['initialization'][key])
    checks.eq('exact actual runtime current S115',new['arrays']['current'],base['arrays']['current'])
    for key in ('qpos','qvel','act','ctrl'): checks.eq('exact body initial S115 '+key,new['arrays'][key][0],base['arrays'][key][0])
    neural=dict(warm=differences(new['initialization'],base['initialization']),runtime=differences({k:new['arrays'][k] for k in ('spikes','features','applied')},{k:base['arrays'][k] for k in ('spikes','features','applied')}),final=differences(npz(Path(new['path'])/'NEURAL_FINAL.npz'),npz(S115/'NEURAL_FINAL.npz')),release_metadata_equal=js(Path(new['path'])/'NEURAL_RELEASE_META.json')==js(S115/'NEURAL_RELEASE_META.json'),policy='Expected same neural events under fixed current, but actual spikes/features/history/release metadata are measured, NOT equality-forced.')
    frozen={}
    for stage,features,actual in [('warm',base['initialization']['features'],new['initialization']['decoded']),('runtime',base['arrays']['features'],new['arrays']['applied'])]:
        predicted=mapped(features,candidate,mapping); currentfeatures=new['initialization']['features'] if stage=='warm' else new['arrays']['features']; samefeatures=np.array_equal(features,currentfeatures)
        frozen[stage]=dict(features_equal=bool(samefeatures),prediction=predicted,actual=actual,actual_minus_frozen=actual-predicted,max_abs_delta=float(np.abs(actual-predicted).max()),policy='Frozen S115 feature algebra, not a new neural/physical trajectory prediction; differing actual neural features are reported without false equality failure.')
        if samefeatures: checks.eq(stage+' frozen feature candidate exact clipped90 actual',actual,predicted,1e-14)
    checks.add('warm decoded actually changed',not np.array_equal(new['initialization']['decoded'],base['initialization']['decoded']))
    return checks,items,dict(observed_hashes=observed,weight_file_sources=weight_sources,weight_change=dict(index=[36,4],old=OLD_VALUE,new=NEW_VALUE,changed_elements=int(mask.sum())),source_audit=audit,ledger=ledger,neural_differences=neural,frozen_feature_predictions=frozen,decision=decision(old['physical'],base['physical'],new['physical'],not np.array_equal(new['arrays']['applied'],base['arrays']['applied'])))


def outputs(out,items,details):
    def save(name,header,data):
        path=out/name
        with path.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(data)
        return str(path)
    drive=[]; limits=[]; curves=[]; torques=[]; predictions=[]
    for item in items:
        a=item['arrays']; init=item['initialization']; label=item['label']; mapping=a['actuator_pool']; w=a['weights']
        for phase,features,control in [('warm',init['features'],init['decoded']),('runtime',a['features'],a['applied'])]:
            for k,(f,u) in enumerate(zip(features,control)):
                raw=(f*w).sum(1)
                for act,pool in enumerate(mapping):
                    drive.append([label,phase,k,act,int(pool),*([0.]*5 if pool<0 else f[pool]),*([0.]*5 if pool<0 else w[pool]),0 if pool<0 else raw[pool],u[act]])
        for row in item['physical']['all42_joint_limits']: limits.append([label,row['joint'],row['qpos_index'],row['dof_index'],*row['range'],row['min_q'],row['max_q'],row['max_excess'],row['max_excess_at_ms']])
        for k,t in enumerate(a['time']):
            p=item['physical']; curves.append([label,t*1000,p['load_ratio'][k],*item['feet']['foot_force'][k,:,2],*item['derived']['momentum'][k],item['derived']['velocity'][k,2],a['limit_excess'][k],*p['gates'][k]])
            for d in (6,7,8): torques.append([label,t*1000,2,item['moment']['actuator_names'][2],d,a['ctrl'][k,2],a['act'][k,2],a['actuator_force'][k,2],a['actuator_moment'][k,2,d],a['actuator_force'][k,2]*a['actuator_moment'][k,2,d],a['qfrc_actuator'][k,d]])
    for stage,row in details['frozen_feature_predictions'].items():
        for k,(pred,actual) in enumerate(zip(row['prediction'],row['actual'])):
            for act in range(90): predictions.append([stage,k,act,row['features_equal'],pred[act],actual[act],actual[act]-pred[act]])
    base=items[1]['arrays']; new=items[2]['arrays']; decomposition=[]
    for k,t in enumerate(new['time']):
        for d in (6,7,8):
            f0=base['actuator_force'][k,2]; df=new['actuator_force'][k,2]-f0; m0=base['actuator_moment'][k,2,d]; dm=new['actuator_moment'][k,2,d]-m0
            decomposition.append([t*1000,d,f0,df,m0,dm,df*m0,f0*dm,df*dm,new['actuator_force'][k,2]*new['actuator_moment'][k,2,d]-f0*m0])
    paths=[save('SINGLE_WEIGHT_WARM_RUNTIME_WEIGHTS.csv',['arm','phase','tick','actuator','pool',*[f'feature{k}' for k in range(5)],*[f'weight{k}' for k in range(5)],'raw_drive','actual_clipped90'],drive),save('SINGLE_WEIGHT_PREDICTED_ACTUAL.csv',['phase','tick','actuator','neural_features_equal','frozen_S115_feature_candidate_prediction','actual','actual_minus_predicted'],predictions),save('SINGLE_WEIGHT_ALL42_LIMITS.csv',['arm','joint','qpos_index','dof_index','lower','upper','min_q','max_q','max_excess','at_ms'],limits),save('SINGLE_WEIGHT_ACT2_SIGNED_MOMENTS.csv',['arm','physical_ms','actuator','name','DOF','ctrl','activation','actual_signed_force','signed_moment','actual_signed_torque','aggregate_signed_torque'],torques),save('SINGLE_WEIGHT_ACT2_DECOMPOSITION.csv',['physical_ms','DOF','S115_force','delta_force','S115_moment','delta_moment','force_term','moment_term','cross_term','actual_torque_delta'],decomposition),save('SINGLE_WEIGHT_COMPARISON.csv',['arm','physical_ms','load_ratio',*[l+'_Fz' for l in LEGS],'COM_px','COM_py','COM_pz','COM_vz','all42_max_limit_excess','height_gate','angle_gate','legs_gate','load_gate','nonfoot_gate','limit_gate'],curves)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    for item in items:
        a=item['arrays']; t=a['time']*1000; label=item['label']
        axes[0,0].step(np.arange(6)*.5,a['applied'][:,2],where='post',label=label)
        for d in (6,7,8): axes[0,1].plot(t,a['actuator_force'][:,2]*a['actuator_moment'][:,2,d],label=label+' DOF'+str(d))
        axes[1,0].plot(t,item['physical']['load_ratio'],label=label)
        axes[1,1].plot(t,a['limit_excess'],label=label)
        axes[2,0].plot(t,item['derived']['momentum'][:,2],label=label)
        axes[2,1].plot(t,item['derived']['velocity'][:,2],label=label)
    for ax,title in zip(axes.flat,['Act2 actual command (all90 in CSV)','Act2 actual signed force x moment DOF6/7/8','Six-foot support Fz / weight','All42 maximum joint limit excess','Whole-body COM vertical momentum','Whole-body COM vertical velocity']):
        ax.set_title(title); ax.set_xlabel('physical ms'); ax.legend(fontsize=6); ax.grid(alpha=.25); ax.axvline(1.7,color='gray',ls=':')
    axes[1,0].axhline(.8,color='black',ls=':'); axes[1,1].axhline(1e-6,color='black',ls=':')
    fig.suptitle('S117 single W[36,4] attenuation: S113 / S115 / candidate\nOffline saved-array check; 3ms diagnostic only; units NOT_ESTABLISHED; no checkpoint or walking claim')
    fig.tight_layout(); png=out/'SINGLE_WEIGHT_COMPARISON.png'; fig.savefig(png,dpi=150); plt.close(fig)
    return dict(comparison_png=str(png),tables=paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT); out=parser.parse_args().output.resolve()
    report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')
    try:
        checks,items,details=verify(out); generated=outputs(out,items,details)
        report.update(details,**generated,passed=checks.passed,checks=checks.rows,per_arm={i['label']:{k:i[k] for k in ('path','physical','native')} for i in items})
        require(not any(k in sys.modules for k in ('mujoco','torch')),'physics/neural imported')
    except Exception as exc: report.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'SINGLE_WEIGHT_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),report=str(out/'SINGLE_WEIGHT_INDEPENDENT_CHECK.json')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
