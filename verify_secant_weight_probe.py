"""S118/P1/S2-S4 saved-array-only exact secant weight probe verification.
No runner, neural or physics imports/calls. A secant is a local proposal, not
an optimality/safety prediction; original native COM criteria remain unchanged.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import traceback
import numpy as np
from verify_com_contact_probe import Checks, require, sha, js, npz, write_json, reconstruct_fir
from verify_com_timestep_probe import canonical_checks, reconstruct, physical
from verify_initial_height_probe import raw_feet, balance, differences, LEGS
from verify_extended_initialization_probe import same, source_audit
from verify_single_weight_attenuation import mapped, decision

ROOT = Path(__file__).resolve().parent
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s118-secant-weight-step/20261024T065500Z')
S113 = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us')
S115 = Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z/init80ms')
S117 = Path('/home/sensen/Desktop/engine修复验收/s117-single-weight-attenuation/20261024T064500Z/w36b4_down')
ARM = 'w36b4_secant'


def all42_limits(a, moment):
    """Independent actual-q range excess; never trust cached limit_excess."""
    rows = []; series = []
    for j, limited in enumerate(moment['jnt_limited']):
        if not limited: continue
        adr = moment['jnt_qposadr'][j]; lo, hi = moment['jnt_range'][j]
        q = a['qpos'][:, adr]; excess = np.maximum(np.maximum(lo-q, q-hi), 0)
        series.append(excess)
        rows.append(dict(joint=moment['joint_names'][j], qpos_index=adr,
                         dof_index=moment['jnt_dofadr'][j], range=[lo, hi],
                         min_q=float(q.min()), max_q=float(q.max()),
                         max_excess=float(excess.max()),
                         max_excess_at_ms=float(a['time'][excess.argmax()]*1000)))
    require(len(rows) == 42, 'exactly 42 limited joints')
    matrix = np.stack(series, axis=1)
    ti, ji = np.unravel_index(matrix.argmax(), matrix.shape)
    peak = dict(value=float(matrix[ti, ji]), joint=rows[ji]['joint'],
                qpos_index=rows[ji]['qpos_index'], dof_index=rows[ji]['dof_index'],
                sample_index=int(ti), time_s=float(a['time'][ti]),
                time_ms=float(a['time'][ti]*1000))
    return rows, matrix.max(axis=1), peak


def secant_from_saved(paths=None):
    """Calculate xstar with exact requested operation order from saved arrays."""
    paths = paths or (S113, S115, S117)
    refs = []
    for path in paths:
        a = npz(path/'TRAJECTORY.npz'); m = js(path/'MOMENT_METADATA.json')
        _, _, peak = all42_limits(a, m)
        refs.append(dict(path=str(path), x=float(a['weights'][36,4]), peak=peak))
    r113, r115, r117 = refs
    x115, x117 = r115['x'], r117['x']
    y113, y115, y117 = [r['peak']['value'] for r in refs]
    require(r115['peak']['joint'] == r117['peak']['joint'] and
            r115['peak']['sample_index'] == r117['peak']['sample_index'] and
            r115['peak']['time_s'] == r117['peak']['time_s'],
            'S115/S117 secant peak joint/time differs')
    require(np.isfinite([x115,x117,y113,y115,y117]).all() and y117 != y115,
            'nonfinite or zero secant denominator')
    require(0 < x117 < x115 and y117 < y115, 'invalid secant orientation: require x117<x115 and y117<y115')
    xstar = x117 + (y113-y117)*(x117-x115)/(y117-y115)
    require(np.isfinite(xstar) and 0 < xstar < x117, 'invalid secant: require 0 < xstar < x117')
    return dict(formula='x117+(y113-y117)*(x117-x115)/(y117-y115)',
                x115=x115, x117=x117, y113=y113, y115=y115, y117=y117,
                xstar=float(xstar), same_S115_S117_peak_joint_time=True, references=refs,
                scope='y is maximum all42 limit excess independently reconstructed from actual q; no clipping/rounding or simulation prediction')


def verify(out):
    require(not any(k in sys.modules for k in ('mujoco','torch')), 'physics/neural imported')
    checks = Checks(); observed = {}; pre = js(out/'PREREG.json'); sources = pre['source_hashes']
    canonical_checks(checks, sources, 'source', observed)
    canonical = {str(Path(p).resolve()): h for p,h in sources.items()}
    runner = ROOT/'run_secant_weight_probe.py'
    checks.add('runner source committed', canonical.get(str(runner.resolve())) == sha(runner))
    audit = source_audit(runner, checks)
    checks.add('INPUT_HASHES exact', js(out/'INPUT_HASHES.json') == sources)
    dependencies = js(out/'MODEL_DEPENDENCIES.json')
    canonical_checks(checks, dependencies, 'dependencies', observed)
    checks.add('dependencies committed', all(canonical.get(str(Path(p).resolve())) == h for p,h in dependencies.items()))
    checks.add('complete no failure', js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    checks.add('exact arm', pre['arms']==[ARM])
    checks.add('warm160 actual80ms', pre['initialization_ticks']==160 and pre['initialization_ms']==80 and pre['neural_dt_ms']==.5)
    checks.add('physical3ms six ticks', pre['physical_duration_ms']==3 and pre['physical_neural_ticks_per_arm']==6)
    checks.eq('source offset80ms', pre['source_offset_s'], .08, 1e-12)
    checks.add('dt25us20substeps', pre['timesteps_s']=={ARM:.000025} and pre['substeps']=={ARM:20})
    checks.add('121states120steps', pre['physical_samples_per_arm']=={ARM:121} and pre['physical_steps_per_arm']=={ARM:120})
    count = dict(model_load=1,forward=121,step=120,neural_ticks=166); ledger=js(out/'CALL_LEDGER.json')
    checks.add('exact call ledger', ledger==dict(count,active=None,completed=[ARM],per_arm={ARM:count}), observed=ledger)
    result=js(out/ARM/'RESULT.json')
    checks.add('single result', js(out/'RESULTS.json')==[result] and result['label']==ARM)
    manifests=[Path(p) for p in sources if Path(p).name=='MANIFEST.json']
    xmls=[Path(p) for p in sources if Path(p).suffix=='.xml']
    require(len(manifests)==len(xmls)==1, 'unique mapping and XML')
    rows=sorted(js(manifests[0])['rows'],key=lambda r:r['actuator_index'])
    pools=sorted({tuple(sorted(r['local_indices'])) for r in rows if r['local_indices']})
    lookup={p:i for i,p in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in rows])
    checks.add('66pools90act pool36 only act2', len(pools)==66 and len(mapping)==90 and np.array_equal(np.flatnonzero(mapping==36),[2]))
    secant=secant_from_saved(); new_value=secant['xstar']
    calculation=js(out/'SECANT_CALCULATION.json')
    checks.add('secant calculation bound by prereg',pre['secant_calculation_sha256']==sha(out/'SECANT_CALCULATION.json'))
    for key in ('x115','x117','y115','y117','xstar'):
        checks.eq('runner independent secant '+key,calculation[key],secant[key])
    checks.eq('runner independent S113 target',calculation['targetS113'],secant['y113'])
    checks.add('runner secant valid no clipping',calculation['valid'] and calculation['same_peak_identity'] and calculation['descent_orientation'] and calculation['clipping'] is False)
    for key,index in [('S115',1),('S117',2)]:
        actual=secant['references'][index]['peak']; saved=calculation[key]
        for sk,ak in [('peak','value'),('sample_index','sample_index'),('physical_time_s','time_s'),('joint_name','joint'),('qpos_index','qpos_index'),('dof_index','dof_index')]:
            checks.add('runner independent '+key+' peak '+sk,saved[sk]==actual[ak])
    oldw=npz(S115/'TRAJECTORY.npz')['weights']; w117=npz(S117/'TRAJECTORY.npz')['weights']
    require(oldw.shape==(66,5), 'source W shape66x5')
    candidate=oldw.copy(); candidate[36,4]=new_value
    direction=npz(out/'DIRECTION.npz')
    for key,w in [('weights',oldw),('old_weights',oldw),('new_weights',candidate),(ARM,candidate)]:
        checks.eq('DIRECTION '+key,direction[key],w)
    mask=candidate!=oldw; expected=np.zeros_like(mask); expected[36,4]=True
    checks.eq('independent sole element mask',mask,expected)
    expected117=oldw.copy(); expected117[36,4]=secant['x117']
    checks.eq('S117 sole weight intervention',w117,expected117)
    checks.add('zero mask preserved',np.array_equal(oldw==0,candidate==0))
    checks.add('weight bounds',bool(np.all((candidate>=0)&(candidate<=4))))
    hashes=js(out/'WEIGHT_HASHES.json')
    for name,w in [('old',oldw),('new',candidate)]:
        digest=hashlib.sha256(np.ascontiguousarray(w,dtype=np.float64).tobytes()).hexdigest()
        checks.add(name+' independent weight bytes SHA',hashes[name+'_W_sha256']==digest and pre[name+'_W_sha256']==digest and str(direction[name+'_W_sha256'])==digest,observed_sha256=digest)
    checks.add('source weight file committed',canonical.get(str(Path(pre['same_weights']).resolve()))==sha(pre['same_weights']))
    checks.eq('source weight file exact',npz(pre['same_weights'])['weights'],oldw)
    items=[]
    required=('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz','NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','MODEL_PROVENANCE.json')
    arms=[('S113_dt25us_init40ms',S113,80,oldw),('S115_init80ms',S115,160,oldw),('S117_w36b4_down',S117,160,w117),(ARM,out/ARM,160,candidate)]
    for label,path,warm,w in arms:
        res=js(path/'RESULT.json'); canonical_checks(checks,res['hashes'],label+' artifacts',observed)
        committed={str(Path(p).resolve()) for p in res['hashes']}
        for name in required: checks.add(label+' committed '+name,str((path/name).resolve()) in committed)
        if label!=ARM:
            result_bound=canonical.get(str((path/'RESULT.json').resolve()))==sha(path/'RESULT.json')
            direct_bound=all(canonical.get(str((path/name).resolve()))==sha(path/name) for name in required)
            checks.add(label+' prereg reference commitment',result_bound or direct_bound)
        a=npz(path/'TRAJECTORY.npz'); init=npz(path/'INITIALIZATION.npz'); meta=js(path/'COM_CONTACT_METADATA.json'); moment=js(path/'MOMENT_METADATA.json')
        for key,value in a.items(): checks.add(label+' finite '+key,np.isfinite(value).all())
        for key,value in init.items(): checks.add(label+' finite warm '+key,np.isfinite(value).all())
        checks.eq(label+' time',a['time'],np.arange(121)*.000025,1e-12)
        for key,shape in dict(qpos=(121,49),qvel=(121,48),spikes=(6,17099),features=(6,66,5),applied=(6,90)).items(): checks.add(label+' shape '+key,a[key].shape==shape)
        for key,shape in dict(spikes=(warm,17099),features=(warm,66,5),decoded=(warm,90),history=(80,66)).items(): checks.add(label+' warm shape '+key,init[key].shape==shape)
        checks.eq(label+' runtime W',a['weights'],w); checks.eq(label+' W from cold',init['weights'],w)
        checks.eq(label+' mapping',a['actuator_pool'],mapping)
        checks.eq(label+' qpos release',a['qpos'][0],init['body_qpos'])
        for key in ('qvel','act','ctrl'): checks.add(label+' initial zero '+key,not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' zero externals warnings',not a['qfrc_applied'].any() and not a['xfrc_applied'].any() and not a['warnings'].any())
        checks.eq(label+' hold20',a['ctrl'][1:],np.repeat(a['applied'],20,axis=0))
        checks.eq(label+' fixed actual current',a['current'],np.repeat(init['current'][None],6,axis=0))
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
        if 'foot_order' in meta: checks.add(label+' foot identity order',meta['foot_order']==list(LEGS))
        checks.add(label+' actual actuator names',[r['actuator'] for r in rows]==moment['actuator_names'])
        sparse=np.zeros_like(a['actuator_moment'])
        for ti in range(121):
            for u in range(90):
                start=int(a['moment_rowadr'][ti,u]); size=int(a['moment_rownnz'][ti,u]); cols=a['moment_colind'][ti,start:start+size]
                require(len(set(cols.tolist()))==size and np.all((cols>=0)&(cols<48)),'unique valid moment CSR')
                sparse[ti,u,cols]=a['moment_raw'][ti,start:start+size]
        checks.eq(label+' independent CSR moment',a['actuator_moment'],sparse)
        checks.eq(label+' all90 signed torque sum',np.einsum('tu,tud->td',a['actuator_force'],sparse),a['qfrc_actuator'],2e-12)
        limitrows,limitcurve,peak=all42_limits(a,moment)
        checks.eq(label+' actual q all42 max excess',a['limit_excess'],limitcurve)
        rec=reconstruct(a,meta,checks,label); native=balance(a,rec)
        for key in ('instantaneous','impulse','momentum_FD'): checks.add(label+' native '+key,native[key]['passed'],details=native[key])
        weight=rec['mass']*abs(meta['gravity_world'][2]); checks.eq(label+' native weight',res['weight_native'],weight,1e-12)
        metrics=physical(a,rec,checks,label,weight,xmls[0]); ix=np.flatnonzero(a['time']<=.0017+1e-12)
        metrics.update(initial_foot_Fz=float(feet['foot_force'][0,:,2].sum()),initial_load_ratio=float(metrics['load_ratio'][0]),load_deficit_integral0_1p7_ratio_ms=float(np.trapezoid(np.maximum(.8-metrics['load_ratio'][ix],0),x=a['time'][ix])*1000),max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),max_downward_COM_velocity=float(np.maximum(-rec['velocity'][:,2],0).max()),all42_joint_limits=limitrows,all42_peak=peak,
                       Jload=float(np.trapezoid((np.maximum(.8-metrics['load_ratio'],0)/.8)**2,x=a['time'])/.003),Jlimit=float(np.trapezoid((np.maximum(limitcurve-1e-6,0)/.001)**2,x=a['time'])/.003))
        items.append(dict(label=label,path=str(path),arrays=a,initialization=init,derived=rec,feet=feet,physical=metrics,native=native,moment=moment))
    old,base,attenuated,new=items
    for ref in (base,attenuated):
        prefix=ref['label']+' exact '
        for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz'): same(checks,prefix+name,npz(Path(new['path'])/name),npz(Path(ref['path'])/name))
        for name in ('NEURAL_COLD_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json'): checks.add(prefix+name,js(Path(new['path'])/name)==js(Path(ref['path'])/name))
        for key in ('before','after'): checks.add(prefix+' model numeric '+key,js(Path(new['path'])/'MODEL_PROVENANCE.json')[key]==js(Path(ref['path'])/'MODEL_PROVENANCE.json')[key])
        for key in ('body_qpos','body_qvel','body_act','body_ctrl','current'): checks.eq(prefix+'initial '+key,new['initialization'][key],ref['initialization'][key])
        checks.eq(prefix+'runtime actual current',new['arrays']['current'],ref['arrays']['current'])
        for key in ('qpos','qvel','act','ctrl'): checks.eq(prefix+'body initial '+key,new['arrays'][key][0],ref['arrays'][key][0])
    neural={}; frozen={}
    for ref in (base,attenuated):
        label=ref['label']
        neural[label]=dict(warm=differences(new['initialization'],ref['initialization']),runtime=differences({k:new['arrays'][k] for k in ('spikes','features','applied')},{k:ref['arrays'][k] for k in ('spikes','features','applied')}),final=differences(npz(Path(new['path'])/'NEURAL_FINAL.npz'),npz(Path(ref['path'])/'NEURAL_FINAL.npz')),release_metadata_equal=js(Path(new['path'])/'NEURAL_RELEASE_META.json')==js(Path(ref['path'])/'NEURAL_RELEASE_META.json'),policy='Actual events/features/history/state measured, not equality-forced.')
        for stage,features,actual in [('warm',ref['initialization']['features'],new['initialization']['decoded']),('runtime',ref['arrays']['features'],new['arrays']['applied'])]:
            pred=mapped(features,candidate,mapping); currentfeatures=new['initialization']['features'] if stage=='warm' else new['arrays']['features']; equal=np.array_equal(features,currentfeatures)
            frozen[label+' '+stage]=dict(reference=label,stage=stage,features_equal=bool(equal),prediction=pred,actual=actual,actual_minus_frozen=actual-pred,max_abs_delta=float(np.abs(actual-pred).max()),policy='Frozen reference feature algebra only, not physical/neural trajectory prediction.')
            if equal: checks.eq(label+' '+stage+' frozen candidate clipped90 actual',actual,pred,1e-14)
    checks.add('warm decoded actually changed',not np.array_equal(new['initialization']['decoded'],base['initialization']['decoded']))
    return checks,items,dict(observed_hashes=observed,secant=secant,weight_change=dict(index=[36,4],old=float(oldw[36,4]),new=new_value,changed_elements=int(mask.sum())),source_audit=audit,ledger=ledger,neural_differences=neural,frozen_feature_predictions=frozen,decision=decision(old['physical'],base['physical'],new['physical'],not np.array_equal(new['arrays']['applied'],base['arrays']['applied'])))


def outputs(out,items,details):
    def save(name,header,data):
        path=out/name
        with path.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(data)
        return str(path)
    drive=[]; limits=[]; curves=[]; torques=[]; predictions=[]; decomposition=[]
    for item in items:
        a=item['arrays']; init=item['initialization']; label=item['label']; mapping=a['actuator_pool']; w=a['weights']
        for phase,features,control in [('warm',init['features'],init['decoded']),('runtime',a['features'],a['applied'])]:
            for k,(f,u) in enumerate(zip(features,control)):
                raw=(f*w).sum(1)
                for act,pool in enumerate(mapping): drive.append([label,phase,k,act,int(pool),*([0.]*5 if pool<0 else f[pool]),*([0.]*5 if pool<0 else w[pool]),0 if pool<0 else raw[pool],u[act]])
        for row in item['physical']['all42_joint_limits']: limits.append([label,row['joint'],row['qpos_index'],row['dof_index'],*row['range'],row['min_q'],row['max_q'],row['max_excess'],row['max_excess_at_ms']])
        for k,t in enumerate(a['time']):
            p=item['physical']; curves.append([label,t*1000,p['load_ratio'][k],*item['feet']['foot_force'][k,:,2],*item['derived']['momentum'][k],item['derived']['velocity'][k,2],a['limit_excess'][k],*p['gates'][k]])
            for d in (6,7,8): torques.append([label,t*1000,2,item['moment']['actuator_names'][2],d,a['ctrl'][k,2],a['act'][k,2],a['actuator_force'][k,2],a['actuator_moment'][k,2,d],a['actuator_force'][k,2]*a['actuator_moment'][k,2,d],a['qfrc_actuator'][k,d]])
    for row in details['frozen_feature_predictions'].values():
        for k,(pred,actual) in enumerate(zip(row['prediction'],row['actual'])):
            for act in range(90): predictions.append([row['reference'],row['stage'],k,act,row['features_equal'],pred[act],actual[act],actual[act]-pred[act]])
    new=items[-1]['arrays']
    for ref in items[:-1]:
        base=ref['arrays']
        for k,t in enumerate(new['time']):
            for d in (6,7,8):
                f0=base['actuator_force'][k,2]; df=new['actuator_force'][k,2]-f0; m0=base['actuator_moment'][k,2,d]; dm=new['actuator_moment'][k,2,d]-m0
                decomposition.append([ref['label'],t*1000,d,f0,df,m0,dm,df*m0,f0*dm,df*dm,new['actuator_force'][k,2]*new['actuator_moment'][k,2,d]-f0*m0])
    paths=[save('SECANT_WEIGHT_WARM_RUNTIME_WEIGHTS.csv',['arm','phase','tick','actuator','pool',*[f'feature{k}' for k in range(5)],*[f'weight{k}' for k in range(5)],'raw_drive','actual_clipped90'],drive),save('SECANT_WEIGHT_PREDICTED_ACTUAL.csv',['reference','phase','tick','actuator','neural_features_equal','frozen_feature_candidate_prediction','actual','actual_minus_predicted'],predictions),save('SECANT_WEIGHT_ALL42_LIMITS.csv',['arm','joint','qpos_index','dof_index','lower','upper','min_q','max_q','max_excess','at_ms'],limits),save('SECANT_WEIGHT_ACT2_SIGNED_MOMENTS.csv',['arm','physical_ms','actuator','name','DOF','ctrl','activation','actual_signed_force','signed_moment','actual_signed_torque','aggregate_signed_torque'],torques),save('SECANT_WEIGHT_ACT2_DECOMPOSITION.csv',['reference','physical_ms','DOF','reference_force','delta_force','reference_moment','delta_moment','force_term','moment_term','cross_term','actual_torque_delta'],decomposition),save('SECANT_WEIGHT_COMPARISON.csv',['arm','physical_ms','load_ratio',*[l+'_Fz' for l in LEGS],'COM_px','COM_py','COM_pz','COM_vz','all42_max_limit_excess','height_gate','angle_gate','legs_gate','load_gate','nonfoot_gate','limit_gate'],curves)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    for item in items:
        a=item['arrays']; t=a['time']*1000; label=item['label']
        axes[0,0].step(np.arange(7)*.5,np.r_[a['applied'][:,2],a['applied'][-1,2]],where='post',label=label)
        for d in (6,7,8): axes[0,1].plot(t,a['actuator_force'][:,2]*a['actuator_moment'][:,2,d],label=label+' DOF'+str(d))
        axes[1,0].plot(t,item['physical']['load_ratio'],label=label)
        axes[1,1].plot(t,a['limit_excess'],label=label)
        axes[2,0].plot(t,item['derived']['momentum'][:,2],label=label)
        axes[2,1].plot(t,item['derived']['velocity'][:,2],label=label)
    for ax,title in zip(axes.flat,['Act2 actual command (all90 in CSV)','Act2 actual signed force x moment DOF6/7/8','Six-foot support Fz / weight','Actual q all42 maximum joint limit excess','Whole-body COM vertical momentum','Whole-body COM vertical velocity']):
        ax.set_title(title); ax.set_xlabel('physical ms'); ax.legend(fontsize=6); ax.grid(alpha=.25); ax.axvline(1.7,color='gray',ls=':')
    axes[1,0].axhline(.8,color='black',ls=':'); axes[1,1].axhline(1e-6,color='black',ls=':')
    fig.suptitle('S118 exact secant W[36,4]: S113 / S115 / S117 / candidate\nOffline saved-array check; 3ms diagnostic only; units NOT_ESTABLISHED; no checkpoint or walking claim')
    fig.tight_layout(); png=out/'SECANT_WEIGHT_COMPARISON.png'; fig.savefig(png,dpi=150); plt.close(fig)
    return dict(comparison_png=str(png),tables=paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT); out=parser.parse_args().output.resolve()
    report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')
    try:
        checks,items,details=verify(out); generated=outputs(out,items,details)
        report.update(details,**generated,passed=checks.passed,checks=checks.rows,per_arm={i['label']:{k:i[k] for k in ('path','physical','native')} for i in items})
        require(not any(k in sys.modules for k in ('mujoco','torch')),'physics/neural imported')
    except Exception as exc: report.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'SECANT_WEIGHT_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),report=str(out/'SECANT_WEIGHT_INDEPENDENT_CHECK.json')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
