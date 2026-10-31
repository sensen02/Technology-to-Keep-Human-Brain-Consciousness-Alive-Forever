"""S114/P1/S2-S4 pure offline, independent initial-height evidence check.

No simulator, neural engine, runner import, threshold adaptation, or new physics.
Native primary force/impulse/FD criteria are frozen; 100us diagnostics are not gates.
"""
from pathlib import Path
from collections import Counter
import argparse
import ast
import csv
import json
import sys
import traceback
import numpy as np
from verify_com_contact_probe import Checks, require, sha, js, npz, serial, write_json, reconstruct_fir
from verify_com_timestep_probe import canonical_checks, reconstruct, measure, physical

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us')
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s114-initial-height/20261024T061500Z')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')


def raw_feet(a, meta):
    """Rebuild old runner's tars naming and floor-force convention, not saved foot totals."""
    names = meta['geom_names']; bodyids = np.asarray(meta['geom_bodyid'])
    foot = {}
    for g, gn in enumerate(names):
        gn = (gn or '').lower(); bn = (meta['body_names'][bodyids[g]] or '').lower()
        for li, leg in enumerate(LEGS):
            if 'tars' in gn+bn and (gn.startswith(leg.lower()) or bn.startswith(leg.lower())):
                foot[g] = li
    n = len(a['time']); cap = a['contact_force_raw6'].shape[1]
    frames = a['contact_frame'].reshape(n, cap, 3, 3)
    f = np.einsum('tcji,tcj->tci', frames, a['contact_force_raw6'][:, :, :3])
    floors = set(meta['floor_geom_ids'])
    loads = np.zeros((n, 6, 3)); counts = np.zeros((n, 6), dtype=int)
    nonfoot = np.zeros(n); selected = []; pairs = []
    for t in range(n):
        for c in range(int(a['contact_ncon'][t])):
            g1, g2 = map(int, a['contact_geom_ids'][t, c]); b1, b2 = map(int, a['contact_body_ids'][t, c])
            if t == 0: pairs.append(tuple(sorted((g1, g2))))
            # Existing runner only accumulates explicit floor-plane contacts.
            if (g1 in floors) == (g2 in floors): continue
            ground_first = g1 in floors; g = g2 if ground_first else g1
            signed = f[t, c] * (1 if ground_first else -1)
            dist = float(a['contact_dist'][t, c])
            if g in foot:
                li = foot[g]; loads[t, li] += signed; counts[t, li] += int(dist < 0)
                if t == 0 and (b1 == 0) != (b2 == 0) and signed[2] > 0 and dist < 0:
                    normal = frames[t, c, 0] * (1 if ground_first else -1)
                    selected.append(dict(contact_index=c, geom_ids=[g1, g2], body_ids=[b1, b2],
                        leg=LEGS[li], depth=-dist, signed_world_force=signed.tolist(),
                        normal_toward_foot=normal.tolist(), point_world=a['contact_pos_world'][t, c].tolist()))
            else: nonfoot[t] = max(nonfoot[t], -dist)
    return dict(foot_force=loads, contacts=counts, nonfoot_depth=nonfoot, selected=selected,
                initial_pair_multiset=[dict(geom_ids=list(p), count=k) for p, k in sorted(Counter(pairs).items())],
                initial_ncon=int(a['contact_ncon'][0]), geom_leg={str(g):LEGS[l] for g,l in foot.items()})


def derive_height(a, meta):
    feet = raw_feet(a, meta); selected = feet['selected']
    require(bool(selected), 'no actual carrying penetrating initial external foot-ground contacts')
    require(all(np.array_equal(s['normal_toward_foot'], [0., 0., 1.]) for s in selected), 'selected normals must be +z')
    return .5 * min(s['depth'] for s in selected), feet


def differences(a, b):
    rows = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            rows[key] = dict(equal=False, missing_in='current' if key not in a else 'reference'); continue
        x, y = np.asarray(a[key]), np.asarray(b[key]); same = x.shape == y.shape
        row = dict(equal=bool(same and np.array_equal(x,y)), reference_shape=list(y.shape), current_shape=list(x.shape))
        if same:
            row['different_elements'] = int(np.count_nonzero(x != y))
            row['max_abs_delta'] = float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
        rows[key] = row
    return rows


def balance(a, r, stride=1):
    ix=np.arange(0,len(a['time']),stride); t=a['time'][ix]; p=r['momentum'][ix]; f=r['external_force'][ix]
    inclusive=np.flatnonzero(t<=.0017+1e-12)
    ids=np.flatnonzero((t<=.0017+1e-12)&(np.arange(len(t))>0)&(np.arange(len(t))<len(t)-1))
    fd=np.gradient(p,t,axis=0,edge_order=2)
    instantaneous=measure(r['ma'][ix],f,np.arange(len(t)),1e-8,1e-6)
    impulse=np.vstack((np.zeros(3),np.cumsum(.5*(f[1:]+f[:-1])*np.diff(t)[:,None],axis=0)))
    im=measure(impulse,p-p[0],inclusive,1e-4,.01)
    im['global_denominator']=max(float(r['mass']*np.linalg.norm(r['velocity'][ix][inclusive],axis=1).max()),1e-12)
    im['relative_residual']=im['max_abs_residual']/im['global_denominator']
    im['passed']=bool(im['max_abs_residual']<=1e-4 and im['relative_residual']<=.01)
    im['denominator_definition']='mass * max norm(COM velocity), inclusive0..1.7ms, floor1e-12'
    momentumfd=measure(fd,f,ids,1.,.05); w=momentumfd['worst_index']
    momentumfd.update(worst_time_ms=float(t[w]*1000), original_native_sample_index=int(ix[w]),
        native_stencil_indices=ix[w-1:w+2].tolist(), stencil_times_s=t[w-1:w+2].tolist(),
        stencil_momentum=p[w-1:w+2], fd_force=fd[w], external_force=f[w],
        method='np.gradient full original momentum/time edge_order=2 BEFORE slicing; centered indices1..1.7ms inclusive')
    return dict(instantaneous=instantaneous,impulse=im,momentum_FD=momentumfd,
                time_ms=t*1000,fd_residual=fd-f,native_indices=ix)


def pose_source_check(path, checks):
    """Source evidence only: saved endpoints cannot rule out unrecorded pose writes."""
    tree=ast.parse(path.read_text()); writes=[]
    def is_qpos(node):
        return (isinstance(node,ast.Attribute) and node.attr=='qpos') or (isinstance(node,ast.Subscript) and is_qpos(node.value))
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign)) else []
        for target in targets:
            if is_qpos(target): writes.append(dict(line=node.lineno,source=ast.unparse(node)))
    checks.add('runner sole qpos assignment source evidence',len(writes)==1,assignments=writes,
               boundary='hash-bound source audit, not independent proof of all runtime writes')
    return writes


def decision(reference, current):
    """Conservative predeclared rejection; improvement requires support AND nonregressing safety."""
    earlier=[]
    for gate,old in reference['first_failures_ms'].items():
        new=current['first_failures_ms'][gate]
        if new is not None and (old is None or new<old): earlier.append(gate)
    safety=[]
    for key in ('angle_rad','nonfoot_depth','limit_excess'):
        if current['extrema'][key]['max']>reference['extrema'][key]['max']: safety.append(key)
    for key in ('max_downward_root_velocity','max_downward_COM_velocity'):
        if current[key]>reference[key]: safety.append(key)
    for key in ('min_height_own_initial','min_height_reference_original'):
        if current[key]<reference[key]: safety.append(key)
    lower=current['initial_foot_Fz']<reference['initial_foot_Fz']
    safety_ok=not safety and not earlier
    support=(current['all_gates_support_ms']>reference['all_gates_support_ms'] and
             current['longest_all_gates_true_endpoint_span_ms']>=reference['longest_all_gates_true_endpoint_span_ms'])
    reject=lower or bool(earlier) or bool(safety)
    return dict(reject=bool(reject),lower_initial_load=bool(lower),earlier_failure_gates=earlier,
                safety_regressions=safety,safety_nonregression=safety_ok,support_improvement=support,
                suggest_candidate=bool(not reject and support and safety_ok),
                rule='reject lower initial load OR any earlier gate failure OR any safety regression; suggest requires support improvement AND safety nonregression',
                standing_or_walking_claim=False,deployment=False,checkpoint_replacement=False)


def verify(out):
    require(not any(k in sys.modules for k in ('mujoco','torch')), 'physics/neural loaded')
    checks=Checks(); observed={}; pre=js(out/'PREREG.json'); sources=pre['source_hashes']
    canonical_checks(checks,sources,'source',observed)
    checks.add('INPUT_HASHES agrees',js(out/'INPUT_HASHES.json')==sources)
    deps=js(out/'MODEL_DEPENDENCIES.json'); canonical_checks(checks,deps,'dependencies',observed)
    checks.add('dependencies committed',all(sources.get(p)==d for p,d in deps.items()))
    canonical_sources={str(Path(p).resolve()):d for p,d in sources.items()}
    runner=ROOT/'run_initial_height_probe.py'
    checks.add('runner source committed',canonical_sources.get(str(runner.resolve()))==sha(runner))
    writes=pose_source_check(runner,checks)
    checks.add('single height_up arm',pre['arms']==['height_up'])
    checks.add('fixed timing',pre['initialization_ticks']==80 and pre['physical_duration_ms']==3 and pre['neural_dt_ms']==.5 and pre['physical_neural_ticks_per_arm']==6)
    checks.add('fixed native timestep and substeps',pre['timesteps_s']=={'height_up':.000025} and pre['substeps']=={'height_up':20})
    checks.add('fixed sample and step counts',pre['physical_samples_per_arm']=={'height_up':121} and pre['physical_steps_per_arm']=={'height_up':120})
    ledger=js(out/'CALL_LEDGER.json'); expected=dict(model_load=1,forward=121,step=120,neural_ticks=86)
    checks.add('exact single-arm ledger',ledger==dict(expected,active=None,completed=['height_up'],per_arm={'height_up':expected}),observed=ledger)
    checks.add('complete no failed',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    oldpre=js(REFERENCE.parent/'PREREG.json'); oldres=js(REFERENCE/'RESULT.json')
    newres=js(out/'height_up'/'RESULT.json'); results=js(out/'RESULTS.json')
    checks.add('one result',len(results)==1 and results[0]==newres and newres['label']=='height_up')
    canonical_checks(checks,oldres['hashes'],'reference artifacts',observed)
    canonical_checks(checks,newres['hashes'],'new artifacts',observed)
    for name in ('TRAJECTORY.npz','COM_CONTACT_METADATA.json','INITIALIZATION.npz','MODEL_PROVENANCE.json'):
        checks.add('reference source committed '+name,canonical_sources.get(str((REFERENCE/name).resolve()))==sha(REFERENCE/name))
    olda=npz(REFERENCE/'TRAJECTORY.npz'); oldmeta=js(REFERENCE/'COM_CONTACT_METADATA.json')
    dz,oldfeet=derive_height(olda,oldmeta)
    plan=pre['height_plan']
    checks.eq('preregistered exact offline dz',plan['dz'],dz)
    checks.eq('preregistered old initial qpos',plan['initial_qpos_old'],olda['qpos'][0])
    expected_plan_q=olda['qpos'][0].copy(); expected_plan_q[2]+=dz
    checks.eq('preregistered new initial qpos',plan['initial_qpos_new'],expected_plan_q)
    checks.add('independently selected exact contact IDs',plan['selected_contact_ids']==[s['contact_index'] for s in oldfeet['selected']])
    for independent, saved in zip(oldfeet['selected'],plan['selected_contacts']):
        ci=independent['contact_index']
        for key in ('geom_ids','body_ids'): checks.eq('selected '+str(ci)+' '+key,saved[key],independent[key])
        checks.eq('selected '+str(ci)+' force',saved['force_world_on_fly'],independent['signed_world_force'])
        checks.eq('selected '+str(ci)+' depth',saved['positive_depth'],independent['depth'])
        checks.add('selected '+str(ci)+' leg identity',saved['leg']==independent['leg'])
    reference_commit=js(out/'REFERENCE_ARTIFACTS.json')
    canonical_checks(checks,reference_commit['artifacts'],'retained reference',observed)
    xmls=[Path(p) for p in sources if Path(p).suffix=='.xml']; require(len(xmls)==1,'one XML')
    manifests=[Path(p) for p in sources if Path(p).name=='MANIFEST.json']; require(len(manifests)==1,'one actuator manifest')
    rows=sorted(js(manifests[0])['rows'],key=lambda x:x['actuator_index'])
    pools=sorted({tuple(sorted(row['local_indices'])) for row in rows if row['local_indices']}); lookup={p:i for i,p in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(row['local_indices'])),-1) for row in rows])
    w=npz(pre['same_weights'])['weights']; checks.eq('same S113 source W',w,npz(oldpre['same_weights'])['weights'])
    direction=npz(out/'DIRECTION.npz'); checks.eq('direction source W',direction['weights'],w); checks.eq('direction height W',direction['height_up'],w)
    checks.eq('direction exact dz',direction['dz'],dz)
    checks.eq('direction original initial q',direction['initial_qpos_old'],olda['qpos'][0])
    checks.eq('direction new initial q',direction['initial_qpos_new'],expected_plan_q)
    items=[]; differences_report={}
    for label,path,res in [('dt25us_reference',REFERENCE,oldres),('height_up',out/'height_up',newres)]:
        a=npz(path/'TRAJECTORY.npz'); init=npz(path/'INITIALIZATION.npz'); meta=js(path/'COM_CONTACT_METADATA.json')
        checks.eq(label+' exact clock121',a['time'],np.arange(121)*.000025,1e-12)
        for key,value in a.items(): checks.add(label+' finite '+key,np.isfinite(value).all())
        for key in ('spikes','features','applied','current'): checks.add(label+' six neural ticks '+key,len(a[key])==6)
        checks.eq(label+' W',a['weights'],w); checks.eq(label+' initialization W',init['weights'],w)
        checks.eq(label+' actuator pool',a['actuator_pool'],mapping)
        checks.eq(label+' initial q',a['qpos'][0],init['body_qpos'])
        for key in ('qvel','act','ctrl'): checks.add(label+' zero initial '+key,not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' no warnings',not a['warnings'].any())
        checks.eq(label+'20step hold',a['ctrl'][1:],np.repeat(a['applied'],20,axis=0))
        checks.eq(label+' held actual current',a['current'],np.repeat(init['current'][None],6,axis=0))
        fs,us,history=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,w)
        _,_,warmhistory=reconstruct_fir(init['spikes'],pools,mapping,w)
        for name,x,y in [('warm features',init['features'],fs[:80]),('warm decoded',init['decoded'],us[:80]),('release features',init['release_features'],fs[79]),('release decoded',init['release_decoded'],us[79]),('release history',init['history'],warmhistory),('features',a['features'],fs[80:]),('applied',a['applied'],us[80:]),('final history',npz(path/'NEURAL_FINAL.npz')['FIR_history'],history)]: checks.eq(label+' FIR '+name,x,y,1e-14)
        schedule=js(path/'SCHEDULE.json'); checks.add(label+' schedule six',len(schedule)==6)
        for k,row in enumerate(schedule):
            checks.add(label+' schedule ids '+str(k),row['tick']==k and row['consume_id']==80+k)
            for key,value in dict(source_start_s=(80+k)*.0005,source_end_s=(81+k)*.0005,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=.04).items(): checks.eq(label+' schedule '+str(k)+' '+key,row[key],value,1e-12)
        for name,steps,ms in [('NEURAL_COLD_META.json',0,0),('NEURAL_RELEASE_META.json',80,40),('NEURAL_FINAL_META.json',86,43)]:
            scalars=js(path/name)['neural_scalars']; checks.add(label+' clock '+name,scalars['steps']==steps and abs(scalars['t_ms']-ms)<=1e-10)
        feet=raw_feet(a,meta)
        for key in ('foot_force','contacts','nonfoot_depth'): checks.eq(label+' raw independent '+key,a[key],feet[key],2e-12)
        rec=reconstruct(a,meta,checks,label); native=balance(a,rec); coarse=balance(a,rec,4)
        for key in ('instantaneous','impulse','momentum_FD'): checks.add(label+' native '+key,native[key]['passed'],details=native[key])
        weight=rec['mass']*abs(meta['gravity_world'][2]); checks.eq(label+' weight',res['weight_native'],weight,1e-12)
        metrics=physical(a,rec,checks,label,weight,xmls[0]); t=a['time']; ix=np.flatnonzero(t<=.0017+1e-12)
        metrics.update(initial_foot_Fz=float(feet['foot_force'][0,:,2].sum()),initial_leg_Fz=feet['foot_force'][0,:,2],
            six_leg_impulse0_1p7=np.trapezoid(feet['foot_force'][ix,:,2],x=t[ix],axis=0),
            COM_momentum_samples={str(ms):rec['momentum'][int(round(ms/.025))] for ms in (0,1.7,3)},
            min_height_own_initial=float((a['qpos'][:,2]/a['qpos'][0,2]).min()),
            min_height_reference_original=float((a['qpos'][:,2]/olda['qpos'][0,2]).min()),
            max_downward_root_velocity=float(np.maximum(-a['qvel'][:,2],0).max()),
            max_downward_COM_velocity=float(np.maximum(-rec['velocity'][:,2],0).max()))
        items.append(dict(label=label,path=str(path.resolve()),arrays=a,initialization=init,derived=rec,feet=feet,physical=metrics,native=native,downsample100us_diagnostic_only=coarse))
    old,new=items
    expectedq=old['arrays']['qpos'][0].copy(); expectedq[2]+=dz
    checks.eq('exact dz and only qpos2 changed',new['arrays']['qpos'][0],expectedq)
    checks.add('only qpos2 changed index',np.flatnonzero(new['arrays']['qpos'][0]!=old['arrays']['qpos'][0]).tolist()==[2])
    oldmeta=js(REFERENCE/'COM_CONTACT_METADATA.json'); newmeta=js(out/'height_up'/'COM_CONTACT_METADATA.json')
    checks.add('all original metadata unchanged',all(newmeta.get(k)==v for k,v in oldmeta.items()))
    checks.add('only explicit foot identity metadata added',set(newmeta)-set(oldmeta)=={'foot_order','foot_geom_leg_map'})
    checks.add('explicit foot order matches independent mapping',newmeta['foot_order']==list(LEGS))
    checks.add('identical model all numeric config',js(REFERENCE/'MODEL_PROVENANCE.json')['after']==js(out/'height_up'/'MODEL_PROVENANCE.json')['after'])
    for name in ('CONNECTIVITY.npz','NEURAL_COLD.npz'):
        dif=differences(npz(out/'height_up'/name),npz(REFERENCE/name)); checks.add('same '+name,all(r['equal'] for r in dif.values()))
    for name in ('NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json'):
        # Scalar configuration/clock equality is legitimate; actual neural arrays below may differ.
        checks.add('same neural scalar config '+name,js(out/'height_up'/name)==js(REFERENCE/name))
    differences_report['INITIALIZATION.npz']=differences(new['initialization'],old['initialization'])
    differences_report['warm_and_release_pool_detail']={}
    for key in ('features','release_features','history','release_decoded','decoded','current','spikes'):
        x=new['initialization'][key]; y=old['initialization'][key]
        detail=dict(delta=x.astype(float)-y.astype(float),different_indices=np.argwhere(x!=y))
        if key=='features': detail['different_pool_ids']=np.flatnonzero(np.any(x!=y,axis=(0,2)))
        if key=='release_features': detail['different_pool_ids']=np.flatnonzero(np.any(x!=y,axis=1))
        differences_report['warm_and_release_pool_detail'][key]=detail
    differences_report['interpretation']='Height changes actual contact-dependent sensory current; report actual warm/release/runtime differences. This is not a pure geometry isolation when neural currents/states differ; neural differences alone are NOT a rejection gate.'
    differences_report['NEURAL_FINAL.npz']=differences(npz(out/'height_up'/'NEURAL_FINAL.npz'),npz(REFERENCE/'NEURAL_FINAL.npz'))
    differences_report['runtime']=differences({k:new['arrays'][k] for k in ('current','spikes','features','applied')},{k:old['arrays'][k] for k in ('current','spikes','features','applied')})
    for name in ('INITIAL_OBSERVATION.json','OBSERVATION_00.json'):
        differences_report[name]=dict(equal=js(out/'height_up'/name)==js(REFERENCE/name),reference=js(REFERENCE/name),current=js(out/'height_up'/name))
    contact_comparison=dict(reference_pairs=old['feet']['initial_pair_multiset'],new_pairs=new['feet']['initial_pair_multiset'],
        same_pair_multiset=old['feet']['initial_pair_multiset']==new['feet']['initial_pair_multiset'],
        reference_count=old['feet']['initial_ncon'],new_count=new['feet']['initial_ncon'],
        point_policy='points reported, not required equal: raising qpos2 moves contact points',
        reference_points=old['arrays']['contact_pos_world'][0,:old['feet']['initial_ncon']],
        new_points=new['arrays']['contact_pos_world'][0,:new['feet']['initial_ncon']])
    require(not any(k in sys.modules for k in ('mujoco','torch')), 'physics/neural imported')
    return checks,items,dict(dz_independent=dz,selected_reference_contacts=oldfeet['selected'],source_pose_writes=writes,
        observed_hashes=observed,neural_release_differences=differences_report,initial_contact_comparison=contact_comparison,
        decision=decision(old['physical'],new['physical']),ledger=ledger)


def outputs(out,items,checks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,13))
    rows=[]
    for item in items:
        a=item['arrays']; p=item['physical']; r=item['derived']; t=a['time']*1000; label=item['label']
        axes[0,0].plot(t,p['load_ratio'],label=label)
        axes[0,1].plot(t,a['qpos'][:,2]/a['qpos'][0,2],label=label+' own initial')
        axes[0,1].plot(t,a['qpos'][:,2]/items[0]['arrays']['qpos'][0,2],ls='--',label=label+' reference initial')
        axes[1,0].plot(t,r['velocity'][:,2],label=label)
        axes[1,1].plot(t,a['limit_excess'],label=label+' limit')
        axes[2,0].plot(t,np.max(np.abs(r['force_residual']),axis=1),label=label)
        b=item['native']; ids=b['momentum_FD']['sample_indices']
        axes[2,1].plot(t[ids],np.max(np.abs(b['fd_residual'][ids]),axis=1),label=label)
        for i,ms in enumerate(t):
            rows.append([label,ms,p['load_ratio'][i],a['qpos'][i,2]/a['qpos'][0,2],a['qpos'][i,2]/items[0]['arrays']['qpos'][0,2],r['velocity'][i,2],a['limit_excess'][i],a['nonfoot_depth'][i],*item['feet']['foot_force'][i,:,2],*r['momentum'][i]])
    for ax,title in zip(axes.flat,['Actual six-foot Fz / weight','Root height: own / original reference initial','COM vertical velocity','Joint limit excess','Raw instantaneous COM force residual','Native momentum FD residual (original centered indices)']):
        ax.set_title(title); ax.set_xlabel('physical ms'); ax.grid(alpha=.25); ax.legend(fontsize=7); ax.axvline(1.7,color='grey',ls=':')
    axes[0,0].axhline(.8,color='black',ls=':'); axes[2,0].set_yscale('symlog',linthresh=1e-15)
    fig.suptitle('S114 initial-height diagnostic: offline independent raw evidence\nNo physics rerun, threshold adaptation, support/walking claim; native units NOT_ESTABLISHED')
    fig.tight_layout(); png=out/'HEIGHT_COMPARISON.png'; fig.savefig(png,dpi=150); plt.close(fig)
    csvpath=out/'HEIGHT_COMPARISON.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['arm','time_ms','load_ratio','height_own_initial','height_reference_original','COM_vz','limit_excess','nonfoot_depth',*[leg+'_Fz' for leg in LEGS],'COM_px','COM_py','COM_pz']); writer.writerows(rows)
    with (out/'HEIGHT_ALL_CHECKS.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['name','passed','details_json'])
        for row in checks.rows: writer.writerow([row['name'],row['passed'],json.dumps(row,default=serial,allow_nan=False)])
    return dict(comparison_png=str(png),comparison_csv=str(csvpath),checks_csv=str(out/'HEIGHT_ALL_CHECKS.csv'))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT)
    out=parser.parse_args().output.resolve()
    report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')
    try:
        checks,items,details=verify(out)
        report.update(details,passed=checks.passed,checks=checks.rows,per_arm={item['label']:{key:item[key] for key in ('path','physical','native','downsample100us_diagnostic_only')} for item in items},**outputs(out,items,checks))
    except Exception as exc: report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'HEIGHT_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),report=str(out/'HEIGHT_INDEPENDENT_CHECK.json')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
