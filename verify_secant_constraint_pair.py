"""S122/P1/S2 offline S118 replay and S121 constraint-row pair verification.
Only saved arrays, JSON, hashes and static AST are inspected; no runner imports.
No parameter proposal, causal attribution, model load, forward, step or neural tick.
"""
from pathlib import Path
import argparse
import ast
import csv
import json
import sys
import traceback
import numpy as np
from verify_constraint_row_recording import (Checks, sha, js, npz, serial, exact,
    compare_npz, bound_hashes, jacobian, CATEGORIES, ENUM_CATEGORY)
from verify_secant_weight_probe import all42_limits

ROOT = Path(__file__).resolve().parent
ARM = 'w36b4_secant'
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s122-secant-constraint-pair/20261024T073500Z/w36b4_secant')
HISTORICAL = Path('/home/sensen/Desktop/engine修复验收/s118-secant-weight-step/20261024T065500Z/w36b4_secant')
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
NPZS = ('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz')
JSONS = ('NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','MODEL_PROVENANCE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json')


def canonical(hashes):
    result = {}
    for name, digest in hashes.items():
        key = str(Path(name).resolve())
        if key in result and result[key] != digest:
            raise ValueError('conflicting canonical commitments: '+key)
        result[key] = digest
    return result


def committed(checks, hashes, path, label):
    path = Path(path).resolve()
    checks.add(label+' hash commitment '+str(path), canonical(hashes).get(str(path)) == sha(path))


def audit_sources(checks, hashes):
    paths = [ROOT/'run_secant_constraint_recording.py', ROOT/'run_secant_weight_probe.py', ROOT/'run_constraint_row_recording.py']
    for path in paths: committed(checks, hashes, path, 'actual canonical source')
    wrapper = ast.parse(paths[0].read_text())
    calls = []; writes = []
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.endswith(('from_xml_path','mj_forward','mj_step','mj_resetData','mj_resetDataKeyframe')) or name == 'net.step': calls.append(name)
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign)) else []
        for target in targets:
            name = ast.unparse(target)
            if 'qpos' in name or name.startswith(('d.','m.','net.')): writes.append(ast.unparse(node))
    checks.add('wrapper introduces no explicit physics/neural calls or state writes', not calls and not writes, calls=calls, writes=writes)
    recorder = ast.parse(paths[2].read_text()); found=[]
    for node in recorder.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'constraint_record':
            found.append(node.name); bad=[]
            for child in ast.walk(node):
                if isinstance(child,ast.Call) and any(s in ast.unparse(child.func) for s in ('mj_forward','mj_step','from_xml','net.step','mj_jac','mj_mul','mj_contactForce')): bad.append(ast.unparse(child))
                targets=child.targets if isinstance(child,ast.Assign) else [child.target] if isinstance(child,(ast.AugAssign,ast.AnnAssign)) else []
                for target in targets:
                    if ast.unparse(target).startswith(('d.','m.','net.')): bad.append(ast.unparse(child))
            checks.add('imported S121 constraint_record passive',not bad,forbidden=bad)
    checks.add('actual S121 recorder exists',found==['constraint_record'])
    return dict(wrapper=str(paths[0]),original=str(paths[1]),passive_helpers=str(paths[2]),boundary='Static source audit plus actual ledger; not proof against unrecorded runtime events.')


def reconstruct_rows(checks, path, label):
    """Parameterised S121 actual-enum/address/CSR row partition, no residual bucket."""
    a=npz(path/'TRAJECTORY.npz'); z=npz(path/'CONSTRAINT_ROWS.npz')
    meta=js(path/'CONSTRAINT_ROW_METADATA.json'); moment=js(path/'MOMENT_METADATA.json')
    enums={k:int(v) for k,v in meta['constraint_enum'].items()}; inverse={v:k for k,v in enums.items()}
    cone=int(meta['cone']); cone_enum=meta['cone_enum']; nv=int(np.asarray(z['nv']).flat[0])
    checks.add(label+' actual cone recognized',cone in [int(v) for v in cone_enum.values()])
    checks.add(label+' all121 all48',nv==48 and len(a['time'])==121 and len(z['nefc'])==121)
    checks.add(label+' original same instant',exact(z['time'],a['time'])['identical'])
    projections=[]; mapping=[]; states=[]; counts={c:0 for c in CATEGORIES}
    for i in range(121):
        p=f'sample{i:03d}_'; tag=label+' '+p; n=int(z['nefc'][i])
        sparse=bool(np.asarray(z['is_sparse']).flat[0 if np.asarray(z['is_sparse']).size==1 else i])
        J=jacobian(z,p,n,nv,sparse,checks)
        types=z[p+'efc_type']; ids=z[p+'efc_id']; force=z[p+'efc_force']
        checks.add(tag+' valid row vectors',types.shape==ids.shape==force.shape==(n,) and np.isfinite(force).all() and np.isfinite(J).all())
        lo,hi=map(int,z['row_offsets'][i:i+2]); checks.add(tag+' exact ragged count',hi-lo==n)
        for key,value in [('efc_type',types),('efc_id',ids),('efc_force',force)]: checks.add(tag+' flattened '+key,exact(z[key][lo:hi],value)['identical'])
        addresses=z[p+'contact_efc_address']; dims=z[p+'contact_dim']; geoms=z[p+'contact_geom']; owners={}
        checks.add(tag+' contact shapes',len(addresses)==len(dims)==len(geoms) and geoms.shape==(len(addresses),2))
        for ci,(address,dim) in enumerate(zip(addresses,dims)):
            start,dim=int(address),int(dim)
            checks.add(tag+' valid geom '+str(ci),np.all((geoms[ci]>=0)&(geoms[ci]<int(meta['ngeom']))))
            if start==-1: continue
            if dim==1: enum='mjCNSTR_CONTACT_FRICTIONLESS'; size=1
            elif cone==int(cone_enum['mjCONE_PYRAMIDAL']): enum='mjCNSTR_CONTACT_PYRAMIDAL'; size=2*(dim-1)
            else: enum='mjCNSTR_CONTACT_ELLIPTIC'; size=dim
            valid=dim in (1,3,4,6) and start>=0 and start+size<=n
            checks.add(tag+' actual contact address '+str(ci),valid,address=start,dim=dim)
            if not valid: raise ValueError(tag+' invalid contact address')
            for r in range(start,start+size):
                checks.add(tag+' actual type/id/contact '+str(r),int(types[r])==enums[enum] and int(ids[r])==ci and r not in owners)
                owners[r]=ci
        category=np.zeros((len(CATEGORIES),nv)); rowproj=J*force[:,None]; statecounts={c:0 for c in CATEGORIES}; active_limits=[]; force_limits=[]
        for r in range(n):
            typ,identity=int(types[r]),int(ids[r]); enum=inverse.get(typ)
            cat=ENUM_CATEGORY.get(enum,'other' if enum is not None else 'unrecognized')
            category[CATEGORIES.index(cat)]+=rowproj[r]; counts[cat]+=1; statecounts[cat]+=1
            if cat.startswith('contact_'): checks.add(tag+' mapped contact row '+str(r),r in owners)
            joint=''
            if cat=='joint_limit':
                valid=0<=identity<len(moment['joint_names']) and bool(moment['jnt_limited'][identity]); checks.add(tag+' actual joint namespace '+str(r),valid)
                if valid: joint=moment['joint_names'][identity]
                active_limits.append(dict(row=r,joint=joint,force=float(force[r])))
                if force[r]!=0: force_limits.append(dict(row=r,joint=joint,force=float(force[r])))
            elif cat in ('tendon_limit','friction_tendon'): checks.add(tag+' tendon namespace '+str(r),0<=identity<int(meta['ntendon']))
            elif cat=='friction_dof': checks.add(tag+' DOF namespace '+str(r),0<=identity<nv)
            elif cat=='equality': checks.add(tag+' equality namespace '+str(r),0<=identity<int(meta['neq']))
            ci=owners.get(r,''); geom=list(map(int,geoms[ci])) if ci!='' else ['','']
            mapping.append([label,i,float(a['time'][i]*1000),r,typ,enum or 'UNRECOGNIZED',identity,cat,ci,*geom,joint,float(force[r]),*rowproj[r,[6,7,8]],json.dumps(J[r].tolist()),json.dumps(rowproj[r].tolist())])
        total=J.T@force; target=a['qfrc_constraint'][i]
        absres=float(np.max(np.abs(total-target))); scale=max(float(np.max(np.abs(total))),float(np.max(np.abs(target))),1e-12); relres=absres/scale
        checks.add(tag+' all48 J.Tforce abs AND rel',absres<=1e-8 and relres<=1e-6,absolute_residual=absres,relative_residual=relres)
        sumres=float(np.max(np.abs(category.sum(0)-total)))
        checks.add(tag+' partition no residual assignment',sumres<=1e-8 and sumres/max(float(np.max(np.abs(total))),1e-12)<=1e-6)
        projections.append(category); states.append(dict(sample=i,time_ms=float(a['time'][i]*1000),counts=statecounts,absolute_residual=absres,relative_residual=relres,active_limit_rows=active_limits,nonzero_limit_force_rows=force_limits))
    return dict(arrays=a,moment=moment,projections=np.asarray(projections),mapping=mapping,states=states,counts=counts)


def first_event(a, mask):
    ids=np.flatnonzero(mask)
    return None if not len(ids) else dict(sample=int(ids[0]),time_ms=float(a['time'][ids[0]]*1000))


def events(item):
    a=item['arrays']; m=item['moment']; rows,curve,peak=all42_limits(a,m)
    upper=np.stack([a['qpos'][:,m['jnt_qposadr'][j]]-m['jnt_range'][j][1] for j,limited in enumerate(m['jnt_limited']) if limited],axis=1)
    return dict(first_load_failure=first_event(a,a['load_ratio']<.8),first_limit_row_activity=first_event(a,[bool(s['active_limit_rows']) for s in item['states']]),first_nonzero_limit_force=first_event(a,[bool(s['nonzero_limit_force_rows']) for s in item['states']]),first_upper_cross_gt_1e_minus6=first_event(a,np.any(upper>1e-6,axis=1)),all42_peak=peak,all42=rows)


def verify(out, historical, reference):
    checks=Checks(); report={}; out=out.resolve(); historical=historical.resolve(); reference=reference.resolve()
    checks.add('offline no physics/neural imports',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    hashes=js(out.parent/'INPUT_HASHES.json'); pre=js(out.parent/'PREREG.json')
    report['source_hashes']=bound_hashes(checks,hashes,'S122 source'); checks.add('source prereg exact',hashes==pre['source_hashes'])
    report['source_audit']=audit_sources(checks,hashes)
    for label,path in [('S118',historical),('S121',reference),('S122',out)]:
        result=js(path/'RESULT.json'); report[label+'_artifact_hashes']=bound_hashes(checks,result['hashes'],label+' artifacts')
        if label!='S122': committed(checks,hashes,path/'RESULT.json',label)
        for name in (*NPZS,*JSONS): committed(checks,result['hashes'],path/name,label)
        if label in ('S121','S122'):
            for name in ('CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'): committed(checks,result['hashes'],path/name,label)
    for label,path in [('S118',historical),('S121',reference)]: report[label+'_source_hashes']=bound_hashes(checks,js(path.parent/'INPUT_HASHES.json'),label+' original sources')
    passed_path=reference/'CONSTRAINT_ROW_INDEPENDENT_CHECK.json'; passed=js(passed_path)
    committed(checks,hashes,passed_path,'original S121 passed verification')
    checks.add('original S121 verification passed',passed.get('passed') is True and passed.get('historical_exact_replay') is True and all(r['passed'] for r in passed['checks']))
    checks.add('original S121 verifier SHA actual',passed['verifier_sha256']==sha(ROOT/'verify_constraint_row_recording.py'))
    for name in ('TRAJECTORY.npz','CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'):
        key=str((reference/name).resolve()); bindings=canonical(passed['new_artifact_hashes']); checks.add('S121 passed report binds actual '+name,bindings.get(key)==sha(reference/name))
    report['S121_passed_verification']=dict(path=str(passed_path),sha256=sha(passed_path),verifier_sha256=passed['verifier_sha256'])
    counts=dict(model_load=1,forward=121,step=120,neural_ticks=166); ledger=js(out.parent/'CALL_LEDGER.json')
    checks.add('exact ledger no extra',ledger==dict(counts,active=None,completed=[ARM],per_arm={ARM:counts}),observed=ledger)
    checks.add('RESULT calls exact',js(out/'RESULT.json')['calls']==counts)
    checks.add('complete not failed',js(out.parent/'COMPLETE.json')['status']=='COMPLETE' and not (out.parent/'FAILED.json').exists())
    comparisons={name:compare_npz(checks,historical/name,out/name) for name in NPZS}
    for name, values in comparisons.items():
        for key, value in values.items(): checks.add('exact historical dtype '+name+':'+key,value['dtype_a']==value['dtype_b'])
    for name in JSONS: checks.add('exact historical JSON '+name,js(historical/name)==js(out/name))
    report['historical_array_comparisons']=comparisons
    report['historical_exact_replay']=all(r['passed'] for r in checks.rows if r['name'].startswith(('exact ','all historical keys','exact historical JSON')))
    old=reconstruct_rows(checks,reference,'S121'); new=reconstruct_rows(checks,out,'S122')
    checks.add('matched physical time all121',exact(old['arrays']['time'],new['arrays']['time'])['identical'])
    checks.add('physical0..3ms',np.allclose(new['arrays']['time'],np.arange(121)*.000025,rtol=0,atol=1e-12))
    checks.add('same actual moment joint metadata',old['moment']==new['moment'])
    event={label:events(item) for label,item in [('S121',old),('S122',new)]}
    for label in event:
        load=event[label]['first_load_failure']; checks.add(label+' first load1.925ms',load is not None and abs(load['time_ms']-1.925)<=1e-10)
    report.update(events=event,per_arm={label:dict(row_type_counts=item['counts'],per_state=item['states']) for label,item in [('S121',old),('S122',new)]},criteria=dict(abs=1e-8,rel=1e-6,operator='AND',scope='all121 all48 same original instant',classification='actual enums, IDs, contact addresses/dims/cone; never residual'),physical_units='NOT_ESTABLISHED',new_calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),new_parameter_proposal=False)
    return checks,report,old,new


def components(item):
    a=item['arrays']; p=item['projections']; torques=a['actuator_force'][:,:,None]*a['actuator_moment']
    contact=p[:,:3].sum(1); limit=p[:,3:5].sum(1); other=p[:,5:].sum(1)
    return dict(contact=contact,limit=limit,other=other,act2=torques[:,2],othermuscle=torques.sum(1)-torques[:,2],total=p.sum(1),qfrcact=a['qfrc_actuator'],passive=a['qfrc_passive'],bias=a['qfrc_bias'],applied=a['qfrc_applied'],constraint=a['qfrc_constraint'],Ma=a['Ma'],qacc=a['qacc'])


def outputs(out,old,new,report):
    def save(name,header,rows):
        path=out/name
        with path.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(header); w.writerows(rows)
        return str(path)
    a=old['arrays']; b=new['arrays']; m=old['moment']; c0=components(old); c1=components(new); keys=list(c0)
    rows=[]; state=[]
    for i,t in enumerate(a['time']):
        for d in (6,7,8):
            j=int(m['dof_jntid'][d]); qi=int(m['jnt_qposadr'][j]); lo,hi=m['jnt_range'][j]
            row=[i,t*1000,d,m['joint_names'][j]]
            for key in keys: row.extend([c0[key][i,d],c1[key][i,d],c1[key][i,d]-c0[key][i,d]])
            for key,idx in [('qpos',qi),('qvel',d)]: row.extend([a[key][i,idx],b[key][i,idx],b[key][i,idx]-a[key][i,idx]])
            row.extend([lo,hi,max(a['qpos'][i,qi]-hi,0),max(b['qpos'][i,qi]-hi,0)])
            rows.append(row)
        for label,item in [('S121',old),('S122',new)]:
            x=item['arrays']; _,globalex,peak=all42_limits(x,item['moment']); rollj=item['moment']['joint_names'].index('joint_LFCoxa_roll'); rollqi=item['moment']['jnt_qposadr'][rollj]; rolld=item['moment']['jnt_dofadr'][rollj]
            state.append([label,i,t*1000,x['qpos'][i,rollqi],x['qvel'][i,rolld],globalex[i],x['load_ratio'][i],bool(item['states'][i]['active_limit_rows']),bool(item['states'][i]['nonzero_limit_force_rows'])])
    header=['sample','physical_ms','DOF','joint',*[f'{key}_{label}' for key in keys+['qpos','qvel'] for label in ('old','new','delta')],'lower','upper','old_upper_excess','new_upper_excess']
    paths=[save('SECANT_CONSTRAINT_DOF678_FULL121.csv',header,rows),save('SECANT_CONSTRAINT_DOF678_WINDOW.csv',header,[r for r in rows if 1.7-1e-10<=r[1]<=2.825+1e-10]),save('SECANT_CONSTRAINT_STATE_EVENTS.csv',['arm','sample','physical_ms','LFCoxa_roll_q','LFCoxa_roll_v','global_all42_excess','load_ratio','limit_row_activity','actual_limit_force_nonzero'],state),save('SECANT_CONSTRAINT_ALL_ROW_MAPPING.csv',['arm','sample','physical_ms','row','actual_enum_value','actual_enum_name','efc_id','category','actual_contact_index','geom1','geom2','joint','efc_force','projection_DOF6','projection_DOF7','projection_DOF8','J_all48_json','projection_all48_json'],old['mapping']+new['mapping'])]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for window,name in [(None,'SECANT_CONSTRAINT_PAIR_FULL.png'),((1.7,2.825),'SECANT_CONSTRAINT_PAIR_WINDOW.png')]:
        fig,axes=plt.subplots(5,3,figsize=(18,18)); t=a['time']*1000
        for col,d in enumerate((6,7,8)):
            j=int(m['dof_jntid'][d]); qi=int(m['jnt_qposadr'][j]); lo,hi=m['jnt_range'][j]
            for label,item,comp,style in [('S121',old,c0,'--'),('S122',new,c1,'-')]:
                x=item['arrays']
                for key in ('contact','limit','other','total'): axes[0,col].plot(t,comp[key][:,d],ls=style,label=label+' '+key)
                for key in ('act2','othermuscle','qfrcact'): axes[1,col].plot(t,comp[key][:,d],ls=style,label=label+' '+key)
                axes[2,col].plot(t,x['qpos'][:,qi],ls=style,label=label+' q'); axes[3,col].plot(t,x['qvel'][:,d],ls=style,label=label+' v')
                _,ex,_=all42_limits(x,m); axes[4,col].plot(t,ex,ls=style,label=label+' all42 excess')
                ev=report['events'][label]
                for key,color,ls in [('first_load_failure','grey',':'),('first_limit_row_activity','orange','--'),('first_nonzero_limit_force','red',':'),('first_upper_cross_gt_1e_minus6','purple','-.')]:
                    if ev[key] is not None:
                        for ax in axes[:,col]: ax.axvline(ev[key]['time_ms'],color=color,ls=ls,alpha=.6,label=label+' '+key if ax is axes[4,col] else None)
            axes[0,col].set_title('DOF'+str(d)+' '+m['joint_names'][j]); axes[2,col].axhline(hi,color='black',ls=':',label='actual upper'); axes[2,col].axhline(lo,color='grey',ls=':',label='actual lower'); axes[4,col].axhline(1e-6,color='black',ls=':')
            for row,ylabel in enumerate(('constraint signed components','actual signed muscle torque','joint q','joint v','all42 global excess')):
                ax=axes[row,col]; ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=5); ax.set_xlabel('physical ms')
                if window: ax.set_xlim(*window)
        fig.suptitle('S122 vs S121 same-time saved-state coupling; native units NOT_ESTABLISHED\nLimit reaction after crossing is not cause of early load failure; contact delta is algebra, not causal attribution')
        fig.tight_layout(); path=out/name; fig.savefig(path,dpi=140); plt.close(fig); paths.append(str(path))
    # Explicitly descriptive judgment, conditional on actually observed event ordering.
    after={}
    for label,ev in report['events'].items():
        f=ev['first_nonzero_limit_force']; cross=ev['first_upper_cross_gt_1e_minus6']; load=ev['first_load_failure']
        after[label]=dict(limit_force_after_load=f is not None and load is not None and f['time_ms']>load['time_ms'],limit_force_at_or_after_cross=f is not None and cross is not None and f['time_ms']>=cross['time_ms'])
    report['independent_judgment']=dict(observed_event_order=after,statement='Observed actuator2 and other-muscle signed torque, contact projection, joint q/v and inertial response are coupled along the trajectories. Limit row activity is not identical to nonzero reaction. Where the reported ordering holds, limit reaction at/after crossing cannot explain the earlier load failure. Contact-projection differences close an algebraic partition, not a causal explanation. No independent new parameter direction or weight value is established.',causal_attribution='NOT_ESTABLISHED',new_parameter='NONE',behavior_acceptance=False)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT); p.add_argument('--historical',type=Path,default=HISTORICAL); p.add_argument('--reference',type=Path,default=REFERENCE); args=p.parse_args(); out=args.output.resolve()
    report=dict(output=str(out),historical=str(args.historical.resolve()),reference=str(args.reference.resolve()),verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),passed=False)
    try:
        checks,data,old,new=verify(out,args.historical,args.reference); report.update(data); report['outputs']=outputs(out,old,new,report); report['checks']=checks.rows; report['passed']=checks.passed
        if any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')): raise RuntimeError('offline boundary violated')
    except Exception as exc: report.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    path=out/'SECANT_CONSTRAINT_PAIR_INDEPENDENT_CHECK.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],report=str(path),error=report.get('error'),failed_checks=[r['name'] for r in report.get('checks',[]) if not r['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
