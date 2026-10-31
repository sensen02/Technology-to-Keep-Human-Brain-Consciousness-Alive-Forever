"""S121/P1/S2 saved-arrays-only independent constraint-row verification.
Never imports a runner, MuJoCo, torch, neural engine, or loads a model.
Exact S115 replay is a checked proposition, never an assumed historical claim.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z/init80ms')
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms')
CATEGORIES = ('contact_frictionless', 'contact_pyramidal', 'contact_elliptic', 'joint_limit', 'tendon_limit', 'friction_dof', 'friction_tendon', 'equality', 'other', 'unrecognized')
ENUM_CATEGORY = {'mjCNSTR_CONTACT_FRICTIONLESS':'contact_frictionless', 'mjCNSTR_CONTACT_PYRAMIDAL':'contact_pyramidal', 'mjCNSTR_CONTACT_ELLIPTIC':'contact_elliptic', 'mjCNSTR_LIMIT_JOINT':'joint_limit', 'mjCNSTR_LIMIT_TENDON':'tendon_limit', 'mjCNSTR_FRICTION_DOF':'friction_dof', 'mjCNSTR_FRICTION_TENDON':'friction_tendon', 'mjCNSTR_EQUALITY':'equality'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


class Checks:
    def __init__(self): self.rows=[]
    def add(self, name, passed, **details):
        self.rows.append(dict(name=name, passed=bool(passed), **details))
    @property
    def passed(self): return all(r['passed'] for r in self.rows)


def exact(a,b):
    result=dict(shape_a=list(a.shape), shape_b=list(b.shape), dtype_a=str(a.dtype), dtype_b=str(b.dtype))
    if a.shape != b.shape: return dict(result, identical=False)
    equal = a == b
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc': equal = equal | (np.isnan(a)&np.isnan(b))
    bad=np.argwhere(~equal)
    result.update(identical=bool(not len(bad)), unequal_elements=int(len(bad)), first_differing_indices=bad[:12].tolist())
    if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
        delta=np.abs(a.astype(np.complex128)-b.astype(np.complex128))
        result['max_abs_difference']=float(np.max(delta)) if delta.size else 0.
    return result


def compare_npz(checks, old, new):
    a,b=npz(old),npz(new)
    checks.add('all historical keys retained '+old.name, set(a)<=set(b), missing=sorted(set(a)-set(b)), added=sorted(set(b)-set(a)))
    rows={}
    for key in sorted(set(a)&set(b)):
        rows[key]=exact(a[key],b[key]); checks.add('exact '+old.name+':'+key,rows[key]['identical'],**rows[key])
    return rows


def bound_hashes(checks, hashes, label):
    observed={}
    for p,digest in hashes.items():
        path=Path(p); actual=sha(path) if path.is_file() else None
        checks.add(label+' '+str(path),actual==digest, expected=digest, observed=actual); observed[str(path)]=actual
    return observed


def pick(d,*names):
    for name in names:
        if name in d: return d[name]
    raise KeyError('required one of '+repr(names))


def source_audit(checks, runner, original):
    """Static evidence only: compare explicit physics/neural call sites and pose writes."""
    def scan(path):
        tree=ast.parse(path.read_text()); calls=[]; poses=[]; recorder=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                name=ast.unparse(n.func)
                if name.endswith(('from_xml_path','mj_forward','mj_step','mj_resetData','mj_resetDataKeyframe')) or name=='net.step': calls.append(name)
            targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
            for t in targets:
                if 'qpos' in ast.unparse(t): poses.append(ast.unparse(n))
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and ('constraint' in n.name and 'record' in n.name):
                bad=[]
                for c in ast.walk(n):
                    if isinstance(c,ast.Call) and any(s in ast.unparse(c.func) for s in ('mj_forward','mj_step','from_xml','net.step','mj_jac','mj_mul','mj_contactForce')): bad.append(ast.unparse(c))
                    ts=c.targets if isinstance(c,ast.Assign) else [c.target] if isinstance(c,(ast.AugAssign,ast.AnnAssign)) else []
                    for t in ts:
                        if isinstance(t,(ast.Attribute,ast.Subscript)) and ast.unparse(t).startswith(('d.','m.','net.')): bad.append(ast.unparse(c))
                recorder.append(dict(name=n.name,forbidden_calls_or_state_writes=bad))
        return dict(explicit_physics_neural_calls=sorted(calls),qpos_assignments=poses,recorder_functions=recorder)
    a,b=scan(original),scan(runner)
    checks.add('static passive wrapper introduces zero physics/neural calls',not b['explicit_physics_neural_calls'],original_delegated_calls=a['explicit_physics_neural_calls'],wrapper_calls=b['explicit_physics_neural_calls'])
    checks.add('static passive wrapper zero qpos writes original sole initial write',not b['qpos_assignments'] and len(a['qpos_assignments'])==1,original_assignments=a['qpos_assignments'],wrapper_assignments=b['qpos_assignments'])
    checks.add('static recorder is passive',bool(b['recorder_functions']) and all(not r['forbidden_calls_or_state_writes'] for r in b['recorder_functions']),functions=b['recorder_functions'])
    return dict(original=a,new=b,boundary='Static hash-bound AST checks plus actual ledger; not proof against unrecorded runtime events.')


def jacobian(z, prefix, n, nv, sparse, checks):
    dense=np.asarray(z[prefix+'efc_J_dense']); raw=np.asarray(z[prefix+'efc_J_raw'])
    checks.add(prefix+'dense J shape',dense.shape==(n,nv))
    if sparse:
        adr=z[prefix+'efc_J_rowadr']; nnz=z[prefix+'efc_J_rownnz']; cols=z[prefix+'efc_J_colind']; J=np.zeros((n,nv))
        checks.add(prefix+'CSR row metadata size',len(adr)==n and len(nnz)==n)
        for r in range(n):
            start,size=int(adr[r]),int(nnz[r]); c=cols[start:start+size]
            valid=0<=start and 0<=size and start+size<=len(raw) and len(c)==size and len(set(c.tolist()))==size and np.all((c>=0)&(c<nv))
            checks.add(prefix+'CSR valid row '+str(r),valid)
            if not valid: raise ValueError(prefix+' malformed actual CSR')
            J[r,c]=raw[start:start+size]
    else:
        if raw.size != n*nv: raise ValueError(prefix+' malformed dense raw J')
        J=raw.reshape(n,nv)
    diff=exact(J,dense); checks.add(prefix+'raw J independent reconstruction',diff['identical'],**diff)
    return J


def verify(out, reference):
    checks=Checks(); report={}; out=out.resolve(); reference=reference.resolve(); parent=out.parent
    checks.add('no physics/neural imported',not any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond')))
    hashes=js(parent/'INPUT_HASHES.json'); pre=js(parent/'PREREG.json')
    report['input_hashes']=bound_hashes(checks,hashes,'new source')
    checks.add('source commitments agree',hashes==pre['source_hashes'])
    oldres=js(reference/'RESULT.json'); newres=js(out/'RESULT.json')
    report['S115_artifact_hashes']=bound_hashes(checks,oldres['hashes'],'S115 artifacts')
    report['new_artifact_hashes']=bound_hashes(checks,newres['hashes'],'new artifacts')
    # S115 input source chain is validated without modifying historical commitments.
    report['S115_source_hashes']=bound_hashes(checks,js(reference.parent/'INPUT_HASHES.json'),'S115 source')
    for p in (reference/'RESULT.json',out/'CONSTRAINT_ROWS.npz',out/'CONSTRAINT_ROW_METADATA.json'):
        commitments=hashes if p==reference/'RESULT.json' else newres['hashes']
        canonical_commitments = {str(Path(name).resolve()): digest for name, digest in commitments.items()}
        checks.add('committed '+str(p),canonical_commitments.get(str(p.resolve()))==sha(p))
    runner=ROOT/'run_constraint_row_recording.py'
    checks.add('runner hash bound',hashes.get(str(runner))==sha(runner))
    report['source_audit']=source_audit(checks,runner,ROOT/'run_extended_initialization_probe.py')
    counts=dict(model_load=1,forward=121,step=120,neural_ticks=166)
    ledger=js(parent/'CALL_LEDGER.json'); report['actual_ledger']=ledger
    checks.add('actual ledger exact no extra explicit calls',ledger==dict(counts,active=None,completed=['init80ms'],per_arm={'init80ms':counts}))
    checks.add('RESULT calls exact',newres['calls']==counts)
    checks.add('run COMPLETE not FAILED',js(parent/'COMPLETE.json')['status']=='COMPLETE' and not (parent/'FAILED.json').exists())
    comparisons={}
    for name in ('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz'):
        comparisons[name]=compare_npz(checks,reference/name,out/name)
    for name in ('NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','MODEL_PROVENANCE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json'):
        checks.add('exact JSON '+name,js(reference/name)==js(out/name))
    report['historical_array_comparisons']=comparisons
    report['historical_exact_replay']=all(x['identical'] for values in comparisons.values() for x in values.values()) and all(r['passed'] for r in checks.rows if r['name'].startswith(('exact JSON','all historical keys')))
    a=npz(out/'TRAJECTORY.npz'); z=npz(out/'CONSTRAINT_ROWS.npz'); meta=js(out/'CONSTRAINT_ROW_METADATA.json'); moment=js(out/'MOMENT_METADATA.json')
    enums=pick(meta,'constraint_enum','constraint_type_enum','mjtConstraint','enum_members')
    if '__members__' in enums: enums=enums['__members__']
    enums={k:int(v) for k,v in enums.items()}; inverse={v:k for k,v in enums.items()}
    cone=int(pick(meta,'cone','cone_type')); cone_enum=pick(meta,'cone_enum','mjtCone')
    if '__members__' in cone_enum: cone_enum=cone_enum['__members__']
    checks.add('actual cone recognized',cone in [int(v) for v in cone_enum.values()])
    nv=int(np.asarray(z['nv']).flat[0]); checks.add('full 48 DOFs and121 original states',nv==48 and len(a['time'])==121 and len(z['nefc'])==121)
    checks.add('row sample time exactly original instant',exact(z['time'],a['time'])['identical'])
    projections=[]; mapping=[]; perstate=[]; allcounts={c:0 for c in CATEGORIES}
    for i in range(121):
        p=f'sample{i:03d}_'; n=int(z['nefc'][i]); sparse=bool(np.asarray(z['is_sparse']).flat[0 if np.asarray(z['is_sparse']).size==1 else i]); J=jacobian(z,p,n,nv,sparse,checks)
        types=np.asarray(z[p+'efc_type']); ids=np.asarray(z[p+'efc_id']); force=np.asarray(z[p+'efc_force'])
        checks.add(p+'row vectors valid',types.shape==ids.shape==force.shape==(n,) and np.isfinite(force).all() and np.isfinite(J).all())
        if 'row_offsets' in z:
            lo,hi=map(int,z['row_offsets'][i:i+2]); checks.add(p+'ragged offset count',hi-lo==n)
            for key,value in [('efc_type',types),('efc_id',ids),('efc_force',force)]: checks.add(p+'concatenated '+key,exact(z[key][lo:hi],value)['identical'])
        address=pick(z,p+'contact_efc_address',p+'contact_efcaddress'); dims=pick(z,p+'contact_dim',p+'contact_dims'); geoms=pick(z,p+'contact_geom',p+'contact_geoms')
        checks.add(p+'contact metadata shape',len(address)==len(dims)==len(geoms) and geoms.shape==(len(address),2))
        owners={}; contacts=[]
        for ci,(address0,dim0) in enumerate(zip(address,dims)):
            start,dim=int(address0),int(dim0)
            checks.add(p+'contact valid geom '+str(ci),np.all((geoms[ci]>=0)&(geoms[ci]<int(meta['ngeom']))))
            if start==-1: contacts.append(dict(contact=ci,address=-1,dim=dim,rows=[])); continue
            if dim==1: enum='mjCNSTR_CONTACT_FRICTIONLESS'; size=1
            elif cone==int(cone_enum['mjCONE_PYRAMIDAL']): enum='mjCNSTR_CONTACT_PYRAMIDAL'; size=2*(dim-1)
            else: enum='mjCNSTR_CONTACT_ELLIPTIC'; size=dim
            rr=list(range(start,start+size)); valid=dim in (1,3,4,6) and start>=0 and start+size<=n
            checks.add(p+'contact actual address range '+str(ci),valid,address=start,dim=dim,rows=rr)
            if not valid: raise ValueError(p+' invalid active contact address')
            checks.add(p+'contact actual type/id '+str(ci),all(int(types[r])==enums[enum] and int(ids[r])==ci for r in rr),rows=rr,expected_type=enum)
            for r in rr:
                checks.add(p+'no contact row overlap '+str(r),r not in owners); owners[r]=ci
            contacts.append(dict(contact=ci,address=start,dim=dim,rows=rr))
        rowproj=J*force[:,None]; total=J.T@force; category=np.zeros((len(CATEGORIES),nv)); statecounts={c:0 for c in CATEGORIES}
        for r in range(n):
            typ,identity=int(types[r]),int(ids[r]); enum=inverse.get(typ); cat=ENUM_CATEGORY.get(enum,'other' if enum is not None else 'unrecognized'); statecounts[cat]+=1; allcounts[cat]+=1
            category[CATEGORIES.index(cat)]+=rowproj[r]
            if cat.startswith('contact_'): checks.add(p+'contact row mapped '+str(r),r in owners)
            joint=''; geom1=geom2=''; ci=owners.get(r,'')
            if ci!='': geom1,geom2=map(int,geoms[ci])
            if cat=='joint_limit':
                valid=0<=identity<len(moment['joint_names']) and bool(moment['jnt_limited'][identity]); checks.add(p+'joint limit id namespace '+str(r),valid,efc_id=identity)
                if valid: joint=moment['joint_names'][identity]
            elif cat in ('tendon_limit','friction_tendon'): checks.add(p+'tendon id namespace '+str(r),0<=identity<int(meta['ntendon']))
            elif cat=='friction_dof': checks.add(p+'DOF id namespace '+str(r),0<=identity<nv)
            elif cat=='equality': checks.add(p+'equality id namespace '+str(r),0<=identity<int(meta['neq']))
            mapping.append([i,a['time'][i]*1000,r,typ,enum or 'UNRECOGNIZED',identity,cat,ci,geom1,geom2,joint,force[r],*rowproj[r,[6,7,8]],json.dumps(J[r].tolist()),json.dumps(rowproj[r].tolist())])
        target=a['qfrc_constraint'][i]; absres=float(np.max(np.abs(total-target))); scale=max(float(np.max(np.abs(total))),float(np.max(np.abs(target))),1e-12); relres=absres/scale
        checks.add(p+'all48 J.T force same-instant original qfrc',absres<=1e-8 and relres<=1e-6,absolute_residual=absres,relative_residual=relres,scale=scale)
        sumres=float(np.max(np.abs(category.sum(0)-total))); checks.add(p+'categories partition all rows no residual assignment',sumres<=1e-8 and sumres/max(float(np.max(np.abs(total))),1e-12)<=1e-6,absolute_residual=sumres)
        projections.append(category); perstate.append(dict(sample=i,time_ms=float(a['time'][i]*1000),nefc=n,counts=statecounts,absolute_residual=absres,relative_residual=relres,contacts=contacts))
    report.update(row_type_counts=allcounts,per_state=perstate,criteria=dict(absolute_max=1e-8,relative_max=1e-6,operator='AND',relative_scale='max absolute component of either side at the same state, floor1e-12',scope='all121 states all48DOF original instant',category_policy='Every row independently assigned by actual enum; unrecognized retained, no residual assignment'),physical_units='NOT_ESTABLISHED',causal_claim=False,S117_S118_attribution='NOT_ESTABLISHED',new_physics_neural_calls_by_verifier=dict(model_load=0,forward=0,step=0,neural_ticks=0))
    return checks,report,a,moment,np.asarray(projections),mapping


def outputs(out,a,moment,projections,mapping):
    def save(name,header,rows):
        path=out/name
        with path.open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
        return str(path)
    rows=[]
    for i,t in enumerate(a['time']):
        for d in (6,7,8):
            j=int(moment['dof_jntid'][d]); qidx=int(moment['jnt_qposadr'][j]); lo,hi=moment['jnt_range'][j]; q=a['qpos'][i,qidx]; torque=a['actuator_force'][i]*a['actuator_moment'][i,:,d]
            rows.append([i,t*1000,d,moment['joint_names'][j],q,a['qvel'][i,d],lo,hi,max(lo-q,q-hi,0),*projections[i,:,d],projections[i,:,d].sum(),a['qfrc_constraint'][i,d],a['qfrc_actuator'][i,d],torque.sum(),json.dumps(torque.tolist())])
    header=['sample','physical_ms','DOF','joint','q','v','lower_limit','upper_limit','limit_excess',*CATEGORIES,'row_projection_sum','original_qfrc_constraint','actual_qfrc_actuator','actual_muscle_force_times_moment_sum','per_muscle_signed_torque_json']
    paths=[save('CONSTRAINT_DOF678_FULL121.csv',header,rows),save('CONSTRAINT_DOF678_FIRSTLOAD_TO_PEAK.csv',header,[r for r in rows if 1.925-1e-10<=r[1]<=2.825+1e-10]),save('CONSTRAINT_ALL_ROW_MAPPING.csv',['sample','physical_ms','row','actual_enum_value','actual_enum_name','efc_id','category','actual_contact_index','geom1','geom2','actual_joint','efc_force','projection_DOF6','projection_DOF7','projection_DOF8','J_all48_json','projection_all48_json'],mapping)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(17,12)); t=a['time']*1000
    contact=projections[:,:3].sum(1); limit=projections[:,3]; other=projections[:,4:].sum(1)
    for col,d in enumerate((6,7,8)):
        j=int(moment['dof_jntid'][d]); qidx=int(moment['jnt_qposadr'][j]); lo,hi=moment['jnt_range'][j]
        ax=axes[0,col]
        for value,label in [(contact[:,d],'contact (all actual contact enums)'),(limit[:,d],'joint limit'),(other[:,d],'other retained types'),(a['qfrc_actuator'][:,d],'actual muscle torque')]: ax.plot(t,value,label=label)
        ax.set_title('DOF'+str(d)+' '+moment['joint_names'][j]); ax.set_ylabel('signed generalized force (native)'); ax.legend(fontsize=7)
        axes[1,col].plot(t,a['qpos'][:,qidx],label='actual q'); axes[1,col].axhline(lo,color='grey',ls='--',label='lower'); axes[1,col].axhline(hi,color='black',ls='--',label='upper'); axes[1,col].legend(fontsize=8); axes[1,col].set_ylabel('q and actual joint limits')
        axes[2,col].plot(t,a['qvel'][:,d],label='actual qvel'); axes[2,col].legend(); axes[2,col].set_ylabel('joint velocity'); axes[2,col].set_xlabel('physical time (ms)')
        for ax in axes[:,col]: ax.axvspan(1.925,2.825,color='grey',alpha=.12); ax.grid(alpha=.25)
    fig.suptitle('S121 saved original-instant row projection: descriptive only; no S117/S118 causal attribution\nNative units NOT_ESTABLISHED; shaded first-load-failure to limit peak 1.925–2.825ms')
    fig.tight_layout(); path=out/'CONSTRAINT_ROW_BREAKDOWN.png'; fig.savefig(path,dpi=160); plt.close(fig); paths.append(str(path))
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT); parser.add_argument('--reference',type=Path,default=REFERENCE); args=parser.parse_args(); out=args.output
    report=dict(output=str(out.resolve()),reference=str(args.reference.resolve()),verifier_sha256=sha(__file__))
    try:
        checks,data,a,moment,projections,mapping=verify(out,args.reference); report.update(data); report['outputs']=outputs(out,a,moment,projections,mapping); report['checks']=checks.rows; report['passed']=checks.passed
    except Exception as exc:
        report.update(passed=False,error=str(exc),traceback=traceback.format_exc())
    path=out/'CONSTRAINT_ROW_INDEPENDENT_CHECK.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],report=str(path.resolve()),error=report.get('error'),failed_checks=[r['name'] for r in report.get('checks',[]) if not r['passed']]),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
