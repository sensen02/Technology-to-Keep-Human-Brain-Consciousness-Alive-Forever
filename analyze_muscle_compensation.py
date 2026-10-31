"""S123/P1/S2 pure frozen-array analysis. No simulator, XML, model or neural imports.
Signed algebra is descriptive, not physical causal attribution. Fresh output only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收')
OLD = BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms'
NEW = BASE/'s122-secant-constraint-pair/20261024T073500Z/w36b4_secant'
FORMAL = BASE/'s119-training-constraints/20261024T070600Z/TRAINING_CONSTRAINTS.json'
OUT = BASE/'s123-muscle-compensation/20261024T074500Z'
DOFS = [6, 7, 8]
ATOL, RTOL = 1e-12, 1e-10


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def js(p): return json.loads(Path(p).read_text())
def dump(p, d): p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def bits(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    return np.any(a.view(np.uint8).reshape(a.shape+(a.dtype.itemsize,)) != b.view(np.uint8).reshape(b.shape+(b.dtype.itemsize,)), axis=-1)


def first(mask, time):
    ids = np.flatnonzero(mask)
    return None if not len(ids) else dict(sample=int(ids[0]), time_ms=float(time[ids[0]]))


def onset(a, b, t):
    exact = bits(a, b)
    numerical = np.abs(b-a) > ATOL + RTOL*np.maximum(np.abs(a), np.abs(b))
    if a.ndim > 1:
        exact = np.any(exact, axis=tuple(range(1,a.ndim)))
        numerical = np.any(numerical, axis=tuple(range(1,a.ndim)))
    return dict(first_exact_bit_difference=first(exact,t), first_numerically_distinguishable=first(numerical,t), max_abs_delta=float(np.max(np.abs(b-a))))


def save(out, name, header, rows):
    p = out/name
    with p.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    return p


def projections(path, a):
    z=arrays(path/'CONSTRAINT_ROWS.npz'); meta=js(path/'CONSTRAINT_ROW_METADATA.json')
    contact_types=[v for k,v in meta['constraint_enum'].items() if k.startswith('mjCNSTR_CONTACT_')]
    limit_type=meta['constraint_enum']['mjCNSTR_LIMIT_JOINT']
    contact=[]; limits=[]; activity=[]; residual=[]
    for i in range(len(a['time'])):
        p=f'sample{i:03d}_'; n=int(z['nefc'][i]); nv=int(z['nv'][i])
        if z['is_sparse'][i]:
            J=np.zeros((n,nv)); raw=z[p+'efc_J_raw']; adr=z[p+'efc_J_rowadr']; nnz=z[p+'efc_J_rownnz']; col=z[p+'efc_J_colind']
            for r in range(n):
                ix=np.arange(adr[r],adr[r]+nnz[r]); J[r,col[ix]]=raw[ix]
        else: J=z[p+'efc_J_raw'].reshape(n,nv)
        f=z[p+'efc_force']; typ=z[p+'efc_type']; c=np.isin(typ,contact_types); l=typ==limit_type
        residual.append(float(np.max(np.abs(J.T@f-a['qfrc_constraint'][i]))))
        contact.append(J[c].T@f[c]); limits.append(J[l].T@f[l]); activity.append(bool(np.any(l)))
    assert max(residual)<1e-8
    return np.array(contact), np.array(limits), np.array(activity), max(residual)


def analyze(out):
    assert not out.exists(), 'Fresh output required; refusing overwrite'
    assert not any(k in sys.modules for k in ('mujoco','torch'))
    # Protect every file in the frozen experiment roots and formal constraints root.
    frozen={str(p):sha(p) for root in (OLD.parent,NEW.parent,FORMAL.parent) for p in root.rglob('*') if p.is_file()}
    bindings={}; verified={}
    for path in (OLD,NEW):
        for manifest in (path.parent/'INPUT_HASHES.json',path/'RESULT.json'):
            d=js(manifest); h=d['hashes'] if manifest.name=='RESULT.json' else d
            for filename,digest in h.items():
                p=Path(filename)
                assert p.is_absolute() and p.is_file(), str(p)
                assert sha(p)==digest, 'Frozen commitment mismatch '+str(p)
                bindings[str(p)]=digest
        report=path/('CONSTRAINT_ROW_INDEPENDENT_CHECK.json' if path==OLD else 'SECANT_CONSTRAINT_PAIR_INDEPENDENT_CHECK.json')
        d=js(report)
        assert d['passed'] is True and all(c['passed'] for c in d['checks'])
        verifier=ROOT/('verify_constraint_row_recording.py' if path==OLD else 'verify_secant_constraint_pair.py')
        assert sha(verifier)==d['verifier_sha256']
        # Explicitly bind the passed reports' actual trajectory/row data commitments.
        h=d['new_artifact_hashes'] if path==OLD else d['S122_artifact_hashes']
        canonical={str(Path(k).resolve()):v for k,v in h.items()}
        for name in ('TRAJECTORY.npz','CONSTRAINT_ROWS.npz','CONSTRAINT_ROW_METADATA.json'):
            assert canonical[str((path/name).resolve())]==sha(path/name)
        verified[str(report)]={'sha256':sha(report),'passed':True,'verifier_sha256':sha(verifier)}
    a,b=arrays(OLD/'TRAJECTORY.npz'),arrays(NEW/'TRAJECTORY.npz')
    ia,ib=arrays(OLD/'INITIALIZATION.npz'),arrays(NEW/'INITIALIZATION.npz')
    ma,mb=js(OLD/'MOMENT_METADATA.json'),js(NEW/'MOMENT_METADATA.json')
    assert ma==mb and not np.any(bits(a['time'],b['time']))
    assert not np.any(bits(a['actuator_pool'],b['actuator_pool']))
    assert a['actuator_force'].shape==(121,90)
    t=a['time']*1000; keep=np.flatnonzero(t<=2.825+1e-10); t=t[keep]
    assert len(keep)==114
    ca,la,al,ra=projections(OLD,a); cb,lb,bl,rb=projections(NEW,b)
    oldlimit=first(al,a['time']*1000); newlimit=first(bl,b['time']*1000)
    pre=t<min(oldlimit['time_ms'],newlimit['time_ms'])-1e-10
    F0=a['actuator_force'][keep]; F1=b['actuator_force'][keep]
    M0=a['actuator_moment'][keep][:,:,DOFS]; M1=b['actuator_moment'][keep][:,:,DOFS]
    T0=F0[:,:,None]*M0; T1=F1[:,:,None]*M1; delta=T1-T0
    dF=F1-F0; dM=M1-M0
    ft=dF[:,:,None]*M0; mt=F0[:,:,None]*dM; xt=dF[:,:,None]*dM
    decomposition_error=float(np.max(np.abs(delta-ft-mt-xt)))
    projection_error=max(float(np.max(np.abs(T.sum(1)-arm['qfrc_actuator'][keep][:,DOFS]))) for T,arm in ((T0,a),(T1,b)))
    assert decomposition_error<1e-12 and projection_error<1e-12
    idx=int(np.argmin(abs(t-2.5))); assert abs(t[idx]-2.5)<1e-10
    others=np.arange(90)!=2; signed=delta[idx,:,2]
    ranked=sorted(np.flatnonzero(others),key=lambda ch:float(signed[ch]),reverse=True)
    top=sorted(np.flatnonzero(others),key=lambda ch:abs(float(signed[ch])),reverse=True)[:6]
    other_sum=float(signed[others].sum()); act2=float(signed[2])
    assert abs(other_sum-.03589112560425)<1e-12 and abs(act2-(-.027214851704012))<1e-12
    out.mkdir(parents=True)
    dump(out/'PREREG.json',dict(scope='S123/P1/S2 purefrozen old=S121 new=S122; all90 DOF678 native0..2.825ms',formula='deltaF*M0 + F0*deltaM + deltaF*deltaM; anchor S121',precision=dict(abs=ATOL,rel=RTOL,rule='abs(delta)>abs+rel*max(abs(old),abs(new)); exact is bytewise including signed zero'),input_sha256={**frozen,**bindings,str(Path(__file__)):sha(__file__)},zero_calls=dict(model_load=0,forward=0,step=0,neural_tick=0,weight_write=0)))
    rows=[]; state=[]; firstrows=[]; rankrows=[]
    for ch in range(90):
        name=ma['actuator_names'][ch]; pool=int(a['actuator_pool'][ch])
        for field in ('ctrl','act','actuator_force','actuator_length','actuator_velocity'):
            record=onset(a[field][keep,ch],b[field][keep,ch],t)
            firstrows.append(['channel',ch,name,pool,field,'',json.dumps(record['first_exact_bit_difference']),json.dumps(record['first_numerically_distinguishable']),record['max_abs_delta']])
        for k,dof in enumerate(DOFS):
            for field,x,y in (('moment',M0[:,ch,k],M1[:,ch,k]),('torque',T0[:,ch,k],T1[:,ch,k])):
                r=onset(x,y,t);firstrows.append(['channel',ch,name,pool,field,dof,json.dumps(r['first_exact_bit_difference']),json.dumps(r['first_numerically_distinguishable']),r['max_abs_delta']])
            for i in range(len(t)):
                rows.append([i,t[i],bool(pre[i]),ch,name,pool,dof,F0[i,ch],F1[i,ch],dF[i,ch],M0[i,ch,k],M1[i,ch,k],dM[i,ch,k],T0[i,ch,k],T1[i,ch,k],delta[i,ch,k],ft[i,ch,k],mt[i,ch,k],xt[i,ch,k]])
        for i,source in enumerate(keep):
            state.append([i,t[i],bool(pre[i]),ch,name,pool,*[float(v) for field in ('ctrl','act','actuator_force','actuator_length','actuator_velocity') for v in (a[field][source,ch],b[field][source,ch],b[field][source,ch]-a[field][source,ch])]])
    for k,dof in enumerate(DOFS):
        joint=ma['dof_jntid'][dof]; qi=ma['jnt_qposadr'][joint]; name=ma['joint_names'][joint]
        for field,x,y in (('q',a['qpos'][keep,qi],b['qpos'][keep,qi]),('v',a['qvel'][keep,dof],b['qvel'][keep,dof]),('contact_row_projection',ca[keep,dof],cb[keep,dof])):
            r=onset(x,y,t);firstrows.append(['joint',joint,name,'',field,dof,json.dumps(r['first_exact_bit_difference']),json.dumps(r['first_numerically_distinguishable']),r['max_abs_delta']])
    for rank,ch in enumerate(ranked,1):
        rankrows.append([rank,ch,ma['actuator_names'][ch],int(a['actuator_pool'][ch]),T0[idx,ch,2],T1[idx,ch,2],signed[ch],ft[idx,ch,2],mt[idx,ch,2],xt[idx,ch,2],ch in top,'positive delta is less-negative torque, not positive torque' if T0[idx,ch,2]<0 and T1[idx,ch,2]<0 and signed[ch]>0 else 'signed algebra only'])
    save(out,'ALL90_SIGNED_TORQUE_DECOMPOSITION.csv',['sample','time_ms','strict_prelimit','channel','muscle','pool','DOF','force_old','force_new','delta_force','moment_old','moment_new','delta_moment','torque_old','torque_new','delta_torque','deltaF_M0','F0_deltaM','deltaF_deltaM'],rows)
    save(out,'ALL90_ALIGNED_MUSCLE_STATE.csv',['sample','time_ms','strict_prelimit','channel','muscle','pool',*[f'{f}_{s}' for f in ('ctrl','act','force','length','velocity') for s in ('old','new','delta')]],state)
    save(out,'FIRST_DETECTABLE_DIFFERENCES.csv',['kind','id','name','pool','field','DOF','first_exact_bit_json','first_numerical_json','max_abs_delta'],firstrows)
    save(out,'SIGNED_RANK_OTHER89_2p5ms_DOF8.csv',['signed_rank','channel','muscle','pool','torque_old','torque_new','delta_torque','deltaF_M0','F0_deltaM','deltaF_deltaM','abs_top6','interpretation'],rankrows)
    jointrows=[]
    for i,source in enumerate(keep):
        for k,dof in enumerate(DOFS):
            joint=ma['dof_jntid'][dof]; qi=ma['jnt_qposadr'][joint]
            jointrows.append([i,t[i],bool(pre[i]),dof,ma['joint_names'][joint],qi,*[v for x,y in ((a['qpos'][source,qi],b['qpos'][source,qi]),(a['qvel'][source,dof],b['qvel'][source,dof]),(ca[source,dof],cb[source,dof]),(la[source,dof],lb[source,dof]),(T0[i,:,k].sum(),T1[i,:,k].sum())) for v in (x,y,y-x)]])
    save(out,'JOINT_CONTACT_ROW_PROJECTION.csv',['sample','time_ms','strict_prelimit','DOF','joint','qpos_index',*[f'{f}_{s}' for f in ('q','v','contact','joint_limit','total_actuator') for s in ('old','new','delta')]],jointrows)
    checks={f'other89_{field}_all121_bit_identical':not bool(np.any(bits(a[field][:,others],b[field][:,others]))) for field in ('ctrl','act')}
    initial={field:dict(bit_identical=not bool(np.any(bits(ia[field],ib[field]))),max_abs_delta=float(np.max(np.abs(ib[field]-ia[field])))) for field in ('body_act','body_ctrl','body_qpos','body_qvel')}
    missing=['actuator_gainprm','actuator_biasprm','actuator_gaintype','actuator_biastype','actuator_acc0','actuator_lengthrange','actuator_dynprm','actuator_dyntype','actuator_actadr','actuator_actnum']
    inventory={label:{'array_parameter_keys_present':[k for k in missing if k in arm], 'metadata_parameter_keys_present':[k for k in missing if k in meta]} for label,arm,meta in (('old',a,ma),('new',b,mb))}
    summary=dict(scope='S123/P1/S2 offline native all90 x DOF678 x114 samples 0..2.825ms',old=str(OLD),new=str(NEW),formal_constraints=str(FORMAL),formal_sha256=sha(FORMAL),passed_upstream_verification=verified,precision=dict(abs=ATOL,rel=RTOL,exact='bytewise first difference, including signed zero',numerical='abs(new-old)>1e-12+1e-10*max(abs(old),abs(new))',interpretation='Detectability only, not physical causal timing; saved sampling resolution 0.025ms'),events=dict(old_first_load_failure=first(a['load_ratio']<.8,a['time']*1000),new_first_load_failure=first(b['load_ratio']<.8,b['time']*1000),old_first_limit=oldlimit,new_first_limit=newlimit,comparisons='strictly before 2.625ms only; later samples archival context, not comparative attribution'),ctrl_activation_checks=checks,initialization_state=initial,decomposition_anchor='S121 old at each same native sample',decomposition_max_abs_residual=decomposition_error,actuator_projection_max_abs_residual=projection_error,independent_contact_projection_max_abs_residual=dict(old=ra,new=rb),at_2p5ms_DOF8=dict(other89_delta=other_sum,act2_delta=act2,total_delta=float(signed.sum()),positive_other89_sum=float(signed[others & (signed>0)].sum()),negative_other89_sum=float(signed[others & (signed<0)].sum()),all89_retained=True,abs_top6=[int(c) for c in top],top6_details=[dict(channel=int(c),name=ma['actuator_names'][c],old_torque=float(T0[idx,c,2]),new_torque=float(T1[idx,c,2]),delta=float(signed[c])) for c in top]),force_active_passive=dict(status='NOT_SAVED / NOT_IDENTIFIABLE',missing_full_parameters=missing,inventory=inventory,reason='Total actuator_force saved, no separately saved active/passive muscle force or full raw gain/bias/acc0/lengthrange parameters; qfrc_passive is not passive muscle force. No XML/model loaded.'),physical_units='NOT_ESTABLISHED',causal_claim=False,new_parameter_proposal=False,calls=dict(model_load=0,forward=0,step=0,neural_tick=0,weight_write=0),analysis_source=str(Path(__file__)),analysis_sha256=sha(__file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,2,figsize=(17,17),sharex=True)
    for ch in top:
        label=f'{ch}: {ma["actuator_names"][ch]}'
        axs[0,0].plot(t,delta[:,ch,2],label=label)
        axs[0,1].plot(t,b['actuator_force'][keep,ch]-a['actuator_force'][keep,ch],label=str(ch))
        for ax,field in ((axs[1,0],'ctrl'),(axs[1,1],'act'),(axs[2,0],'actuator_length'),(axs[2,1],'actuator_velocity')): ax.plot(t,b[field][keep,ch]-a[field][keep,ch],label=str(ch))
    for k,dof in enumerate(DOFS):
        joint=ma['dof_jntid'][dof]; qi=ma['jnt_qposadr'][joint]
        axs[3,0].plot(t,b['qpos'][keep,qi]-a['qpos'][keep,qi],label=f'q DOF{dof}')
        axs[3,1].plot(t,cb[keep,dof]-ca[keep,dof],label=f'contact DOF{dof}')
    titles=['Signed torque delta DOF8; abs top6 other89 incl tiny cancellations','Signed total muscle force delta','Control delta (all other89 verified exact)','Activation delta (all other89 verified exact)','Actuator length delta','Actuator velocity delta','Joint q delta (qvel saved in CSV)','Independent contact-row J.T force delta']
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title);ax.axhline(0,color='gray',lw=.5);ax.axvline(1.925,color='red',ls='--',label='load failure 1.925');ax.axvline(2.625,color='black',ls=':');ax.axvline(2.650,color='gray',ls=':');ax.axvspan(2.625,2.825,color='gray',alpha=.15);ax.grid(alpha=.2);ax.legend(fontsize=6);ax.set_xlabel('native physical time (ms)')
    fig.suptitle('S123 frozen S122 minus S121; signed algebra NOT causal. Shaded >=2.625ms: archival only. Native units NOT_ESTABLISHED.')
    fig.tight_layout();fig.savefig(out/'MUSCLE_COMPENSATION.png',dpi=160);plt.close(fig)
    assert not any(k in sys.modules for k in ('mujoco','torch'))
    after={p:sha(p) for p in frozen}
    assert frozen==after, 'Frozen inputs modified'
    assert all(sha(p)==digest for p,digest in bindings.items())
    summary['all_frozen_before_after_identical']=True
    summary['frozen_file_count']=len(frozen)
    summary['verified_commitment_count']=len(bindings)
    summary['passed']=True
    dump(out/'SUMMARY.json',summary)
    dump(out/'FROZEN_BEFORE_AFTER.json',dict(before=frozen,after=after,all_identical=True,external_commitments=bindings))
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(passed=True,output=str(out),other89_delta=other_sum,act2_delta=act2,top6=[int(c) for c in top],checks=checks),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    analyze(parser.parse_args().out)
