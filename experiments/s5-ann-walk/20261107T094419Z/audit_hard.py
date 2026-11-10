"""Independent138-row hard-contact QP audit, using saved states and FK only.
No QP/controller imports or physics steps. Original120constraints and added18
native contact-acceleration equalities are explicitly checked separately.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T092817Z/audit_qp.py'
EXPECTED='ffe401295641d3e282d8be0a68f469810dcd6965976f7fda219e5deef97da8e5'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T094419Z')


def patch(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path,nargs='?',default=OUT/'hard_contact_screen')
    ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv)
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    text=patch(raw.decode(),"    assert all(sha(path)==digest for path,digest in before.items())",
        "    globals()['_LAST_REPORT']=report\n    assert all(sha(path)==digest for path,digest in before.items())")
    old="        residual=M@acc+bias-S@tau-modeled"
    new=old+"\n        hard=(np.einsum('lij,j->li',J,acc)+jdv)*active[:,None]\n        scale=np.r_[np.full(3,9801.),np.full(45,100.)]\n        norm=np.maximum(np.linalg.norm(J*scale[None,None,:],axis=2),1e-8)\n        globals()['_HARD'].append(dict(time=float(t),native=hard.copy(),normalized=hard/norm,\n            log_error=err(hard,qp['hard_contact_acceleration_residual'][k]),\n            actual=(np.einsum('lij,j->li',J,d.qacc)+jdv)*active[:,None]))"
    text=patch(text,old,new)
    ns={'__name__':'independent_hard_contact_audit','__file__':str(SOURCE),'_HARD':[]}
    exec(compile(text,str(SOURCE),'exec'),ns)
    code=ns['main']([str(args.folder),'--no-write'])
    if code==3:return code
    r=ns['_LAST_REPORT'];load=ns['load'];sha=ns['sha'];stats=ns['stats'];folder=args.folder
    qp=load(folder/'QP.npz');protocol=json.loads((folder/'PREREGISTRATION.json').read_text());hard=protocol['hard_contact']
    assert hard['total_constraints']==138 and hard['rows']==[120,138]
    assert protocol['solver']['fixed_constraints']==138
    tol=float(hard['native_acceleration_tolerance']);normtol=float(protocol['parameters']['constraint_tolerance'])
    rows=ns['_HARD'];native=np.asarray([x['native'] for x in rows]);normalized=np.asarray([x['normalized'] for x in rows])
    actual=np.asarray([x['actual'] for x in rows]);logs=np.asarray([x['log_error'] for x in rows]);solved=~qp['fallback'].astype(bool)
    assert len(rows)==len(qp['time']) and native.shape[1:]==(6,3)
    active=qp['activelegs'].astype(bool);inactive_zero=bool(np.all(native[~active]==0))
    solved_native=float(abs(native[solved]).max()) if solved.any() else None
    solved_normalized=float(abs(normalized[solved]).max()) if solved.any() else None
    added=dict(declared_138_constraints=True,all18_logged_vector_entries_match=bool(logs.max()<=1e-6),
        inactive_rows_exact_zero=inactive_zero,
        solved_native_contact_acceleration=bool(solved.any() and solved_native<=tol),
        solved_normalized_added_rows=bool(solved.any() and solved_normalized<=normtol))
    original=r['checks'].pop('solved_hard_constraints_consistent');r['checks']['original_120_constraints_consistent']=original
    r['checks'].update(added);r['checks']['all_138_hard_model_constraints_consistent']=bool(original and all(added.values()))
    adaptation=protocol['adaptation'];registered={path:Path(path).exists() and sha(path)==digest for path,digest in (
        (adaptation['source'],adaptation['sha256']),(adaptation['effective_source'],adaptation['effective_sha256']),
        (protocol['baseline']['source'],protocol['baseline']['sha256']))}
    r['checks']['hard_controller_and_effective_source_unchanged']=all(registered.values())
    first=lambda mask:float(qp['time'][np.flatnonzero(mask)[0]]) if np.any(mask) else None
    selected=sorted(set(int(abs(qp['time']-t).argmin()) for t in (1.,1.05,1.1,1.15,1.2,1.3,1.5,2.,3.)))
    details=[dict(time=float(qp['time'][i]),fallback=bool(qp['fallback'][i]),active_legs=int(active[i].sum()),
        predicted_hard_acceleration_max=float(abs(native[i]).max()),actual_active_contact_acceleration_max=float(abs(actual[i]).max()),
        logged_native_residual_error=float(logs[i])) for i in range(len(rows))]
    r.update(stage='S5 independent138-constraint hard-contact QP audit',status='HARD_MODEL_CONSISTENT_ONLY' if all(r['checks'].values()) else 'HARD_MODEL_AUDIT_FAILED',
        actual_QP_goal='NOT_ESTABLISHED',added_18_constraints=dict(native_tolerance=tol,normalized_row_tolerance=normtol,
            all_ticks_native=stats(native),solved_ticks_native_max=solved_native,solved_ticks_normalized_max=solved_normalized,
            log_reconstruction_max=float(logs.max()),actual_contact_acceleration=stats(actual),
            first_solved_native_violation=first(solved&(np.max(abs(native),axis=(1,2))>tol)),
            first_actual_active_contact_acceleration_above1=first(np.max(abs(actual),axis=(1,2))>1),
            note='Actual contact acceleration flag1 is diagnostic only; native hard-model acceptance uses declared1e-3. Actual MuJoCo contact remains compliant.'),
        selected_hard_contact_diagnostics=[details[i] for i in selected],hard_contact_policy_diagnostics=details,
        hard_contact_registered_sources_unchanged=registered,
        audit_source_adaptation=dict(source=str(SOURCE),source_sha256=EXPECTED,effective_audit_sha256=hashlib.sha256(text.encode()).hexdigest(),
            changes='Capture report and append explicit native/normalized6x3 contact-acceleration checks inside already independently reconstructed contact-J loop; no original model equations changed.'),
        limitations=r['limitations']+['A138-row solved planning model is not proof actual torque/contact realization. Rigidzeroacceleration is an extra teacher assumption; actual motor/contact dynamics and fallback remain separate evidence.'])
    r['input_hashes'][str(Path(__file__))]=sha(__file__)
    if not args.no_write:
        path=OUT/'HARD_CONTACT_AUDIT.json'
        if path.exists():path.read_bytes()
        path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t=qp['time'];fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
        axes[0].semilogy(t,np.maximum(np.max(abs(native),axis=(1,2)),1e-16),label='predicted hard contact acceleration max');axes[0].axhline(tol,color='r',ls=':',label='native hard tolerance');axes[0].legend()
        axes[1].plot(t,np.max(abs(actual),axis=(1,2)),label='actual loaded material-contact acceleration max');axes[1].legend()
        axes[2].plot(t,[d['optimal_vs_actual_endpoint_rms'] for d in r['policy_diagnostics']],label='optimal vs actual endpoint torque RMS');axes[2].fill_between(t,0,1,where=~solved,color='r',alpha=.2,label='fallback');axes[2].legend();axes[2].set_xlabel('Policy time seconds')
        fig.suptitle('All138planningconstraints are audited; actual realization is a separate failed physical question')
        fig.savefig(OUT/'HARD_CONTACT_AUDIT.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status=r['status'],checks=r['checks'],solver=r['solver'],added18=r['added_18_constraints'],early=r['early_failure_timing'],selected=r['selected_policy_diagnostics'],physics_steps=0)),flush=True)
    return 0 if all(r['checks'].values()) and not qp['fallback'].any() else 2


if __name__=='__main__':raise SystemExit(main())
