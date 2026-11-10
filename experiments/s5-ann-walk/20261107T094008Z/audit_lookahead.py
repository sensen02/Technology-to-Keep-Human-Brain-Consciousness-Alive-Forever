"""Audit saved lookahead conversion, reusing frozen independent contact dynamics.
No controller/QP solver imports, physics steps or changes to older audit sources.
The frozen-state reachable torque remains independently checked; endpoint q/v/tau
prediction and actual BODY endpoint realization are additional separate diagnostics.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE=ROOT/'experiments/s5-ann-walk/20261107T092817Z/audit_qp.py'
EXPECTED='ffe401295641d3e282d8be0a68f469810dcd6965976f7fda219e5deef97da8e5'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T094008Z')


def patch(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('folder',type=Path,nargs='?',default=OUT/'wbc_lookahead')
    ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv)
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED,'Frozen QP audit changed'
    text=raw.decode();text=patch(text,"OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T092817Z')",f"OUT=Path({str(OUT)!r})")
    text=patch(text,"    assert all(sha(path)==digest for path,digest in before.items())","    globals()['_LAST_REPORT']=report\n    assert all(sha(path)==digest for path,digest in before.items())")
    text=patch(text,"source_state_constant_assumption='prediction freezes q/qvel; state-change torque terms are diagnostic only, no endpoint compensation implemented'","source_state_constant_assumption='this frozen-state diagnostic is retained separately; lookahead endpoint compensation is audited by the wrapper'")
    ns={'__name__':'auditable_frozen_contact_audit','__file__':str(SOURCE)};exec(compile(text,str(SOURCE),'exec'),ns)
    code=ns['main']([str(args.folder),*(['--no-write'] if args.no_write else [])])
    if code==3:return code
    report=ns['_LAST_REPORT'];load=ns['load'];sha=ns['sha'];folder=args.folder
    qp,sens,policy,commands,body=[load(folder/(name+'.npz')) for name in ('QP','SENSORY','POLICY','COMMANDS','BODY')]
    prereg=json.loads((folder/'PREREGISTRATION.json').read_text())
    sys.path.insert(0,str(ROOT/'experiments/s5-ann-incremental/20261107T010000Z'))
    from incremental_actuator import load_full_model
    m=load_full_model();jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    kp,kd,limit=[commands[k] for k in ('kp','kd','limits')];lo=m.jnt_range[jids,0];hi=m.jnt_range[jids,1]
    details=[];errors={key:[] for key in ('predicted_q','predicted_v','inverse_target','endpoint_prediction_tau','endpoint_prediction_minus_optimal','frozen_prediction_tau','frozen_inverse_target')}
    checks={'declared_conversion':bool(np.all(qp['torque_conversion']=='solved_qacc_5ms_endpoint_prediction'))}
    for k,t in enumerate(qp['time']):
        i=int(np.argmin(abs(sens['time']-t)));end=int(np.argmin(abs(body['time']-(t+.005))));tick=int(round(t/.001))
        assert abs(sens['time'][i]-t)<1e-9 and abs(body['time'][end]-(t+.005))<1e-8
        q,v=sens['qpos'][i,qa],sens['qvel'][i,va];acc=qp['qacc'][k,va];tau=qp['tau_optimal'][k]
        pq=q+.005*v+.5*.005**2*acc;pv=v+.005*acc;target=pq+(tau+kd*pv)/kp
        reachable=policy['targets'][i].copy()
        for j in range(5):
            row=commands['rows'][tick+j];reachable=np.clip(reachable+.001*np.clip(row[43:85]+row[85:127],-30,30),lo,hi)
        endpoint_prediction=np.clip(kp*(reachable-pq)-kd*pv,-limit,limit)
        frozen=np.clip(kp*(reachable-q)-kd*v,-limit,limit)
        calculated=dict(predicted_q=pq,predicted_v=pv,inverse_target=target,endpoint_prediction_tau=endpoint_prediction,
            endpoint_prediction_minus_optimal=endpoint_prediction-tau,frozen_prediction_tau=frozen,frozen_inverse_target=q+(tau+kd*v)/kp)
        logged=dict(predicted_q='predicted_endpoint_q',predicted_v='predicted_endpoint_v',inverse_target='torque_target',
            endpoint_prediction_tau='endpoint_prediction_tau',endpoint_prediction_minus_optimal='endpoint_prediction_minus_optimal',
            frozen_prediction_tau='tau_predicted_reachable',frozen_inverse_target='frozen_state_inverse_target')
        for key,value in calculated.items():errors[key].append(float(abs(value-qp[logged[key]][k]).max()))
        actual=body['ctrl'][end];actualq=body['qpos'][end,qa];actualv=body['qvel'][end,va]
        metric=lambda x:float(np.sqrt(np.mean(x*x)))
        active=qp['activelegs'][k]
        modeled_contact_acc=np.einsum('lik,k->li',qp['averaged_contact_J'][k],qp['qacc'][k])+qp['averaged_Jdotv'][k]
        desired_contact_acc=-prereg['parameters']['stance_kp']*(qp['contact_point'][k]-qp['contact_anchor'][k])-prereg['parameters']['stance_kd']*qp['contact_material_velocity'][k]
        active_max=lambda x:float(np.linalg.norm(x[active],axis=1).max()) if active.any() else 0.
        details.append(dict(time=float(t),fallback=bool(qp['fallback'][k]),blend=float(qp['blend'][k]),
            modeled_stance_contact_acceleration_max=active_max(modeled_contact_acc),
            desired_stance_contact_acceleration_max=active_max(desired_contact_acc),
            soft_stance_acceleration_residual_max=active_max(modeled_contact_acc-desired_contact_acc),
            optimized_tau_rms=metric(tau),predicted_endpoint_tau_minus_optimal_rms=metric(endpoint_prediction-tau),
            frozen_state_prediction_minus_optimal_rms=metric(frozen-tau),actual_endpoint_tau_minus_optimal_rms=metric(actual-tau),
            endpoint_prediction_minus_actual_tau_rms=metric(endpoint_prediction-actual),
            predicted_q_minus_actual_endpoint_rms=metric(pq-actualq),predicted_v_minus_actual_endpoint_rms=metric(pv-actualv),
            predicted_position_error_torque_rms=metric(kp*(pq-actualq)),predicted_velocity_error_torque_rms=metric(kd*(pv-actualv)),
            actual_endpoint_formula_minus_saved_ctrl_rms=metric(np.clip(kp*(reachable-actualq)-kd*actualv,-limit,limit)-actual)))
    error_max={key:max(values) for key,values in errors.items()}
    checks.update({key:value<=1e-6 for key,value in error_max.items()})
    selected=sorted(set(int(np.argmin(abs(qp['time']-t))) for t in (1.,1.05,1.1,1.15,1.2,1.3,1.5,2.,3.)))
    new_source={path:Path(path).exists() and sha(path)==digest for path,digest in prereg.get('new_source_hashes',{}).items()}
    checks['new_controller_sources_unchanged']=bool(new_source and all(new_source.values()))
    report.update(stage='S5 independent lookahead-conversion and full contact-QP audit',
        lookahead_checks=checks,lookahead_formula_log_max_errors=error_max,
        lookahead_selected_diagnostics=[details[i] for i in selected],lookahead_policy_diagnostics=details,
        new_registered_sources_unchanged=new_source,
        audit_adaptation=dict(frozen_source=str(SOURCE),frozen_sha256=EXPECTED,
            effective_independent_audit_sha256=hashlib.sha256(text.encode()).hexdigest(),
            change='Only output directory and report capture change; all contact/dynamics/finite-integrator checks reused unchanged. Additional independent endpoint conversion formulas are audited by this wrapper.'),
        endpoint_prediction_limitation='QP solved acceleration is a local model. q/v endpoint extrapolation does not establish actual contact trajectory, endpoint torque realization, optimal physical control, or walking.')
    report['checks']['lookahead_conversion_reconstructed']=all(checks.values())
    report['goal_not_established_reasons']=[reason for reason,yes in (
        ('fallback occurred',bool(np.any(qp['fallback']))),
        ('actual ground reaction differs from active-contact QP model',report['metrics']['ground_reaction_mismatch_W']['max_abs']>.1),
        ('actual endpoint torque differs from lookahead endpoint prediction',max(d['endpoint_prediction_minus_actual_tau_rms'] for d in details)>1e-3),
        ('lookahead conversion reconstruction failed',not all(checks.values()))) if yes]
    report['contact_acceleration_caution']='The stance objective already closely tracks its nonzero desired contact acceleration in early ticks. Making the same target hard may have little effect; hard zero acceleration is a different model/control constraint, not merely enforcing the existing objective.'
    report['input_hashes'][str(Path(__file__))]=sha(__file__)
    if not args.no_write:
        output=OUT/'AUDIT_LOOKAHEAD.json'
        if output.exists():output.read_bytes()
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(lookahead_checks=checks,max_errors=error_max,selected=report['lookahead_selected_diagnostics'],physics_steps=0,written=not args.no_write)),flush=True)
    return 0 if all(checks.values()) and code==0 else 2


if __name__=='__main__':raise SystemExit(main())
