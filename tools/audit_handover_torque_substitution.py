"""Frozen-state substitution bounds; never use fitted activations as control."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.optimize import lsq_linear
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
M=Path('/home/sensen/Desktop/engine修复验收/s3-mn-muscle-chain/20261101T184041Z')
T=Path('/home/sensen/Desktop/engine修复验收/s5-source-training-handover/20261028T130000Z')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    assert (out/'PREREGISTRATION.json').exists() and not (out/'RESULT.json').exists()
    files=[M/'CHAIN_ARRAYS.npz',M/'AUDIT.json',M/'MANIFEST_SHA256.json',S/'MODEL.xml',S/'HANDOVER_BEFORE.npz',S/'HANDOVER_AFTER.npz',S/'MAPPING.json',T/'MODELBUILD/MODEL.xml',T/'TRAINING/TORQUE_IMITATION.json',T/'TRIAL/HANDOVER_RULE.json',Path(__file__)]
    hashes={str(p):sha(p) for p in files}
    expected='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
    assert hashes[str(S/'MODEL.xml')]==expected and hashes[str(T/'MODELBUILD/MODEL.xml')]==expected
    manifest=json.loads((M/'MANIFEST_SHA256.json').read_text())
    for p in [M/'CHAIN_ARRAYS.npz',M/'AUDIT.json']:assert hashes[str(p)]==manifest[str(p)]
    z=np.load(M/'CHAIN_ARRAYS.npz');before=np.load(S/'HANDOVER_BEFORE.npz');after=np.load(S/'HANDOVER_AFTER.npz')
    for key in ['qpos','qvel','act','time']:assert np.array_equal(before[key],after[key])
    assert abs(float(before['time'])-1.3)<1e-10
    mapping=json.loads((S/'MAPPING.json').read_text())['rows'];groups={}
    for r in mapping:
        if r['local_indices']:groups.setdefault(tuple(r['local_indices']),[]).append(r['actuator_index'])
    assert len(groups)==66
    mapped=[r['actuator_index'] for r in mapping if r['local_indices']]
    assert len(mapped)==74
    G=np.zeros((90,66))
    for j,ix in enumerate(groups.values()):G[ix,j]=1
    A=z['unit_activation_torque'][0].T
    passive=z['moment'][0].T@z['passive_force'][0]
    actual=z['actual_active_torque'][0].sum(0)+passive
    np.testing.assert_allclose(A@before['act']+passive,actual,rtol=0,atol=1e-10)
    assert abs(A[:6]).max()<1e-10 and abs(before['qfrc_applied'][:6]).max()==0
    assist=before['qfrc_applied'];total=actual+assist
    floor=np.full(90,.0001);base=passive+A@floor
    fits=[];raw={}
    for target_name,target in [('total_replacement',total),('auxiliary_only',assist)]:
        for permission,P in [('tied66',G),('independent74',np.eye(90)[:,mapped]),('relaxed90',np.eye(90))]:
            B=A[6:]@P;y=target[6:]-base[6:]
            fit=lsq_linear(B,y,bounds=(0.,.9999),method='bvls',tol=1e-12,max_iter=1000)
            act=floor+P@fit.x;prediction=passive+A@act;r=prediction[6:]-target[6:];grad=B.T@r
            violation=np.where(fit.x<=1e-8,np.minimum(grad,0),np.where(fit.x>=.9999-1e-8,np.maximum(grad,0),grad))
            result=dict(target=target_name,permission=permission,success=bool(fit.success),status=int(fit.status),
                message=fit.message,residual_l2=float(np.linalg.norm(r)),
                residual_relative_to_target=float(np.linalg.norm(r)/max(np.linalg.norm(target[6:]),1e-12)),
                independent_kkt_violation=float(abs(violation).max()),
                active_variables=int((fit.x>1e-8).sum()),upper_bound_variables=int((fit.x>.9999-1e-8).sum()))
            assert fit.success
            fits.append(result);raw[target_name+'_'+permission+'_activation']=act
            raw[target_name+'_'+permission+'_residual']=r
    history=json.loads((T/'TRAINING/TORQUE_IMITATION.json').read_text())
    result=dict(source_sha256=hashes,model_sha256=expected,source_state_s=float(before['time']),
        historical_training_signal=history['signal'],
        historical_handover_rule=json.loads((T/'TRIAL/HANDOVER_RULE.json').read_text()),
        actual_muscle_torque_norm=float(np.linalg.norm(actual[6:])),
        auxiliary_torque_norm=float(np.linalg.norm(assist[6:])),
        total_replacement_torque_norm=float(np.linalg.norm(total[6:])),
        same_state_decomposition_error=float(abs(A@before['act']+passive-actual).max()),
        fits=fits,calls=dict(model_load=0,forward=0,body_step=0,neural_step=0),
        interpretation='Correct instantaneous substitution target is actual muscle plus held auxiliary torque at this fixed state; historical direct imitation is a different objective. These relaxed static fits neither certify standing nor prove all autonomous movement impossible.')
    assert hashes=={str(p):sha(p) for p in files}
    np.savez_compressed(out/'FIT_ARRAYS.npz',unit_activation_torque=A,passive_torque=passive,
        actual_torque=actual,auxiliary_torque=assist,total_replacement_torque=total,group_matrix=G,**raw)
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
