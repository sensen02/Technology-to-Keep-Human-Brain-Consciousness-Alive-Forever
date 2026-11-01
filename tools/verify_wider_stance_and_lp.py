"""New-process saved-array verification; no model, LP, neural or body calls."""
from pathlib import Path
import json,hashlib
import numpy as np
W=Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wider-stance/20261101T205257Z')
L=Path('/home/sensen/Desktop/engine修复验收/s5-stance-lp-feasibility/20261101T205257Z')

def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    table=json.loads((W/'SEED_TABLE.json').read_text());v=json.loads((W/'VARIABLES.json').read_text())
    print('variables',list(v))
    minima={};linecounts={}
    for line in (W/'EVALUATIONS.jsonl').open():
        r=json.loads(line)
        # Inspect logged seed field without making assumptions about native variable order.
        k=str(r.get('seed_k',r.get('retained_fraction_k',r.get('k'))))
        if k=='None':k=str(r.get('seed'))
        linecounts[k]=linecounts.get(k,0)+1
        assert abs(sum(b['sse'] for b in r['blocks'].values())-r['cost'])<1e-10
        if r['role']=='optimizer' and (k not in minima or r['cost']<minima[k]['cost']):minima[k]=r
    checks=[];rows=[]
    for seed in table['seeds']:
        k=seed['retained_fraction_k'];s=json.loads((W/f'SEED_{k}_SUMMARY.json').read_text());z=np.load(W/f'SEED_{k}_FINAL_STATE.npz')
        assert not s['gate_passed']
        assert np.array_equal(z['x'],np.array(minima[str(k)]['x']))
        assert abs(minima[str(k)]['cost']-float(z['residual']@z['residual']))<1e-12
        np.testing.assert_allclose(z['activation'],.0001+.9999*z['group_matrix']@((z['x'][43:]-.0001)/.9999),rtol=0,atol=1e-14)
        assert (z['x']>=np.array(v['lower'])-1e-12).all() and (z['x']<=np.array(v['upper'])+1e-12).all()
        np.testing.assert_allclose(z['moment'].T@z['actuator_force'],z['qfrc_actuator'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(z['compiled_vertex_distances'],z['nonfoot_distances'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(z['residual'][:48],z['qacc']/9810,atol=1e-14,rtol=0)
        assert np.array_equal(z['activation'],z['ctrl']) and np.max(abs(z['act_dot']))<=1e-10
        assert not np.any(z['qfrc_applied']) and not np.any(z['xfrc_applied'])
        assert (z['nonfoot_distances']>=.02).all() and (z['nonfoot_distances']<1000).all()
        assert abs(z['qacc']).max()>.1
        f=z['foot_force'].sum(0);hv=float(np.linalg.norm(f[:2])/f[2]);cost=float(z['residual']@z['residual'])
        actual_pose=z['x'][:42];names=table['hinge_names'];indices=[names.index(n) for n in table['contract_order']]
        rows.append(dict(k=k,SSE=cost,max_qacc=float(abs(z['qacc']).max()),hv=hv,foot_load=float(f[2]/z['weight_native']),seed_projection_count=seed['n_projections'],native_angle_abs_sum_final=float(abs(actual_pose[indices]).sum())))
    assert len(rows)==5
    high=[r['hv'] for r in rows if r['k']>=1.1]
    supported=all(b<=a for a,b in zip(high,high[1:])) and min(high)<.15
    assert not supported
    lp=np.load(L/'BEST_LP.npz');solution=lp['solution'];C=lp['C'];rhs=lp['rhs'];B=lp['B'];y=lp['y'];nc=len(lp['points'])
    inequality_violation=float(np.maximum(C@solution-rhs,0).max());bound_violation=float(max(np.maximum(lp['lower']-solution,0).max(),np.maximum(solution-lp['upper'],0).max()))
    assert inequality_violation<1e-7 and bound_violation<1e-7
    r=B@solution[:B.shape[1]]-y
    np.testing.assert_allclose(r,lp['residual'],atol=1e-12,rtol=0)
    slack=solution[-48:];assert (abs(r)<=slack+1e-7).all()
    result=json.loads((L/'RESULT.json').read_text());assert len(result['history'])==185 and result['counts']['LP']==185
    assert abs(lp['objective']@solution-result['best']['lp_l1'])<1e-9
    assert abs(r).max()>.001 and not result['LP_candidate']
    forces=solution[66:B.shape[1]].reshape(nc,3)
    assert (forces[:,2]>=-1e-8).all()
    assert (abs(forces[:,:2]).sum(1)<=lp['mu']*forces[:,2]+1e-7).all()
    floor=np.full(90,.0001)
    np.testing.assert_allclose(lp['activation'],floor+.9999*lp['group_matrix']@solution[:66],rtol=0,atol=1e-14)
    Wgt=24.446354755878005
    np.testing.assert_allclose(B[:,:66],.9999*lp['A']@lp['group_matrix']/Wgt,rtol=0,atol=1e-12)
    np.testing.assert_allclose(y,(lp['bias']-lp['joint_passive']-lp['baseline'])/Wgt,rtol=0,atol=1e-12)
    assert (lp['gaps']<=.001).all()
    report=dict(wider_rows=rows,wider_monotonic_improvement_supported=supported,LP_checks=dict(inequality_violation=inequality_violation,bound_violation=bound_violation,lp_inf=float(abs(r).max()),lp_l1=float(lp['objective']@solution),contact_points=nc,pose_trials=185),calls=dict(model=0,forward=0,LP=0,body_step=0,neural_step=0),limitations='Primal saved-array verification only, not a dual optimality certificate or independent repeat of the outer search. Wider seeds are initial guesses; k does not constrain optimized footprint.')
    dump(W/'INDEPENDENT_ARRAY_VERIFICATION.json',report);dump(L/'INDEPENDENT_ARRAY_VERIFICATION.json',report)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
