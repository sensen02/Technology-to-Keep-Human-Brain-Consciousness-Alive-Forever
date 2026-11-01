"""Saved dual bookkeeping only; no new LP or mechanics evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s5-stance-contact-relaxations/20261101T205953Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-stance-dual-attribution/20261101T205953Z')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False)
    files=[S/f'LP_{i}_d_diamond_physiological.npz' for i in range(2)]
    dump(O/'PREREGISTRATION.json',dict(scope='Pure saved-dual decomposition, no model/LP/forward/step',source_sha256={str(p):sha(p) for p in [Path(__file__),*files]},limits='Dual terms can be signed and cancel. They are an algebraic optimality account at one fixed problem, not physical force shares or a causal ablation. Bound marginals are local optimal-value sensitivities only while applicable dual face persists.'))
    rows=[]
    for i,p in enumerate(files):
        a=np.load(p);W=float(a['W']);z=a['ineqlin_marginals'];dual_y=z[:48]-z[48:96]
        terms=dict(bias=float(dual_y@a['bias']/W),joint_passive=float(-dual_y@a['joint_passive']/W),muscle_baseline=float(-dual_y@a['baseline']/W))
        lo=a['lower'];hi=a['upper'];lm=a['lower_marginals'];um=a['upper_marginals'];fl=np.isfinite(lo);fu=np.isfinite(hi)
        bound=float(lo[fl]@lm[fl]+hi[fu]@um[fu]);other=float(a['rhs'][96:]@z[96:]);primal=float(a['objective']@a['solution']);total=sum(terms.values())+bound+other
        assert abs(total-primal)<1e-7
        G=a['G'];upper=um[:66];order=np.argsort(upper)[:8]
        pools=[dict(pool=int(j),actuators=np.flatnonzero(G[:,j]).tolist(),upper_bound_marginal=float(upper[j]),normalized_pool_value=float(a['solution'][j])) for j in order if upper[j]<-1e-9]
        # The dual target vector highlights rows that the optimum's witness uses.
        witness=[dict(dof=int(j),dual=float(dual_y[j]),target_over_W=float(a['y'][j]),residual_over_W=float(a['residual'][j])) for j in np.argsort(abs(a['residual']))[::-1][:8]]
        rows.append(dict(pose=i,primal=primal,dual_terms=terms,finite_bound_term=bound,other_rhs_term=other,dual_total=total,duality_gap=abs(primal-total),most_negative_pool_upper_marginals=pools,largest_residual_witness_rows=witness))
    dump(O/'RESULT.json',dict(rows=rows,counts=dict(model_load=0,forward=0,LP=0,body_step=0,neural_step=0)))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
