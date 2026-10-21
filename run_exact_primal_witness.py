"""Construct and verify an EXACT rational primal feasible point at s=21/20 (offline only).

Method: solve the float LP at s=21/20, take its largest-|x| columns as the free set (size n),
round the remaining columns to dyadic rationals, solve the n x n rational system exactly for the
free columns, then verify every constraint exactly. No simulation.
"""
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib,zipfile
import numpy as np
from scipy.optimize import linprog
GATE=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624/native_diamond.npz')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def enc(x):return [str(x.numerator),str(x.denominator)]
def main():
    out=Path('/home/sensen/Desktop/engine修复验收')/('S2-exact-primal-witness-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
    r=dict(status='RUNNING',scope='EXACT_RATIONAL_PRIMAL_FEASIBILITY_AT_S_21_20',offline_only=True)
    hashes={str(GATE):sha(GATE),str(Path(__file__)):sha(Path(__file__))}
    try:
        with np.load(GATE) as z:A,b,lo,hi=z['A'].copy(),z['target'].copy(),z['lo'].copy(),z['hi'].copy()
        ex=lambda x:F(float(x));m=len(lo);n=A.shape[0];S=F(21,20)
        # float LP at s=21/20: primal feasibility A_s x = b, lo<=x_m<=hi, x_c>=0
        Aeq=np.hstack([float(S)*A[:,:m],A[:,m:]])
        sol=linprog(np.zeros(A.shape[1]),A_eq=Aeq,b_eq=b,bounds=[(lo[j],hi[j]) for j in range(m)]+[(0,None)]*(A.shape[1]-m),method='highs')
        assert sol.success,(sol.status,sol.message)
        x0=sol.x[:A.shape[1]]
        r['float_lp']=dict(status=int(sol.status),objective=float(sol.fun),min_muscle_slack=float(min(min(x0[j]-lo[j],hi[j]-x0[j]) for j in range(m))),min_contact=float(min(x0[m:])))
        # exact solve: pin the active set exactly to its bounds, then solve for a full-rank subset
        M=[[ex((S*A[i,j]) if j<m else A[i,j]) for j in range(A.shape[1])] for i in range(n)]
        pinned={};free_all=[]
        for j in range(A.shape[1]):
            if j<m:
                if abs(x0[j]-lo[j])<1e-9:pinned[j]=ex(lo[j])
                elif abs(x0[j]-hi[j])<1e-9:pinned[j]=ex(hi[j])
                else:free_all.append(j)
            else:
                if x0[j]<1e-9:pinned[j]=F(0)
                else:free_all.append(j)
        order=sorted(free_all,key=lambda j:-abs(x0[j]));solve=[];rank=0
        for j in order:
            trial=[[M[i][k] for k in solve]+[M[i][j]] for i in range(n)]
            if np.linalg.matrix_rank(np.array([[float(v) for v in row] for row in trial]),tol=1e-12)>rank:
                solve.append(j);rank+=1
            if rank==n:break
        assert rank==n,'no full-rank solve set'
        flo=[j for j in free_all if j not in solve]
        q=F(2)**26;xr=dict(pinned);xr.update({j:F(int(round(x0[j]*q)),int(q)) for j in flo})
        Aug=[[M[i][j] for j in solve]+[ex(b[i])-sum(M[i][j]*xr[j] for j in pinned)+0-sum(M[i][j]*xr[j] for j in flo)] for i in range(n)]
        for col in range(n):
            piv=next((ii for ii in range(col,n) if Aug[ii][col]!=0),None);assert piv is not None,'singular'
            Aug[col],Aug[piv]=Aug[piv],Aug[col];pv=Aug[col][col];Aug[col]=[v/pv for v in Aug[col]]
            for ii in range(n):
                if ii!=col and Aug[ii][col]!=0:
                    f=Aug[ii][col];Aug[ii]=[a-f*c2 for a,c2 in zip(Aug[ii],Aug[col])]
        for ii,j in enumerate(solve):xr[j]=Aug[ii][n]
        xr={j:xr[j] for j in range(A.shape[1])}
        # exact verification
        resid=[sum(M[i][j]*xr[j] for j in range(A.shape[1]))-ex(b[i]) for i in range(n)]
        bounds_ok=all(ex(lo[j])<=xr[j]<=ex(hi[j]) for j in range(m)) and all(xr[j]>=0 for j in range(m,A.shape[1]))
        r['exact_witness']=dict(scale=enc(S),rest_denominator=enc(q),
            equality_residual_exactly_zero=all(v==0 for v in resid),bounds_hold_exactly=bounds_ok,
            feasible_exactly=bool(all(v==0 for v in resid) and bounds_ok),
            min_muscle_slack_float=float(min(min(xr[j]-ex(lo[j]),ex(hi[j])-xr[j]) for j in range(m))),
            min_contact_float=float(min(xr[j] for j in range(m,A.shape[1]))),
            x_numerators=[enc(xr[j]) for j in range(A.shape[1])],pinned_columns=sorted(pinned),free_solve_columns=solve,floating_columns=flo)
        r['status']='PASS_EXACT_WITNESS_CONSTRUCTED' if r['exact_witness']['feasible_exactly'] else 'FAIL_EXACT_WITNESS_NOT_FEASIBLE'
    except Exception as e:
        import traceback;r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc()[-800:])
    finally:
        r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=hashes=={p:sha(p) for p in hashes}
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(out);print(r['status']);print(json.dumps({k:v for k,v in r.get('exact_witness',{}).items() if k!='x_numerators'},indent=1)[:500]);print(r.get('error',''))
    return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
