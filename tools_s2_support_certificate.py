"""Exact rational separation of saved float LP; no MuJoCo or behavior."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parent
SRC=Path('/home/sensen/Desktop/engine修复验收/S2-support-gate-20261014T172624')
# Declared optional override so the identical exact machinery can run on another gate output.
import sys as _sys
if len(_sys.argv)>1: SRC=Path(_sys.argv[1])
KIND=_sys.argv[2] if len(_sys.argv)>2 else 'diamond'   # declared: diamond|outer|union

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(x): return F(float(x))
def encode(x): return [str(x.numerator),str(x.denominator)]
def dot(a,b): return sum((x*y for x,y in zip(a,b)),F(0))

def reconstruct(rows,rhs,n):
 basis={};numeric=[]
 for row,v in zip(rows,rhs):
  norm=float(np.linalg.norm(row))
  if norm==0: continue
  unit=np.asarray(row)/norm
  trial=np.array(numeric+[unit])
  if np.linalg.matrix_rank(trial,tol=1e-9)<=len(numeric): continue
  numeric.append(unit)
  a=[rat(x) for x in row];b=rat(v)
  for j,(q,c) in sorted(basis.items()):
   if a[j]:
    f=a[j];a=[x-f*y for x,y in zip(a,q)];b-=f*c
  piv=next((j for j,x in enumerate(a) if x),None)
  if piv is None: continue
  f=a[piv];a=[x/f for x in a];b/=f;basis[piv]=(a,b)
  if len(basis)==n: break
 if len(basis)!=n: raise ValueError('active candidate lacks full rank')
 x=[F(0)]*n
 for j,(a,b) in sorted(basis.items(),reverse=True): x[j]=b-dot(a[j+1:],x[j+1:])
 return x

def verify(A,b,lo,hi,y):
 cols=[[rat(v) for v in A[:,j]] for j in range(A.shape[1])]
 products=[dot(c,y) for c in cols];bound=sum((max(rat(l)*v,rat(h)*v) for l,h,v in zip(lo,hi,products)),F(0));required=dot([rat(v) for v in b],y);gap=required-bound
 ok=all(v<=0 for v in products[len(lo):]) and gap>0
 return dict(valid=ok,gap=encode(gap),gap_float=float(gap),required_float=float(required),upper_float=float(bound),max_contact_dot_float=float(max(products[len(lo):],default=F(0))),strict_positive_contact_columns=sum(v>0 for v in products[len(lo):])),products

def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/('S2-support-certificate-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 inputs=[SRC/(f+'_'+KIND+'.npz') for f in ('native','zero')];hashes={str(p):sha(p) for p in inputs};results=[]
 for floor,p in zip(('native','zero'),inputs):
  with np.load(p,allow_pickle=False) as z: A=z['A'];b=z['target'];scale=z['scale'];lo=z['lo'];hi=z['hi']
  S=A/scale[:,None];t=b/scale;n=len(b);m=len(lo)
  U=np.vstack([np.hstack([S[:,m:].T,np.zeros((A.shape[1]-m,m))]),np.hstack([lo[:,None]*S[:,:m].T,-np.eye(m)]),np.hstack([hi[:,None]*S[:,:m].T,-np.eye(m)])])
  res=linprog(np.r_[-t,np.ones(m)],A_ub=U,b_ub=np.zeros(len(U)),bounds=[(-1,1)]*n+[(None,None)]*m,method='highs',options=dict(time_limit=10,maxiter=10000,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
  entry=dict(floor=floor,input=str(p),input_sha256=hashes[str(p)],solver_status=int(res.status),status='INDETERMINATE')
  if res.success:
   w=res.x[:n];y0=w/scale;rows=[];rhs=[]
   for j in range(m,A.shape[1]):
    if abs(S[:,j]@w)<1e-8: rows.append(A[:,j]);rhs.append(0.)
   for j in range(m):
    if abs(S[:,j]@w)<1e-8: rows.append(A[:,j]);rhs.append(0.)
   for j in range(n):
    if abs(abs(w[j])-1)<1e-8:
     row=np.zeros(n);row[j]=scale[j];rows.append(row);rhs.append(float(np.sign(w[j])))
   # Add exact candidate coordinates only if active equations underdetermine.
   for j in range(n):
    row=np.zeros(n);row[j]=1;rows.append(row);rhs.append(float(y0[j]))
   y=reconstruct(rows,rhs,n)
   # Repair tiny positive unbounded-ray products using an EXACT vertical separator.
   # Only valid when every saved contact/limit column has strictly positive row2.
   _,pre=verify(A,b,lo,hi,y)
   positives=[(j,v) for j,v in enumerate(pre[m:],m) if v>0]
   correction=F(0)
   if positives and all(rat(A[2,j])>0 for j in range(m,A.shape[1])):
    correction=max(v/rat(A[2,j]) for j,v in positives)
    y[2]-=correction
   check,prod=verify(A,b,lo,hi,y);entry.update(check)
   entry['exact_vertical_repair']=encode(correction)
   entry.update(y_rational=[encode(v) for v in y],y_float=[float(v) for v in y],candidate_dual_objective=float(-res.fun),rational_scope='exact binary float64 matrix only, NOT interval physical model')
   if check['valid']: entry['status']='EXACT_STORED_MATRIX_SEPARATION'
   entry['top_weighted_rows']=sorted([dict(dof=i,weight=float(v),weighted_target=float(v*rat(b[i]))) for i,v in enumerate(y)],key=lambda x:abs(x['weight']),reverse=True)[:12]
  results.append(entry)
  (out/'certificate.json').write_text(json.dumps(dict(results=results,input_hashes=hashes,script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'S2_SUPPORT_CERTIFICATE_PROTOCOL.zh-CN.md')),indent=2)+'\n')
 assert hashes=={p:sha(p) for p in hashes}
 print(out);print([(r['floor'],r['status'],r.get('gap_float')) for r in results])
 return 0 if all(r['status']=='EXACT_STORED_MATRIX_SEPARATION' for r in results) else 1
if __name__=='__main__': raise SystemExit(main())
