"""S124 independent scalar MuJoCo3.9 muscle arithmetic; no simulator import."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as E
import numpy as np
B=Path('/home/sensen/Desktop/engine修复验收');OUT=B/'s124-muscle-reconstructability/20261024T075500Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fl(l,lo,hi):
 if l<lo or l>hi:return 0.
 a=(lo+1)*.5;b=(1+hi)*.5
 if l<=a:return .5*((l-lo)/max(1e-15,a-lo))**2
 if l<=1:return 1-.5*((1-l)/max(1e-15,1-a))**2
 if l<=b:return 1-.5*((l-1)/max(1e-15,b-1))**2
 return .5*((hi-l)/max(1e-15,hi-b))**2
def evaluate(length,vel,act,node):
 g=list(map(float,node.get('gainprm').split()));p=list(map(float,node.get('biasprm').split()));lr=list(map(float,node.get('lengthrange').split()))
 assert g[2]>0 and p[2]>0
 l0=(lr[1]-lr[0])/max(1e-15,g[1]-g[0]);l=g[0]+(length-lr[0])/max(1e-15,l0);v=vel/max(1e-15,l0*g[6]);y=g[8]-1
 if v<=-1:fv=0.
 elif v<=0:fv=(v+1)**2
 elif v<=y:fv=g[8]-(y-v)**2/max(1e-15,y)
 else:fv=g[8]
 gain=-g[2]*fl(l,g[4],g[5])*fv
 l0b=(lr[1]-lr[0])/max(1e-15,p[1]-p[0]);lb=p[0]+(length-lr[0])/max(1e-15,l0b);mid=(1+p[5])*.5
 if lb<=1:bias=0.
 elif lb<=mid:bias=-p[2]*p[7]*.5*((lb-1)/max(1e-15,mid-1))**2
 else:bias=-p[2]*p[7]*(.5+(lb-mid)/max(1e-15,mid-1))
 return gain*act+bias,gain,bias
def main():
 nodes=E.parse(XML).getroot().find('actuator');maxerr=0.;deriverror=0.;records=[]
 for rel in ('s121-constraint-row-recording/20261024T072500Z/init80ms','s122-secant-constraint-pair/20261024T073500Z/w36b4_secant'):
  p=B/rel;meta=json.loads((p/'MODEL_PROVENANCE.json').read_text())['before'];arr=np.array([list(map(float,n.get('lengthrange').split())) for n in nodes],dtype=np.float64);assert hashlib.sha256(arr.tobytes()).hexdigest()==meta['model.actuator_lengthrange']['sha256']
  for key in ('gainprm','biasprm','dynprm'):
   vals=[list(map(float,n.get(key).split())) for n in nodes];a=np.array([v+[0.]*(10-len(v)) for v in vals]);assert hashlib.sha256(a.tobytes()).hexdigest()==meta['model.actuator_'+key]['sha256']
  with np.load(p/'TRAJECTORY.npz') as z:
   armerr=0.
   for i in range(121):
    for j in range(6):
     l,v,a=z['actuator_length'][i,j],z['actuator_velocity'][i,j],z['act'][i,j];f,g,b=evaluate(l,v,a,nodes[j]);armerr=max(armerr,abs(f-z['actuator_force'][i,j]));eps=1e-5
     fd=(evaluate(l,v,a+eps,nodes[j])[0]-evaluate(l,v,a-eps,nodes[j])[0])/(2*eps);deriverror=max(deriverror,abs(fd-g)/max(1.,abs(g)))
   maxerr=max(maxerr,armerr);records.append(dict(arm=rel,maxforceerror=armerr))
 assert maxerr<1e-10 and deriverror<1e-8
 result=dict(passed=True,verified_force_samples=1452,max_abs_force_error=maxerr,derivative_finite_difference_max_relative=deriverror,per_arm=records,xml_sha256=sha(XML),verifier_sha256=sha(__file__),model_loads=0,forward=0,steps=0)
 (OUT/'PARENT_INDEPENDENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
