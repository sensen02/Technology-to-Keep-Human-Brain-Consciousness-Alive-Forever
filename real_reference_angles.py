"""Unsigned three-point geometric angle reference, not anatomical DOFs or control."""
from pathlib import Path
import numpy as np,json,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from read_real_sixleg_reference import S,LEGS
O=Path('/home/sensen/Desktop/engine修复验收/s4-real-geometric-angles/20261027T215000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def angles(points):
 a,b,c=np.moveaxis(np.asarray(points,dtype=float),-2,0);u=a-b;v=c-b;nu=np.linalg.norm(u,axis=-1);nv=np.linalg.norm(v,axis=-1);valid=np.isfinite(points).all(axis=(-2,-1))&(nu>0)&(nv>0);out=np.full(valid.shape,np.nan);un=np.zeros_like(u);vn=np.zeros_like(v);np.divide(u,nu[...,None],out=un,where=valid[...,None]);np.divide(v,nv[...,None],out=vn,where=valid[...,None]);out[valid]=np.arctan2(np.linalg.norm(np.cross(un,vn),axis=-1)[valid],np.sum(un*vn,axis=-1)[valid]);return out,valid

def main():
 O.mkdir(parents=True,exist_ok=False);names=json.loads((S/'info__kp_names.json').read_text());raw=np.load(S/'bout_0000__kp_data.npy');p=raw.reshape(-1,50,3);sk=json.loads((S/'SKELETON.json').read_text());edges={frozenset((sk['node_names'][a],sk['node_names'][b])) for a,b in sk['edges']};triples=[[l+'_Tro',l+'_FeTi',l+'_TiTa'] for l in LEGS]
 for a,b,c in triples:assert frozenset((a,b)) in edges and frozenset((b,c)) in edges
 idx=np.array([[names.index(n) for n in triple] for triple in triples]);points=p[:,idx,:];theta,valid=angles(points);frame=np.arange(6555,7240);assert len(p)==len(frame);inputs=[S/'bout_0000__kp_data.npy',S/'info__kp_names.json',S/'SKELETON.json',S/'REFERENCE_METADATA.json',Path('/home/sensen/Desktop/engine修复验收/s4-fulltext-evidence/20261027T214000Z/FULLTEXT_CONTRACT.json'),Path(__file__)];h={str(f):sha(f) for f in inputs};sm=json.loads((S/'FINAL_MANIFEST_SHA256.json').read_text())
 for f in inputs:
  if f.parent==S:assert sha(f)==sm[f.name]
 tests=[];r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
 for label,t in [('translation',points.astype(float)+[11,-7,2]),('rotation',points.astype(float)@r.T),('positive_scale',points.astype(float)*3.7)]:
  a,m=angles(t);np.testing.assert_array_equal(m,valid);np.testing.assert_allclose(a,theta,atol=2e-13,rtol=0);tests.append(dict(test=label,passed=True,max_abs_rad=float(np.nanmax(abs(a-theta)))))
 synthetic=np.array([[[1,0,0],[0,0,0],[0,1,0]],[[1,0,0],[0,0,0],[-1,0,0]],[[0,0,0],[0,0,0],[1,0,0]],[[np.nan,0,0],[0,0,0],[1,0,0]],[[np.inf,0,0],[0,0,0],[1,0,0]]]);a,m=angles(synthetic);np.testing.assert_allclose(a[:2],[np.pi/2,np.pi]);assert m.tolist()==[True,True,False,False,False] and np.isnan(a[2:]).all();tests.append(dict(test='known_angles_zero_nonfinite_mask',passed=True))
 np.savez_compressed(O/'ANGLES.npz',source_frame=frame,raw_original_array=raw,angle_rad=theta,angle_deg=np.rad2deg(theta),valid=valid,triplet_indices_in_original_names=idx);dump(O/'TRIPLETS_AND_SCOPE.json',dict(legs=LEGS,triplets=triples,vertex='middle FeTi point',topology_path=str(S/'SKELETON.json'),skeleton_node_order='resolve edges via skeleton names, not array indices; array uses separately saved original names',scope='unsigned three-point interior angle only, not full anatomical DOF or signed flexion',source_frame=[6555,7239],nominal_rate='800Hz context only, not used in computation',unknowns=['physical units','source axes','record behavior annotation','data license','ground/contact'],invariance_limit='rigid translation/rotation/positive uniform-scale invariance does not remove nonrigid reconstruction errors or unknown preprocessing',no_interpolation=True,no_smoothing=True));dump(O/'TESTS.json',dict(exit_code=0,tests=tests,valid_per_leg=valid.sum(axis=0).tolist(),invalid_per_leg=(~valid).sum(axis=0).tolist()));dump(O/'INPUT_CODE_HASHES.json',h)
 fig,axs=plt.subplots(3,2,figsize=(15,10),sharex=True,sharey=True)
 for j,(ax,leg) in enumerate(zip(axs.flat,LEGS)):
  ax.plot(frame,np.rad2deg(theta[:,j]),lw=1);ax.set_title(leg+' : Tro - FeTi - TiTa');ax.set_ylabel('Unsigned interior angle / deg');ax.set_xlabel('Original source frame');ax.grid(alpha=.25)
 fig.suptitle('Real animal reconstructed points / frame-domain geometry ONLY / behavior unverified');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(O/'SIXLEG_ANGLES.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S4_REAL_SIXLEG_ANGLES.png').write_bytes((O/'SIXLEG_ANGLES.png').read_bytes());assert all(sha(Path(f))==v for f,v in h.items());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,model_load=0,forward=0,body_step=0,neural_step=0,failures=[]));print('6 source-named topology-backed angles verified;4 deterministic checks pass')
if __name__=='__main__':main()
