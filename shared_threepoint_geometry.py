"""Source-tagged geometric interface: unresolved definitions remain NaN."""
from pathlib import Path
import json,hashlib,sys,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from real_reference_angles import angles,LEGS
P=Path(__file__).parent
X=P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
A=Path('/home/sensen/Desktop/engine修复验收/s4-real-geometric-angles/20261027T215000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal/BODY.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s4-shared-threepoint-geometry/20261027T220000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read_geometry(points,axis,definition_valid):
 """points shape(N,6,3,3), axis length N. No time conversion or reconstruction."""
 points=np.asarray(points,dtype=float);assert points.shape==(len(axis),6,3,3);theta,valid=angles(points);valid&=np.asarray(definition_valid,dtype=bool)[None,:];theta[~valid]=np.nan
 return dict(axis=np.asarray(axis),points=points,angle_rad=theta,valid=valid)
def connection_points(model,data,leg):
 """Model-only anchors: Tro composite anchor, FeTi hinge, TiTa welded frame.

 TiTa is the Tarsus1 child frame origin ONLY after source WeldJoint validation.
 This rule does not establish correspondence to animal observed landmarks.
 """
 import mujoco
 jid=lambda n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
 bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,leg+'Tarsus1')
 assert bid>=0
 return np.array([data.xanchor[jid('joint_'+leg+'Trochanter_pitch')],data.xanchor[jid('joint_'+leg+'Tibia_pitch')],data.xpos[bid]])
def main():
 O.mkdir(parents=True,exist_ok=False);text=X.read_text();root=ET.fromstring(text);parents={c:p for p in root.iter() for c in p};bodies={b.get('name'):b for b in root.iter('body')};mapping=[]
 def ref(e):
  token='<'+e.tag+' name="'+e.get('name')+'"';return dict(path=str(X),line=next(i for i,l in enumerate(text.splitlines(),1) if token in l),attributes=dict(e.attrib),parent=parents[e].get('name'))
 for animal,model in zip(LEGS,['LF','LM','LH','RF','RM','RH']):
  tro=next(j for j in root.iter('joint') if j.get('name')=='joint_'+model+'Trochanter_pitch');tib=next(j for j in root.iter('joint') if j.get('name')=='joint_'+model+'Tibia_pitch');tar=bodies[model+'Tarsus1'];assert not list(tar.findall('joint'))
  mapping.append(dict(animal_leg=animal,model_leg=model,Tro=dict(candidate=ref(tro),status='KINEMATIC_ANCHOR_CANDIDATE_NOT_LANDMARK_CALIBRATED',note='front legs have distinct Trochanter and fixed Femur body origins; do not substitute Femur origin'),FeTi=dict(candidate=ref(tib),status='KINEMATIC_ANCHOR_CANDIDATE_NOT_LANDMARK_CALIBRATED'),TiTa=dict(candidate=ref(tar),status='UNRESOLVED',note='fixed child-body attachment; no joint/site explicitly identifies anatomical TiTa center; do not infer from name'),triplet_status='UNRESOLVED',angle_available=False))
 dump(O/'MODEL_POINT_MAPPING.json',mapping);src=np.load(A/'ANGLES.npz');raw=src['raw_original_array'].reshape(-1,50,3);pts=raw[:,src['triplet_indices_in_original_names'],:];animal=read_geometry(pts,src['source_frame'],[True]*6);np.testing.assert_array_equal(animal['valid'],src['valid']);np.testing.assert_allclose(animal['angle_rad'],src['angle_rad'],atol=0,rtol=0)
 with np.load(B) as saved:time=saved['time'];mask=time<=.040000000001;t=time[mask].copy();qpos=saved['qpos'][mask].copy();qvel=saved['qvel'][mask].copy()
 assert t[0]==0 and len(t)>1;model=read_geometry(np.full((len(t),6,3,3),np.nan),t,[False]*6);assert not model['valid'].any() and np.isnan(model['angle_rad']).all()
 np.savez_compressed(O/'ANIMAL_GEOMETRY.npz',**animal);np.savez_compressed(O/'MODEL_GEOMETRY.npz',**model,saved_qpos_unmodified=qpos,saved_qvel_unmodified=qvel)
 h={str(p):sha(p) for p in [X,B,A/'ANGLES.npz',A/'TRIPLETS_AND_SCOPE.json',P/'real_reference_angles.py',Path(__file__)]};assert h[str(X)]=='65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6';dump(O/'INPUT_CODE_HASHES.json',h)
 R=np.array([[0,-1.,0],[1,0,0],[0,0,1.]])
 for pp in [pts.astype(float)+[3,4,5],pts.astype(float)@R.T,pts.astype(float)*2.3]:
  rr=read_geometry(pp,animal['axis'],[True]*6);np.testing.assert_allclose(rr['angle_rad'],animal['angle_rad'],atol=1e-13)
 rev=read_geometry(pts[:,:,::-1,:],animal['axis'],[True]*6);np.testing.assert_allclose(rev['angle_rad'],animal['angle_rad'],atol=1e-13);synthetic=np.zeros((1,6,3,3));bad=read_geometry(synthetic,[0],[True]*6);assert not bad['valid'].any() and np.isnan(bad['angle_rad']).all()
 dump(O/'VERIFICATION.json',dict(exit_code=0,animal_source_angle_exact=True,translation_rotation_positive_scale_invariant=True,endpoint_order_reversal_invariant=True,vertex='FeTi remains middle, not permuted',zero_segment_mask=True,model_unresolved_mask=True,model_samples=len(t),model_time_range_s=[float(t[0]),float(t[-1])],restoration_status='NOT_EXECUTED_UNRESOLVED_POINT_DEFINITION',world_position_restoration_error=None,qpos_is_not_geometric_angle=True,models_loaded=0,forward=0,step=0,neural=0))
 fig,axs=plt.subplots(6,3,figsize=(16,19))
 for j,leg in enumerate(LEGS):
  p=pts[0,j].astype(float);p-=p[1];axs[j,0].plot(p[:,0],p[:,2],'-o');
  for k,n in enumerate(['Tro','FeTi(vertex)','TiTa']):axs[j,0].annotate(n,(p[k,0],p[k,2]))
  axs[j,0].set_title(leg+' source-frame6555 x/z projection');axs[j,0].set_aspect('equal',adjustable='datalim');axs[j,1].plot(animal['axis'],np.rad2deg(animal['angle_rad'][:,j]));axs[j,1].set_title(leg+' animal / original frame');axs[j,1].set_ylabel('Unsigned interior angle / deg');axs[j,2].set_xlim(0,.04);axs[j,2].text(.5,.5,'MODEL '+mapping[j]['model_leg']+'\nTiTa center UNRESOLVED\nNo world points or angle fabricated',ha='center',transform=axs[j,2].transAxes);axs[j,2].set_title('Historical0-40ms / angle unavailable');axs[j,2].set_xlabel('Saved simulation time / s')
 fig.suptitle('Shared three-point definition audit / source projection is NOT model anatomy / no behavioral comparison');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'DEFINITION_AUDIT.png',dpi=110);plt.close(fig);Path('/home/sensen/Desktop/S4_SHARED_GEOMETRY.png').write_bytes((O/'DEFINITION_AUDIT.png').read_bytes());assert all(sha(Path(p))==v for p,v in h.items());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,failures=[],model_load=0,forward=0,body_step=0,neural_step=0));print('shared interface exact; model6 triplets unresolved, no model load; saved samples',len(t))
if __name__=='__main__':main()
