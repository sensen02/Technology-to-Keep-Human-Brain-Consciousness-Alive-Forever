"""Local OpenSim welded connection evidence and exactly three saved-state forwards."""
from shared_threepoint_geometry import X,B,P,sha,dump,connection_points
from pathlib import Path
import xml.etree.ElementTree as ET,numpy as np,json,sys
O=Path('/home/sensen/Desktop/engine修复验收/s4-tita-connection-source/20261027T221000Z')
U=P/'venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml'
S=P/'outputs/upstream_muscle_round11/best_combined.osim'
def main():
 O.mkdir(parents=True,exist_ok=False);files=[X,U,S,B,P/'shared_threepoint_geometry.py',P/'tools_mirror_muscles.py',P/'tools_stand_and_ground.py',Path(__file__)];h={str(p):sha(p) for p in files};rr=ET.parse(S).getroot();current=ET.parse(X).getroot();up=ET.parse(U).getroot();rows=[]
 def ref(p,tag,name):return dict(path=str(p),line=next(i for i,l in enumerate(p.read_text().splitlines(),1) if '<'+tag+' name="'+name+'"' in l),sha256=sha(p))
 for leg in ['LF','LM','LH','RF','RM','RH']:
  name=leg+'Tarsus1';j=next(j for j in rr.iter('WeldJoint') if j.get('name')=='joint_'+name);frames={f.get('name'):f for f in j.find('frames')};parent=frames[j.findtext('socket_parent_frame')];child=frames[j.findtext('socket_child_frame')];assert parent.findtext('socket_parent')=='/bodyset/'+leg+'Tibia' and child.findtext('socket_parent')=='/bodyset/'+name;pos=np.fromstring(parent.findtext('translation','0 0 0'),sep=' ');cp=np.fromstring(child.findtext('translation','0 0 0'),sep=' ');assert np.all(cp==0);body=next(b for b in current.iter('body') if b.get('name')==name);ub=next(b for b in up.iter('body') if b.get('name')==name);xmlpos=np.fromstring(body.get('pos'),sep=' ');assert body.get('pos')==ub.get('pos');delta=float(abs(xmlpos-pos).max());assert delta<5e-5
  rows.append(dict(leg=leg,source_weld=ref(S,'WeldJoint','joint_'+name),parent_frame=dict(socket=parent.findtext('socket_parent'),translation=pos.tolist(),orientation=parent.findtext('orientation','0 0 0')),child_frame=dict(socket=child.findtext('socket_parent'),translation=cp.tolist(),orientation=child.findtext('orientation','0 0 0'),missing_translation_semantics='OpenSim default zero'),installed_source=ref(U,'body',name),frozen=ref(X,'body',name),xml_local_position=xmlpos.tolist(),source_to_xml_rounding_max=delta,model_connection_status='SUPPORTED_WELDED_CONNECTION_FRAME',animal_landmark_status='NOT_ESTABLISHED'))
 extra=[]
 for leg in ['LF','RF']:
  j=next(j for j in rr.iter('WeldJoint') if j.get('name')=='joint_'+leg+'Femur');f=list(j.find('frames'))[0];b=next(b for b in current.iter('body') if b.get('name')==leg+'Femur');a=np.fromstring(f.findtext('translation'),sep=' ');c=np.fromstring(b.get('pos'),sep=' ');assert abs(a-c).max()<5e-5;extra.append(dict(leg=leg,source=ref(S,'WeldJoint','joint_'+leg+'Femur'),parent_frame_translation=a.tolist(),xml_body=ref(X,'body',leg+'Femur'),xml_position=c.tolist(),meaning='nonzero fixed Trochanter->Femur offset, not interchangeable origins'))
 dump(O/'CONNECTION_EVIDENCE.json',dict(tita=rows,front_femur_offsets=extra,conversion_commit='NOT_ESTABLISHED; local OpenSim numerical/structural agreement is not proof of exact converter invocation',no_turaga_asset_used=True))
 from engine.canonical_physics import load_model
 import mujoco
 m=load_model();d=mujoco.MjData(m);saved=np.load(B);times=saved['time'];q=saved['qpos'];vel=saved['qvel'];indices=[int(np.argmin(abs(times-t))) for t in [0,.010,.025]];positions=[];errs=[]
 for i,target in zip(indices,[0,.010,.025]):
  assert abs(times[i]-target)<1e-10;d.qpos[:]=q[i];d.qvel[:]=vel[i];d.time=float(times[i]);mujoco.mj_forward(m,d);pts=[]
  for leg,row in zip(['LF','LM','LH','RF','RM','RH'],rows):
   pp=connection_points(m,d,leg);pts.append(pp);tib=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,leg+'Tibia');tar=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,leg+'Tarsus1');expected=d.xpos[tib]+d.xmat[tib].reshape(3,3)@np.array(row['xml_local_position']);errs.append(float(abs(expected-d.xpos[tar]).max()))
  positions.append(pts);np.testing.assert_array_equal(d.qpos,q[i]);np.testing.assert_array_equal(d.qvel,vel[i])
 np.savez_compressed(O/'THREE_SAVED_STATE_CONNECTIONS.npz',time=times[indices],qpos=q[indices],qvel=vel[indices],model_connection_points=np.array(positions));assert max(errs)<1e-12;dump(O/'VERIFICATION.json',dict(exit_code=0,models_loaded=1,mj_forward=3,mj_step=0,neural_step=0,state_times_s=times[indices].tolist(),qpos_qvel_unchanged=True,independent_weld_parent_child_transform_error_max=max(errs),model_connection_supported=True,animal_landmark_correspondence='NOT_ESTABLISHED',no_new_angle_curves=True));assert all(sha(Path(p))==v for p,v in h.items());dump(O/'INPUT_CODE_HASHES.json',h);(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'shared_threepoint_geometry.py').write_bytes((P/'shared_threepoint_geometry.py').read_bytes());print('6 welded connections supported;1load3forward0step;max residual',max(errs))
if __name__=='__main__':main()
