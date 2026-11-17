"""Parent only actual336neural/source/filter ABI check, no Core import/construction."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no numerical execution');return
 cpu1();verify();sentinel=object();prior={k:sys.modules.get(k,sentinel) for k in ('model','common','steering_behavior')};visual,odor,bridge,motor,vp,op,bp,mp=load_policies()
 import torch,numpy as np
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Sensorypreflight '+k)
 check('allaliases_restored',all(sys.modules.get(k,sentinel) is v for k,v in prior.items()));check('actualnewfinal400',bp['step']==400 and bp['format_version']==FORMAT);check('336stateABI',flat(motor.initial_state(1),visual.initial_state(1),odor.initial_state(1),torch.zeros((1,16),dtype=torch.float64)).shape==(1,336));check('allfrozenweights',all(not p.requires_grad for m in (visual,odor,bridge,motor) for p in m.parameters()));check('braininputANvisualodor3',bridge.brain.input_size==3 and bridge.brain.hidden_size==16)
 field=OdorField('O_left');hp=np.zeros(3);hr=np.eye(3);field.anchor(hp,hr);ss0=field.sample(hp,hr,6600);ss1=field.sample(hp,hr,6605);check('realproxy_sourcegeom',ss0['sensor_xyz_now'].shape==(2,3) and np.linalg.norm(field.center-ss0['sensor_xyz_now'].mean(0)[:2])>0);check('previouspending5ms',np.array_equal(ss0['actor_receptors'],np.zeros(2)) and np.array_equal(ss1['actor_receptors'],ss0['pending_after']));check('C_receptor_transduction',(ss0['concentration']>=0).all() and (ss0['filter_after']>=0).all() and (ss0['filter_after']<=1).all())
 with np.load(NEUTRAL/'neural_decisions.npz',allow_pickle=False) as z:x=z['X238'][0].copy()
 loop=BrainNeckLoop(bridge,'V_empty');z=torch.zeros(1,dtype=torch.float64);first=loop.decision(x,z,z,0)
 for _ in range(5):loop.advance()
 second=loop.decision(x,z,z,5);check('warmbrainDNcommand0',np.max(abs(second['brain_hidden_after']))==np.max(abs(second['DN_encoded']))==np.max(abs(second['command']))==0);check('AN2msrelease5msconsume',second['AN_capture_step']==0 and second['AN_release_step']==2 and np.array_equal(second['AN_held'],first['AN_encoded']))
 with torch.no_grad():
  odcmd,oh=odor.step(torch.zeros((1,2),dtype=torch.float64),odor.initial_state(1));dn,h=bridge.brain_step(torch.zeros((1,1),dtype=torch.float64),torch.tensor([[.2,float(odcmd[0,1])]],dtype=torch.float64),torch.zeros((1,16),dtype=torch.float64));turn=bridge.decoder(dn);command=project(turn[:,0]);mean,mh=motor.step(torch.tensor(x[None],dtype=torch.float64),command,motor.initial_state(1))
 check('naturalzero_receptors_odor_bias_recorded',bool(torch.isfinite(odcmd).all()));check('learned_fusion_and_motor_dimensions',dn.shape==(1,2) and h.shape==(1,16) and mean.shape==(1,42));check('source_before_after_unchanged',loop.transport.verify_final()['source_files_unchanged']);check('noCore_or_physical_import',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')));verify();save(ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,Core_constructions=0,new_physics_steps=0,state_total=336,natural_noodor_first_rawturn=float(odcmd[0,1]),zero_bias_not_gated=True,sourcefreeze_sha256=sha(ROOT/'FROZEN.json')));print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,new_physics_steps=0)))
if __name__=='__main__':main()
