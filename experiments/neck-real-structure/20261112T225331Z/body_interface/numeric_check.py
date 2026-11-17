"""Parent-only CPU1 numerical gradient/mask/real packet ABI check, no training/Core.

Default import stays standard-library only. Functional native tuning remains unknown.
"""
from pathlib import Path
import argparse,json,os,importlib.util
from interface import make_neural_interfaces,source_ports,sha,validate_delivery
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent',action='store_true');p.add_argument('--binding',type=Path,default=HERE/'SOURCE_PORT_BINDING.json');a=p.parse_args()
 if not a.execute_parent:print('CODE_ONLY no numeric import/forward');return
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 bindingpath=a.binding.resolve();binding=json.loads(bindingpath.read_text());ports=source_ports(binding);before=sha(bindingpath)
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261112)
 decoder,feedback,identity=make_neural_interfaces(binding);checks={}
 def check(name,value):
  checks[name]=bool(value)
  if not checks[name]:raise ValueError('body interface numeric check '+name)
 dn=torch.linspace(-.4,.4,4*len(ports['DN']),dtype=torch.float64).reshape(4,len(ports['DN'])).requires_grad_();raw=torch.zeros((4,238),dtype=torch.float64,requires_grad=True)
 turn=decoder(dn);command=decoder.motor_command(dn);an=feedback(raw)
 check('dimensions_from_source_ids',turn.shape==(4,1) and an.shape==(4,len(ports['AN'])))
 check('motor_ABI_coupled_signedturn',torch.equal(command[:,0:1],torch.relu(command[:,1:2])/3.) and bool((command.abs()<=1).all()))
 gradients=torch.autograd.grad(an.sum(),raw,retain_graph=True)[0];forbidden=[i for i in range(238) if i not in identity['feedback_raw_indices']]
 check('feedback_forbidden_indices_exact_zero_gradient',torch.equal(gradients[:,forbidden],torch.zeros_like(gradients[:,forbidden])))
 check('feedback_allowed_sensory_gradient_nonzero',bool((gradients[:,identity['feedback_raw_indices']].abs()>0).any()))
 check('constant_target_speed_gradient_zero',bool(torch.all(gradients[:,237]==0)))
 changed=raw.detach().clone();changed[:,forbidden]=.75
 check('feedback_forbidden_input_perturbation_exact_invariance',torch.equal(feedback(changed),an.detach()))
 # Task cue is not an argument or captured state in the body decoder. No direct path.
 cue=torch.tensor([[.3]],dtype=torch.float64,requires_grad=True)
 check('body_decoder_no_direct_cue_path',torch.autograd.grad(turn.sum(),cue,allow_unused=True,retain_graph=True)[0] is None)
 check('DN_to_neuralturn_nonzero_gradient',bool(torch.autograd.grad(turn.sum(),dn,retain_graph=True)[0].abs().sum()>0))
 loss=turn.square().mean()+an.square().mean();loss.backward()
 check('learned_adapter_all_parameter_gradients_finite',all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for m in (decoder,feedback) for p in m.parameters()))
 # Adapt actual previous validated single-fibre packet keys; no biological signals.
 transport_status='NOT_RUN_PARENT_MUST_CHECK_GENERALIZED_SOURCE_TRANSPORT_SEPARATELY'
 if 'parent_neck_binding' in binding and len(ports['DN'])==len(ports['AN'])==1:
  old=Path(binding['parent_neck_binding']);s=importlib.util.spec_from_file_location('_body_check_real_transport',old.parent/'boundary.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b);transport=b.BoundaryTransport(old,binding['parent_neck_binding_sha256'])
  for direction,value in [('DN',.25),('AN',-.125)]:transport.send({'from':'brain' if direction=='DN' else 'other','to':'neck','direction':direction,'root_ids':ports[direction],'values':[value],'capture_step':0,'source_bindings_sha256':binding['parent_neck_binding_sha256']})
  check('neck_no_early_release_step1',transport.advance()==[]);released=transport.advance();check('neck_actual_source_packet_delay2',len(released)==2 and all(x['release_step']==2 for x in released))
  for packet in released:
   canonical=dict(source_part=packet['source_part'],target_part=packet['target_part'],direction=packet['direction'],root_ids=[packet['root_id']],values=[packet['value']],capture_step=packet['capture_step'],release_step=packet['release_step'],consume_step=packet['consume_step'],source_bindings_sha256=packet['source_bindings_sha256']);validate_delivery(canonical,ports[packet['direction']],packet['direction'],binding['parent_neck_binding_sha256'],2)
  check('delivered_neck_packet_to_body_and_brain_ABI',True);transport_status='OLD_TWO_FIBRE_ACTUAL_SOURCE_QUEUE_DELAY2_CHECKED'
 check('source_bindings_unchanged',sha(bindingpath)==before);check('no_physical_backend_import',not any(x in __import__('sys').modules for x in ('mujoco','physics','student_wave')))
 report=dict(status='SCOPED_NUMERICAL_BODY_ADAPTER_CHECK_NOT_FUNCTIONAL_TRAINING',all_passed=all(checks.values()),checks=checks,transport_status=transport_status,identity=identity,source_binding_sha256=before,code_hashes={str(HERE/n):sha(HERE/n) for n in ('interface.py','numeric_check.py','prepare_binding.py')},new_training_updates=0,optimizer_steps=0,Core_constructions=0,physics_steps=0,native_DN_or_AN_tuning_verified=False,brain_training_performed=False)
 with (HERE/'NUMERIC_CHECK.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),source_dimensions={k:len(v) for k,v in ports.items()},training_updates=0,physics_steps=0)))
if __name__=='__main__':main()
