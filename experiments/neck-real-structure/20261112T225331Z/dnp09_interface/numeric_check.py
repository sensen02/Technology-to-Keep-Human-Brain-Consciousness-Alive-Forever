"""Parent-only source N=2/M=1 lazy neural and source-vector transport preflight.

No optimizer, training, motor weights or Core. Dynamic/tuning semantics engineering only.
"""
from pathlib import Path
import argparse,json,os,sys
from api import sha,require,source_ports,make_neural_interfaces,validate_delivery
from source_transport import SourceTransport
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent',action='store_true');args=p.parse_args()
 if not args.execute_parent:print('CODE_ONLY no numerical construction');return
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key)=='1',key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});bp=HERE/'SOURCE_BINDING.json';digest=sha(bp);binding=json.loads(bp.read_text());ports=source_ports(binding);require(len(ports['DN'])==2 and len(ports['AN'])==1,'actual scoped N2M1 source binding required')
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261112);decoder,encoder,identity=make_neural_interfaces(binding);checks={}
 def check(name,v):checks[name]=bool(v);require(checks[name],'DNP09 preflight '+name)
 dn=torch.linspace(-.6,.6,8,dtype=torch.float64).reshape(4,2).requires_grad_();raw=torch.zeros((4,238),dtype=torch.float64,requires_grad=True);turn=decoder(dn);an=encoder(raw);cmd=decoder.motor_command(dn)
 check('source_derived_neural_shapes',turn.shape==(4,1) and an.shape==(4,1));check('frozen_motor_ABI',torch.equal(cmd[:,0:1],torch.relu(cmd[:,1:2])/3.) and bool((cmd.abs()<=1).all()))
 grad=torch.autograd.grad(an.sum(),raw,retain_graph=True)[0];allowed=identity['feedback_raw_indices'];forbidden=[i for i in range(238) if i not in allowed]
 check('111mask_excludes237_and_hidden_actions',len(allowed)==111 and 237 not in allowed and bool(torch.all(grad[:,forbidden]==0)))
 check('body_allowed_sensory_path_nonzero',bool((grad[:,allowed].abs()>0).any()))
 changed=raw.detach().clone();changed[:,forbidden]=.8;check('forbidden_inputs_exact_invariance',torch.equal(an.detach(),encoder(changed)))
 cue=torch.ones((4,1),dtype=torch.float64,requires_grad=True);check('no_taskcue_direct_body_path',torch.autograd.grad(turn.sum(),cue,allow_unused=True,retain_graph=True)[0] is None)
 dg=torch.autograd.grad(turn.sum(),dn,retain_graph=True)[0];check('both_source_DNs_have_neural_gradient',bool((dg.abs().sum(0)>0).all()))
 (turn.square().mean()+an.square().mean()).backward();check('all_learned_parameter_gradients_finite',all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for m in (decoder,encoder) for p in m.parameters()))
 engine=SourceTransport(bp,digest)
 def packet(d,values):return {'from':'brain' if d=='DN' else 'other','to':'neck','direction':d,'root_ids':ports[d],'values':values,'capture_step':engine.step,'source_bindings_sha256':digest}
 engine.send(packet('DN',[.25,-.5]));engine.send(packet('AN',[.125]));check('no_release_at_tick1',engine.advance()==[]);out=engine.advance();check('vector_source_order_and_values',out[0]['root_ids']==ports['DN'] and out[0]['values']==[.25,-.5] and out[1]['root_ids']==ports['AN'])
 check('release2ms_not_consume5ms',all(p['capture_step']==0 and p['release_step']==p['consume_step']==2 for p in out))
 while engine.step<5:check('no_duplicate_release_step'+str(engine.step),engine.advance()==[])
 for event in out:
  decision_packet=dict(event,consume_step=5);validate_delivery(decision_packet,ports[event['direction']],event['direction'],digest,5)
 check('first5msdecision_consumes_already_released_frame',True)
 engine.send(packet('DN',[.4,-.2]));engine.send(packet('AN',[.3]));engine.set_cut('DN',True);engine.advance();cut=engine.advance();check('cutDN_discards_inflight_AN_continues',len(cut)==1 and cut[0]['direction']=='AN');prior=engine.step;engine.set_cut('DN',False);check('restore_no_clock_reset',engine.step==prior);engine.send(packet('DN',[-.1,.2]));engine.advance();restored=engine.advance();check('restore_future_vector_only',len(restored)==1 and restored[0]['values']==[-.1,.2] and restored[0]['capture_step']==prior)
 final=engine.verify_final();check('all_sourcefiles_before_after_unchanged',final['source_files_unchanged'] and sha(bp)==digest);check('no_physical_or_motor_backend',not any(k in sys.modules for k in ('mujoco','physics','student_wave')))
 journal=engine.journal()
 with (HERE/'TRANSPORT_JOURNAL.json').open('x') as f:json.dump(journal,f,indent=2);f.write('\n')
 result=dict(status='SCOPED_DNP09_NEURAL_INTERFACE_PREFLIGHT_NOT_FUNCTIONAL_CALIBRATION',all_passed=all(checks.values()),checks=checks,source_binding_sha256=digest,identity=identity,transport_final=final,actual_release_ms=2,first_decision_consumption_ms=5,code_hashes={str(HERE/n):sha(HERE/n) for n in ('api.py','build_binding.py','source_transport.py','numeric_check.py')},immutable_body_interface_sha256=sha(HERE.parent/'body_interface/interface.py'),journal_sha256=sha(HERE/'TRANSPORT_JOURNAL.json'),training_updates=0,optimizer_steps=0,Core_constructions=0,physical_steps=0,native_motor_AN_tuning_verified=False,full_neck_admitted=False)
 with (HERE/'NUMERIC_CHECK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),DN=2,AN=1,training_updates=0,physics_steps=0)))
if __name__=='__main__':main()
