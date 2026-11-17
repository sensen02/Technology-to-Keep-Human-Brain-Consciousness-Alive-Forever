"""Parent-released loader/ABI check; no Core/physics integration. Default CODE_ONLY."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY: model preflight not executed');return
 from runtime import namespace,entry
 g=namespace();g['cpu1']();f=g['verify_frozen']();motor,behavior,mp,bp=g['load_policies']()
 checks=dict(offline_admission_bound=f['source_hashes'][str(ROOT/'INPUT_BINDINGS.json')]==g['sha'](ROOT/'INPUT_BINDINGS.json'),final500=bp['curve'][-1]['step']==500,same32=behavior.identity()['recurrent_state']==32,motor256=motor.identity()['state_dims']==[128,64,64],all_weights_frozen=all(not p.requires_grad for p in list(motor.parameters())+list(behavior.parameters())),no_physical_import='physics' not in sys.modules and 'mujoco' not in sys.modules)
 for script,name in [('runner.py','main'),('audit.py','verify'),('summarize_B.py','main')]:
  fn=entry(script,name);n=fn.__globals__
  checks[script+'_newROOT']=n['ROOT']==ROOT
  checks[script+'_newSENSOR']=n['SENSOR']==g['SENSOR'] and n['SENSOR_SHA']==g['SENSOR_SHA']
  checks[script+'_newloader']=n['load_policies'].__code__==g['load_policies'].__code__
  checks[script+'_inherited_concentration_newROOT']=n['concentration'].__globals__['ROOT']==ROOT
  checks[script+'_new_freeze']=n['verify_frozen'].__closure__ is None or n['verify_frozen'].__code__==g['verify_frozen'].__code__
 if not all(checks.values()):raise ValueError('B wrapper namespace/loader preflight failed '+str(checks))
 g['save_new'](ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,checkpoint_sha256=g['MOTOR_SHA'],behavior_checkpoint_sha256=g['SENSOR_SHA'],state_layout=[128,64,64,32],state_total=288,model_loads=2,forward_calls=0,Core_constructions=0,physics_steps=0,source_hashes=f['source_hashes']))
 print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
