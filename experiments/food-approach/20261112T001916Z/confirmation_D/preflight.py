"""Parent-lateractual zeroCore intervention equations/strictload/sourceadaptationchecks."""
from contracts import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args();require(a.parent_resource_released,'parentrelease');verify()
 from runtime import namespace,source,actor_input
 import numpy as np
 g=namespace();g['cpu1']();motor,behavior,mp,bp=g['load_policies']();checks={}
 def check(k,v):checks[k]=bool(v)
 check('samepolicy256plus32',motor.identity()['state_dims']==[128,64,64] and behavior.identity()['recurrent_state']==32)
 for script in ('runner.py','audit.py'):compile(source(script),'D::'+script,'exec')
 check('exactsourceadaptations_compile',True);z=np.array([.1,.4]);check('cut_before3_unmodified',np.array_equal(actor_input('cut_L20N',2.995,z),z));check('cut_at3_zero',np.array_equal(actor_input('cut_L20N',3.,z),np.zeros(2)));check('swap_onlyactor',np.array_equal(actor_input('swap_L20N',3.,z),z[::-1]) and np.array_equal(z,np.array([.1,.4])));check('normal_originalinput',np.array_equal(actor_input('normal_L20N',3.,z),z))
 check('fixed27rows',len(trials())==27 and sum(r['control']=='noodor' for r in trials())==1)
 check('fixed2donors_not_posthoc',[(r['donor'],r['forward'],r['angle']) for r in trials() if r['control']=='replay']==[('normal_L20N',.55,-.20),('normal_R30F',.625,.30)])
 require(all(checks.values()),str(checks));verify()
 with (ROOT/'PREFLIGHT.json').open('x') as f:json.dump(dict(all_passed=True,checks=checks,Core_constructions=0,physics_steps=0,sourcefreeze_sha256=sha(ROOT/'FROZEN.json')),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0)))
if __name__=='__main__':main()
