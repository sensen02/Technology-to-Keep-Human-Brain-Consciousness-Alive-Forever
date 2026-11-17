"""Parent-only actual zeroCore source/causality/endpoint-duration checks beforeCphysics."""
from contracts import *
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args();require(a.parent_resource_released,'resourceparentrelease required');verify()
 from runtime import namespace,source
 from spatial import anchor_from_head,concentration_at,arrival_metrics
 import numpy as np
 g=namespace();g['cpu1']();motor,behavior,mp,bp=g['load_policies']();checks={}
 def check(k,v):checks[k]=bool(v)
 check('same256motor32behavior',motor.identity()['state_dims']==[128,64,64] and behavior.identity()['recurrent_state']==32)
 for script in ('runner.py','audit.py'):compile(source(script),script+'::actualC','exec')
 check('exactphysicaladaptationscompile',True)
 local=np.asarray(g['read'](g['GEOMETRY'])['sensor_offsets']['local_xyz']);head=np.array([.8,.1,1.4]);rot=np.eye(3);cfg=read(ROOT/'PREREG_PROPOSAL.json')['layouts'][0];c=anchor_from_head(head,rot,local,cfg);sites=head+local@rot.T
 check('fixedsourceforwardlateral',np.max(np.abs(c-(sites.mean(0)[:2]+np.array([cfg['forward'],cfg['forward']*np.tan(cfg['angle'])]))))<1e-12)
 check('sourceoffbeforeonset',np.array_equal(concentration_at(cfg['id'],.6,sites,c),np.zeros(2)))
 check('leftsourceactualleftCgreater',concentration_at(cfg['id'],1.,sites,c)[0]>concentration_at(cfg['id'],1.,sites,c)[1])
 check('noodorzeroC',np.array_equal(concentration_at('C_noodor',1.,sites,c),np.zeros(2)))
 s=g['ClampFilter']();out0=s.advance(concentration_at(cfg['id'],1.,sites,c));out1=s.advance(concentration_at(cfg['id'],1.005,sites,c));check('pendingcausalCdelay',np.array_equal(out0['receptors'],np.zeros(2)) and np.array_equal(out1['receptors'],out0['pending_after']))
 t=np.arange(201)*.001+6.4;yes=np.ones(201,dtype=bool);check('200endpointspan_not_arrived',not arrival_metrics(t[:200],yes[:200],t[:200]-5.8)['arrived_endpoint_proxy']);check('201endpointspan_arrived',arrival_metrics(t,yes,t-5.8)['arrived_endpoint_proxy'])
 require(all(checks.values()),str(checks));verify();g['save_new'](ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,Core_constructions=0,physics_steps=0,optimizer_updates=0,sourcefreeze_sha256=sha(ROOT/'FROZEN.json'),note='Spatialgeometry/arrivalcontracts and strictcomponents loaded, no claim ofCarrivalbeforephysical'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
