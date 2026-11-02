"""Independent saved muscle parameter/formula review; no body or neural integration."""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s2-reference-muscle-operating-range/20261102T155412Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN'
O=ROOT/'s2-muscle-range-parent-review/20261102T155412Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'RESULT.json').exists()
 paths=[Path(__file__),S/'AUDIT.npz',S/'COMPILED_ACTUAL_ARRAYS.npz',S/'PREREGISTRATION.json',B/'BODY.npz',B/'MAPPING.json'];hashes={str(p):sha(p) for p in paths};dump(O/('BOUND_FLOOR_ADDENDUM.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(scope='Independent scalar nativeformula and measuredcachedstates review;0MuJoCo imports/compile/forward/steps',source_sha256=hashes,thresholds=dict(gain_abs=1e-12,relative_velocity_suppression=1e-3)))
 a=arrays(S/'AUDIT.npz');m=arrays(S/'COMPILED_ACTUAL_ARRAYS.npz');body=arrays(B/'BODY.npz');rows=json.loads((B/'MAPPING.json').read_text())['rows'];ids=a['source_indices'];assert len(ids)==401
 for key in ['time','qpos','qvel','activation','command','muscle_force','foot_force','assist','com']:assert np.array_equal(a[key],body[key][ids]),key
 assert not a['assist'][:,:6].any() and not a['external'].any() and not a['nonfoot_count'].any()
 maxerr=dict(gain=0.,stationary=0.,bias=0.,force=0.,fiber=0.,velocity=0.,torque=0.);counts=dict(lengthzero=0,velocityzero=0,velocitysuppressed=0,gainzero=0);perleg={leg:dict(lengthzero=0,velocityzero=0,gainzero=0,total=0) for leg in ['LF','LM','LH','RF','RM','RH']}
 for t in range(401):
  for u in range(90):
   p=m['actuator_gainprm'][u];r0,r1,force,scale,lmin,lmax,vmax,fpmax,fvmax=map(float,p[:9]);lr=m['actuator_lengthrange'][u];L0=(float(lr[1])-float(lr[0]))/max(1e-15,r1-r0);LT=float(lr[0])-r0*L0;L=(float(a['length'][t,u])-LT)/max(1e-15,L0);V=float(a['velocity'][t,u])/max(1e-15,L0*vmax);F=scale/max(1e-15,float(m['actuator_acc0'][u])) if force<0 else force
   left=(lmin+1)/2;right=(lmax+1)/2
   if L<lmin or L>lmax:fl=0.
   elif L<left:fl=.5*((L-lmin)/max(1e-15,left-lmin))**2
   elif L<1:fl=1-.5*((1-L)/max(1e-15,1-left))**2
   elif L<right:fl=1-.5*((L-1)/max(1e-15,right-1))**2
   else:fl=.5*((lmax-L)/max(1e-15,lmax-right))**2
   fv=0. if V<=-1 else (V+1)**2 if V<=0 else fvmax-((fvmax-1)-V)**2/max(1e-15,fvmax-1) if V<fvmax-1 else fvmax
   bias=0. if L<=1 else -F*fpmax*.5*((L-1)/max(1e-15,right-1))**2 if L<=right else -F*fpmax*(.5+(L-right)/max(1e-15,right-1));gain=-F*fl*fv;stationary=-F*fl
   for k,error in [('gain',abs(gain-a['gain'][t,u])),('stationary',abs(stationary-a['gain_zero_velocity'][t,u])),('bias',abs(bias-a['bias'][t,u])),('force',abs(gain*a['activation'][t,u]+bias-a['muscle_force'][t,u])),('fiber',abs(L-a['normalized_fiber_length'][t,u])),('velocity',abs(V-a['normalized_fiber_velocity'][t,u]))]:maxerr[k]=max(maxerr[k],float(error))
   lengthzero=abs(stationary)<=1e-12;velocityzero=abs(stationary)>1e-12 and abs(gain)<=1e-12;suppressed=abs(stationary)>1e-12 and abs(gain)<=1e-3*abs(stationary);gainzero=abs(gain)<=1e-12
   assert lengthzero==bool(a['force_length_zeroed'][t,u]) and velocityzero==bool(a['force_velocity_zeroed'][t,u]) and suppressed==bool(a['force_velocity_suppressed'][t,u]) and gainzero==bool(a['native_gain_nearzero'][t,u])
   for key,val in [('lengthzero',lengthzero),('velocityzero',velocityzero),('velocitysuppressed',suppressed),('gainzero',gainzero)]:counts[key]+=int(val)
   leg=rows[u]['leg'];perleg[leg]['total']+=1
   for key,val in [('lengthzero',lengthzero),('velocityzero',velocityzero),('gainzero',gainzero)]:perleg[leg][key]+=int(val)
  for j in range(48):
   predicted=math.fsum(float(a['moment'][t,u,j])*float(a['muscle_force'][t,u]) for u in range(90));maxerr['torque']=max(maxerr['torque'],abs(predicted-float(a['qfrc_actuator'][t,j])))
 assert max(maxerr.values())<1e-9
 # Coordinatewise independentactivation relaxation, not static supportequilibrium.
 outside=0;maxbound=0.
 for t in range(401):
  for j in range(42):
   co=a['coefficient_42x90'][t,j];passive=math.fsum(float(a['moment_42x90'][t,j,u])*float(a['bias'][t,u]) for u in range(90));floor=passive+math.fsum(map(float,co))*1e-4;lo=floor+.9999*math.fsum(min(float(v),0.) for v in co);hi=floor+.9999*math.fsum(max(float(v),0.) for v in co);maxbound=max(maxbound,abs(lo-a['independent_lower'][t,j]),abs(hi-a['independent_upper'][t,j]));x=float(a['historical_total_input'][t,j]);outside+=int(x<lo-1e-12 or x>hi+1e-12)
 assert maxbound<1e-10
 report=dict(passed=True,source_sha256=hashes,samples=401,actuators=90,maxerrors=maxerr,independent_box_max_error=maxbound,zero_counts=counts,perleg=perleg,old_body_states_exact=True,historical_total_input_outside_independent_box=outside,missing_mapped_actuators=int(sum(not r['local_indices'] for r in rows)),calls=dict(model_compile=0,forward=0,body_step=0,network=0,neural_step=0,training=0),limits='Countsaredeclaredengineeringthresholds. NativeL/V/Fparameters are modelcoordinates,not measuredfiberlength/physiology. Historicaltotalinput isalgorithm+muscle,not necessarysupporttorque. Independentcoordinateboxes neednotbesimultaneouslyachievable; no globalstandimpossibilityproof.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
