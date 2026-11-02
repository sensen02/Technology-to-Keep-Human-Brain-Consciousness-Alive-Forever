"""Independent XML/array review only: no physics/neural integrations or optimization."""
from pathlib import Path
import json,csv,hashlib,math,xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
M=ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml'
T=ROOT/'s4-real-motion-retarget/20261102T011240Z'
D=ROOT/'s4-middle-leg-retarget-diagnosis/20261102T020329Z'
C=ROOT/'s4-sourced-nt-gain-calibration/20261102T020329Z'
N=ROOT/'s4-sourced-nt-dynamic-replay/20261102T011240Z'
G=ROOT/'s3-sourced-nt-candidate-graph/20261102T004326Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
O=ROOT/'s4-geometry-nt-calibration-review/20261102T020329Z'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rotation(axis,t):
 a=axis/np.linalg.norm(axis);x,y,z=a;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
 return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
def unpack(p):return np.unpackbits(np.load(p)['spikes_packed'],axis=1,count=12000).astype(bool)
def main():
 O.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),M,T/'RETARGETED_TARGETS.npz',D/'DIAGNOSTIC_CANDIDATES.npz',D/'CASES.json',D/'FIRST_SEGMENT_REACHABILITY_CERTIFICATE.json',D/'KKT_ACTIVE_BOUNDS.csv',D/'MULTISTART.csv',C/'PREREGISTRATION.json',C/'SELECTION_TABLE.json',C/'RESULT.json',C/'INPUTS.npz',C/'BASELINE150_REUSED.npz',C/'MAPPING.json',B/'ACTUAL_We.npz',B/'ACTUAL_Wi.npz',G/'ADDED_We.npz']
 names=['GAIN_0p03125','GAIN_0p0625','GAIN_0p125','GAIN_0p25','GAIN_0p5','SELECTED_CURRENT_CUT','VALIDATE150']
 files += [C/f'{name}_{kind}.npz' for name in names for kind in ['NEURAL','DECODE','ACTUAL_We','ACTUAL_Wi']]
 hashes={str(p):sha(p) for p in files};dump(O/'PREREGISTRATION.json',dict(scope='Post-run independent pure arrays/XML review; no refit/reselection/rollout',source_sha256=hashes,checks='Rodrigues FK and analytic proximal necessary bound, native multistart/KKT summary, gain grid and actual CSR, every decoder and selectedcut/150ms prefix.'))
 tree=ET.parse(M);bodies={b.get('name'):b for b in tree.iter('body')};joints={j.get('name'):j for j in tree.iter('joint') if j.get('name')}
 target=np.load(T/'RETARGETED_TARGETS.npz');cases=np.load(D/'DIAGNOSTIC_CANDIDATES.npz');caseinfo=json.loads((D/'CASES.json').read_text());cert=json.loads((D/'FIRST_SEGMENT_REACHABILITY_CERTIFICATE.json').read_text());fkerror=0.;bounderror=0.
 for i,li in enumerate(cases['leg_indices']):
  leg=str(target['leg_names'][li]);v=np.fromstring(bodies[leg+'Femur'].get('pos'),sep=' ');v/=np.linalg.norm(v);rot=np.eye(3)
  for k,suffix in enumerate(['yaw','pitch','roll']):rot=rot@rotation(np.fromstring(joints['joint_'+leg+'Coxa_'+suffix].get('axis'),sep=' '),cases['qpos'][i,target['joint_qpos_addresses'][li,k]])
  actual=np.diff(cases['achieved_points'][i],axis=0)[0];actual/=np.linalg.norm(actual);fkerror=max(fkerror,float(abs(rot@v-actual).max()))
 assert fkerror<1e-12
 for row in cert:
  li=list(target['leg_names']).index(row['leg']);leg=row['leg'];v=np.fromstring(bodies[leg+'Femur'].get('pos'),sep=' ');v/=np.linalg.norm(v);eps=math.acos(np.clip(-v[2],-1,1))
  yaw=np.fromstring(joints['joint_'+leg+'Coxa_yaw'].get('range'),sep=' ');pitch=np.fromstring(joints['joint_'+leg+'Coxa_pitch'].get('range'),sep=' ')
  assert np.max(abs(yaw))<np.pi/2 and np.max(abs(pitch))<np.pi/2
  cap=math.acos(math.cos(max(abs(yaw)))*math.cos(max(abs(pitch))))+eps+1e-10
  ymin=min(0,math.sin(yaw[0]))-2*math.sin(eps/2)-1e-10;ymax=max(0,math.sin(yaw[1]))+2*math.sin(eps/2)+1e-10
  lower=[]
  for vec in target['reference_directions'][:,li,0]:
   lat=math.asin(np.clip(vec[1],-1,1));polar=math.acos(np.clip(-vec[2],-1,1))
   lower.append(max(0,polar-cap,lat-math.asin(ymax),math.asin(ymin)-lat))
  error=float(abs(np.rad2deg(lower)-row['first_segment_lower_bound_deg']).max());bounderror=max(bounderror,error);assert error<1e-10
  assert np.all(np.array(lower)>0) and np.all(np.array(lower)<=target['direction_residual_rad'][:,li,0]+1e-8)
 starts=list(csv.DictReader((D/'MULTISTART.csv').open()));assert len(starts)==36
 for info in caseinfo:
  values=[float(r['geometry_ss']) for r in starts if int(r['case'])==info['case']];assert min(values)==info['best_geometry_ss']
  if not info['mirror']:assert not info['significant_improvement']
 # Independent gain-specific sparse identity and scalar exponential decoder reconstruction.
 baseE=sparse.load_npz(B/'ACTUAL_We.npz');baseI=sparse.load_npz(B/'ACTUAL_Wi.npz');add=sparse.load_npz(G/'ADDED_We.npz');inputs=np.load(C/'INPUTS.npz');motors=inputs['motor_local'];selected=inputs['selected_local'];mapping=json.loads((C/'MAPPING.json').read_text());initial=np.load(C/'INITIAL_BOUNDARY.npz');reference=unpack(N/'BASELINE_NEURAL.npz');table=json.loads((C/'SELECTION_TABLE.json').read_text());rows=[];decodererror=0.;rasters={};decoders={}
 gains=[.03125,.0625,.125,.25,.5];selected_gain=float(table['selected_gain'])
 for name,gain in zip(names,gains+[selected_gain,selected_gain]):
  e=sparse.load_npz(C/f'{name}_ACTUAL_We.npz');inh=sparse.load_npz(C/f'{name}_ACTUAL_Wi.npz');expected=baseE+gain*add
  for actual,want in [(e,expected),(inh,baseI)]:assert all(np.array_equal(a,b) for a,b in [(actual.indptr,want.indptr),(actual.indices,want.indices),(actual.data,want.data)])
  ev=unpack(C/f'{name}_NEURAL.npz');de=np.load(C/f'{name}_DECODE.npz');rasters[name]=ev;decoders[name]=de;rate=initial['decoder_rates'].copy();decay=math.exp(-.0005/.005)
  for tick,spikes in enumerate(ev):
   instant=np.array([sum(bool(spikes[i]) for i in r['local_indices'])/len(r['local_indices'])/.0005 if r['local_indices'] else 0. for r in mapping['rows']]);rate=decay*rate+(1-decay)*instant
   decodererror=max(decodererror,float(abs(rate-de['rates_halfstep_hz'][tick]).max()))
   if tick%2:assert np.allclose(np.clip(rate/100,0,1),de['decoded_next'][tick//2],atol=1e-12,rtol=0)
  assert np.array_equal(de['applied'][0],initial['pending']) and np.array_equal(de['applied'][1:],de['decoded_next'][:-1])
  if name.startswith('GAIN_'):
   mn=int(ev[:,motors].sum());sat=int((de['applied']>=1).sum());bits=int((ev[:,motors]!=reference[:,motors]).sum());passed=54<=mn<=162 and sat<=90 and bits>0
   rows.append(dict(gain=gain,MN_spikes=mn,saturated=sat,MN_different_bits=bits,passed=passed))
 assert decodererror<1e-10 and max(r['gain'] for r in rows if r['passed'])==selected_gain
 for x,y in zip(rows,table['rows']):assert x['gain']==y['gain'] and x['MN_spikes']==y['MN_spikes'] and x['saturated']==y['applied_saturated'] and x['passed']==y['passed']
 assert np.array_equal(rasters['VALIDATE150'][:100],rasters['GAIN_0p03125'])
 for field in ['applied','decoded_next','rates_halfstep_hz']:
  count=100 if field=='rates_halfstep_hz' else 50;assert np.array_equal(decoders['VALIDATE150'][field][:count],decoders['GAIN_0p03125'][field])
 assert not inputs['cut_current_nA'][:,selected].any();keep=np.ones(12000,bool);keep[selected]=False;assert np.array_equal(inputs['saved_current_nA'][:,keep],inputs['cut_current_nA'][:,keep])
 val=decoders['VALIDATE150'];baseline150=np.load(C/'BASELINE150_REUSED.npz');report=dict(passed=True,source_sha256=hashes,geometry=dict(XML_Rodrigues_FK_maxabs=fkerror,scalar_analytic_lower_bound_max_difference_deg=bounderror,certified_frames_per_middle_leg=37,actual_native_diagnostic_cases=4,mirror_only_diagnostic_cases=2,starts_verified=36,significant_native_improvements=0,scope='Necessary bound under frozen transform and native model; not a recommendation to enlarge physiology.'),neural_calibration=dict(grid_reconstructed=rows,selected_gain=selected_gain,decoder_maxabs=decodererror,extension_first50_exact=True,candidate150_saturated=int((val['applied']>=1).sum()),baseline150_saturated=int((baseline150['applied']>=1).sum()),candidate150_elements=int(val['applied'].size),sourcecut_other_inputs_unchanged=True,scope='Short fixed-current engineering calibration; not body support improvement or physiological calibration.'),calls=dict(model_compile=0,body_step=0,neural_step=0,network=0,optimizer=0))
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
