"""Stdlib orchestration/identities; heavy imports only explicit backend() after parent flag."""
from pathlib import Path
import json,hashlib,os,sys,importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parents[2];VIS=PROJECT/'experiments/visual-orientation/20261112T045316Z';BODY=VIS/'landmark_opponent_v7/body';PORTS=ROOT/'dnp09_interface'
FORMAT='ENGINEERING_SOURCE_NECK_OFFLINE_CALIBRATION_V1';DN=['720575941433155799','720575941566493282'];AN=['720575941351919408'];TRAIN=('V_empty','V_center','V_left_pulse','V_right_pulse');DEV=('V_left_to_right','V_right_to_left');MASK=list(range(0,42))+list(range(84,126))+list(range(168,195))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def verify(hashes):
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Source drift '+p)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def backend():
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1 required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});import torch,numpy as np;torch.set_num_threads(1)
 if any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')):raise ValueError('No physics or motor model permitted')
 return torch,np
def binding():
 p=PORTS/'SOURCE_BINDING.json';b=read(p);digest=sha(p)
 if digest!=read(HERE/'PREREG.json')['actual_parent_source_binding_sha256']:raise ValueError('Actual parent preregistered source binding digest mismatch')
 if b.get('synthetic_fixture') is True or b.get('scope')!='SCOPED_ENGINEERING_NECK_BODY_INTERFACE':raise ValueError('Actual scoped engineering source binding required')
 ports=b['ports'];actual={d:[x['root_id'] for x in ports if x['direction']==d] for d in ('DN','AN')}
 if actual!={'DN':DN,'AN':AN}:raise ValueError('Exactly two source DNp09 and one AN original ordering required')
 for row in b['source_files']:verify({row['path']:row['sha256']})
 for port in ports:
  if port['source_reviewed'] is not True or port['structural_evidence_status']!='SOURCE_GEOMETRY_REVIEWED':raise ValueError('Reviewed real source geometry required')
  for e in port['evidence']:verify({e['path']:e['sha256']})
 if b.get('transport_config')!=dict(dt_s=.001,delay_steps=2,gain=1.,activity_min=-1.,activity_max=1.,evidence='ENGINEERING_ASSUMPTION_NOT_MEASURED_CONDUCTION'):raise ValueError('2ms engineering source queue required')
 return p,b,digest
def sources():
 names=('PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')
 result=[HERE/n for n in names]+[ROOT/'connectivity_audit/PARTNER_AUDIT.json',ROOT/'connectivity_audit/DIRECT_INTERFACE_EVIDENCE.json',ROOT/'SELECTED_V888_VERSION_MAPPING.json',PORTS/'api.py',PORTS/'source_transport.py',PORTS/'build_binding.py',PORTS/'NUMERIC_CHECK.json',PORTS/'TRANSPORT_JOURNAL.json',ROOT/'body_interface/interface.py']
 p,b,digest=binding();result.append(p);result.extend(Path(row['path']) for row in b['source_files'])
 for port in b['ports']:result.extend(Path(e['path']) for e in port['evidence'])
 return sorted(set(result),key=str)
def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);p,b,d=binding()
 if d!=f['source_binding_sha256']:raise ValueError('Bound source digest changed')
 return f
def real_queue_rollout(m,body111,cue,binding_path,binding_digest):
 # Parent no_grad diagnostic: source actual1ms FIFO; unaugmented neural values only.
 import torch
 if body111.shape[1]!=1:raise ValueError('Real queue replay is batch1 only')
 prior=list(sys.path);sentinel=object();oldapi=sys.modules.get('api',sentinel)
 try:
  sys.path.insert(0,str(PORTS));sys.modules['api']=module('_calibration_source_api',PORTS/'api.py');transport=module('_calibration_source_transport',PORTS/'source_transport.py').SourceTransport(binding_path,binding_digest)
 finally:
  sys.path[:]=prior
  if oldapi is sentinel:sys.modules.pop('api',None)
  else:sys.modules['api']=oldapi
 h=body111.new_zeros((1,16));held={'AN':body111.new_zeros((1,1)),'DN':body111.new_zeros((1,2))};held_packets={};rows=[[],[],[],[],[]];consumption=[]
 with torch.no_grad():
  for k in range(1920):
   if transport.step!=5*k:raise ValueError('Actual source queue clock drift')
   for direction,packet in held_packets.items():
    used=dict(packet,consume_step=transport.step);transport_api=sys.modules['_calibration_source_api'];transport_api.validate_delivery(used,transport.ports[direction],direction,transport.digest,transport.step);consumption.append(dict(direction=direction,capture_step=packet['capture_step'],queue_release_step=packet['release_step'],neural_consume_step=transport.step))
   an=m.feedback(m.normalize(body111[k]));dn,h=m.brain_step(held['AN'],cue[k],h);turn=m.decoder(held['DN']);rec=m.reconstruction(an)
   for row,value in zip(rows,(turn,dn,an,rec,h)):row.append(value.clone())
   for d,value in (('DN',dn),('AN',an)):transport.send(dict(**{'from':'brain' if d=='DN' else 'other','to':'neck'},direction=d,root_ids=transport.ports[d],values=value[0].tolist(),capture_step=transport.step,source_bindings_sha256=transport.digest))
   for ms in range(5):
    for packet in transport.advance():held[packet['direction']]=body111.new_tensor([packet['values']]);held_packets[packet['direction']]=dict(packet)
 result=transport.verify_final();result.update(neural_consumption_events=consumption,actual_source_queue_journal=transport.journal(),initial_delivered_AN_DN_zero=True,brain_neural_memory=16,motor256_loaded=False)
 if result['actual_steps']!=9600:raise ValueError('Full1920x5ms actual source queue required')
 return tuple(torch.stack(x) for x in rows),result

def load_checkpoint(path,expected_sha,final=True):
 torch,np=backend();f=frozen()
 if sha(path)!=expected_sha:raise ValueError('Actual checkpoint SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['source_binding_sha256']!=f['source_binding_sha256'] or p['step'] not in (0,100,200) or (final and p['step']!=200):raise ValueError('Scoped final200 checkpoint required')
 verify(p['source_hashes']);verify(p['data_hashes']);from model import NeuralBridge
 m=NeuralBridge(p['state_dict']['observation_mean'],p['state_dict']['observation_std']);m.load_state_dict(p['state_dict'],strict=True)
 if m.identity()!=p['identity'] or len(p['curve'])!=p['step']+1:raise ValueError('Checkpoint architecture/update lineage')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Nonfinite model')
 opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=len(list(m.parameters())) or any(float(x['step'])!=p['step'] for x in opt.state.values())):raise ValueError('All optimizer counters must match actual200 updates')
 return m.eval(),p
