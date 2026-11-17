"""Explicit newtimestamp/immutable parent lineage; no heavy imports before parent flag."""
from pathlib import Path
import json,hashlib,sys,importlib.util
HERE=Path(__file__).resolve().parent
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ORIGIN=PROJECT/'experiments/neck-real-structure/20261112T225331Z';OLD=ORIGIN/'neural_calibration';BODY=ORIGIN/'body_pilot';PORTS=ORIGIN/'dnp09_interface'
s=importlib.util.spec_from_file_location('_aggregation_parent_frozen_common',OLD/'common.py');old=importlib.util.module_from_spec(s);sys.modules[s.name]=old;s.loader.exec_module(old)
FORMAT='SOURCE_NECK_FEEDBACK_AGGREGATION_V2';PARENT=OLD/'TRAIN/checkpoint_000200.pt';PARENT_SHA='eed1ad3cc61a1b59086ff2324175f6bf3b5da5aa197165e6c4b009c17027c95e'
TRAIN=('V_empty','V_center','V_left_pulse','V_right_pulse','P_empty','P_left_pulse','P_right_pulse');DEV=('V_left_to_right','V_right_to_left');MASK=old.MASK;DN=old.DN;AN=old.AN
sha=old.sha;read=old.read;save=old.save;verify=old.verify;module=old.module;backend=old.backend;real_queue_rollout=old.real_queue_rollout

def binding():
 p,b,d=old.binding()
 if str(p)!=read(HERE/'PREREG.json')['source_binding'] or d!=read(HERE/'PREREG.json')['source_binding_sha256']:raise ValueError('Newtimestamp source binding mismatch')
 return p,b,d

def initialize():
 torch,np=backend();m,p=old.load_checkpoint(PARENT,PARENT_SHA,final=True);m.requires_grad_(True)
 if any(not torch.equal(v,p['state_dict'][k]) for k,v in m.state_dict().items()):raise ValueError('Parentfinaleed1 initialization must be bitexact')
 return m,p

def sources():
 names=('PREREG.json','common.py','model.py','data.py','preflight.py','train.py','verify_saved.py','evaluate.py','run.py','codecheck.py')
 result=[HERE/n for n in names]+[OLD/n for n in ('common.py','model.py','data.py','FROZEN.json','DATASET.json','train.npz','development.npz','TRAIN/VERIFY_000200.json','EVALUATION/RESULT.json','EVALUATION/VERIFY.json')]+[PARENT,PORTS/'source_transport.py',PORTS/'api.py',PORTS/'NUMERIC_CHECK.json',PORTS/'TRANSPORT_JOURNAL.json',BODY/'runtime.py',BODY/'runner.py',BODY/'audit.py',BODY/'summary.py',BODY/'BODY_SUMMARY.json',BODY/'FROZEN.json']
 p,b,d=binding();result.append(p);result.extend(Path(x['path']) for x in b['source_files'])
 for port in b['ports']:result.extend(Path(x['path']) for x in port['evidence'])
 return sorted(set(result),key=str)

def frozen():
 f=read(HERE/'FROZEN.json');verify(f['source_hashes']);verify(f['data_hashes']);_,_,digest=binding()
 if digest!=f['source_binding_sha256']:raise ValueError('Source binding drift')
 return f

def load_checkpoint(path,digest,final=True):
 torch,np=backend();f=frozen()
 if sha(path)!=digest:raise ValueError('Actual checkpoint SHA')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p['format_version']!=FORMAT or p['parent_sha256']!=PARENT_SHA or p['step'] not in (0,200,400) or (final and p['step']!=400):raise ValueError('Onlyfixed400 newformatlineage allowed')
 if p['source_binding_sha256']!=f['source_binding_sha256'] or len(p['curve'])!=p['step']+1 or p['batch_size']!=7:raise ValueError('Source/batch/update lineage drift')
 verify(p['source_hashes']);verify(p['data_hashes']);m,parent=initialize();m.load_state_dict(p['state_dict'],strict=True)
 if m.identity()!=p['identity']:raise ValueError('Same artificialbrain16/N2M1 architecture required')
 for k in ('observation_mean','observation_std','allowed_raw_indices'):
  if not torch.equal(m.state_dict()[k],parent['state_dict'][k]):raise ValueError('Parent normalization/mask must remain exactly fixed')
 if not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('Nonfinite neural parameters')
 opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 if p['step'] and (len(opt.state)!=len(list(m.parameters())) or any(float(v['step'])!=p['step'] for v in opt.state.values())):raise ValueError('All optimizerstate counters match actual400 updates')
 return m.eval(),p
