"""Actual frozen cervical dynamics from named DN stimuli; no body or ANN calls."""
from scope import *
NECK=BASE/'20261117T164207Z-fixed-neck'
sys.path.insert(0,str(NECK))
import adapters
adapters.install(True)
from adapters import expansion as study
sys.path.insert(0,str(BASE/'20261117T071007Z-ground-dn-response'))
import ground_response as ground
from physical import schedule_indices


def generate(family,variant):
 out=HERE/'COMMANDS'/f'{family}_{variant}'
 if (out/'RESULT.json').exists():
  meta=json.loads((out/'RESULT.json').read_text());assert sha(out/'TRACE.npz')==meta['sha256'];return meta
 out.mkdir(parents=True,exist_ok=False)
 start=2200+170*variant;action=5500+130*variant;off=14500+150*variant;period=(59,65,62)[variant]
 if family.startswith('stride'):
  case=ground.Case(family,SEEDS[variant],period,start,action,off,0)
  params=json.loads((ground.HERE/'REFINE/CALIBRATION.json').read_text())
  neck=ground.Ground(case,False,params)
 else:
  case=study.Case(family,family if family!='speed' else 'cruise',SEEDS[variant],period,start,off)
  neck=study.Neck(case)
  if family=='speed':neck.schedule=schedule_indices(neck.circuit,[('forward',None,start,off),('forward_speed',None,6500+170*variant,10500+170*variant)])
  elif family.startswith('arc'):neck.schedule=schedule_indices(neck.circuit,[('forward',None,start,off),('walking_steer',family.split('_')[1],action,off)])
 sources=freeze([NECK/'PARAMETERS.json',NECK/'READOUT.json',NECK/'fixed_neck.py'])
 for p in (BASE/'20261117T051249Z-bidirectional/CATALOG.json',ground.HERE/'REFINE/CALIBRATION.json'):sources[str(p)]=sha(p)
 write(out/'INPUTS.json',dict(source_sha256=sources,family=family,variant=variant,start=start,action=action,off=off,period=period,steps=20000,front_AN='zero isolated body-command test'))
 commands=[];sent=[];arrived=[]
 for tick in range(20000):
  s,a=neck.step(tick,period,np.zeros(2));commands.append(neck.command.copy());sent.append(s.copy());arrived.append(a.copy())
 np.savez_compressed(out/'TRACE.npz',command=np.array(commands),emitted=np.array(sent),arrived=np.array(arrived),roots=np.array(neck.circuit.port_roots))
 verify_sources(sources)
 meta=dict(family=family,variant=variant,seed=SEEDS[variant],start=start,action=action,off=off,period=period,
  speed_on=6500+170*variant,speed_off=10500+170*variant,steps=20000,new_neural_ticks=20000,sha256=sha(out/'TRACE.npz'),
  ports=list(neck.circuit.port_roots))
 # The catalog itself is source-bound; retain actual roots and emitted/arrived pulse arrays.
 meta['ports']=list(neck.circuit.port_roots)
 write(out/'RESULT.json',meta);print(family,variant,'neck trace ready',flush=True)
 return meta

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held'));a=p.parse_args()
 for variant in ((0,1) if a.split=='dev' else (2,)):
  for family in FAMILIES:generate(family,variant)
