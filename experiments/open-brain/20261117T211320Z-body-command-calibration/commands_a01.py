"""Additional actual DNa01 neck traces within walking-turn calibration."""
from commands_v2 import *
from types import SimpleNamespace

def generate_a01(side,variant):
 family='a01_'+side;out=HERE/'COMMANDS'/f'{family}_{variant}';out.mkdir(parents=True,exist_ok=False)
 start=2200+170*variant;action=5500+130*variant;off=14500+150*variant;period=(59,65,62)[variant]
 case=SimpleNamespace(name='stride_'+side,schedule=lambda:[('forward',None,start,off),('walking_steer_context',side,action,off)])
 neck=ground.Ground(case,False,None)
 sources=freeze([NECK/'PARAMETERS.json',NECK/'READOUT.json',NECK/'fixed_neck.py'])
 write(out/'INPUTS.json',dict(source_sha256=sources,family=family,variant=variant,steps=20000,front_AN='zero isolated command test'))
 commands=[];sent=[];arrived=[]
 for tick in range(20000):
  s,a=neck.step(tick,period,np.zeros(2));commands.append(neck.command.copy());sent.append(s.copy());arrived.append(a.copy())
 np.savez_compressed(out/'TRACE.npz',command=np.array(commands),emitted=np.array(sent),arrived=np.array(arrived),roots=np.array(neck.circuit.port_roots))
 verify_sources(sources)
 write(out/'RESULT.json',dict(family=family,variant=variant,seed=SEEDS[variant],start=start,action=action,off=off,period=period,steps=20000,new_neural_ticks=20000,sha256=sha(out/'TRACE.npz'),ports=list(neck.circuit.port_roots)))
 print(family,variant,'neck trace ready',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held'));a=p.parse_args()
 for v in ((0,1) if a.split=='dev' else (2,)):
  for side in ('left','right'):generate_a01(side,v)
