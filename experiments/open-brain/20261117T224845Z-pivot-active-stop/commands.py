from scope import *
from types import SimpleNamespace
NECK=BASE/'20261117T164207Z-fixed-neck'
sys.path.insert(0,str(NECK));import adapters;adapters.install(True)
sys.path.insert(0,str(BASE/'20261117T071007Z-ground-dn-response'));import ground_response as ground
GROUP={'pivot_left':'walking_steer','pivot_right':'walking_steer','stop60':'halt_bluebell','stop74a':'halt_web_a','stop74b':'halt_web_b'}

def generate(family,variant):
 out=HERE/'COMMANDS'/f'{family}_{variant}';out.mkdir(parents=True,exist_ok=False)
 start=2200+170*variant;action=8000+170*variant;off=15000+150*variant;period=(59,65,62)[variant]
 group=GROUP[family];side=family.split('_')[-1] if family.startswith('pivot') else None
 schedule=[(group,side,start,off)] if family.startswith('pivot') else [('forward',None,start,off),(group,None,action,off)]
 case=SimpleNamespace(name='bluebell',schedule=lambda:schedule)
 neck=ground.Ground(case,False,None)
 sources=freeze([NECK/'PARAMETERS.json',NECK/'READOUT.json',BASE/'20261117T051249Z-bidirectional/CATALOG.json'])
 write(out/'INPUTS.json',dict(source_sha256=sources,schedule=schedule,variant=variant,steps=20000))
 commands=[];sent=[];arrived=[]
 for tick in range(20000):
  s,a=neck.step(tick,period,np.zeros(2));commands.append(neck.command.copy());sent.append(s.copy());arrived.append(a.copy())
 roots=list(neck.circuit.port_roots)
 np.savez_compressed(out/'TRACE.npz',command=np.array(commands),emitted=np.array(sent),arrived=np.array(arrived),roots=np.array(roots))
 catalog=json.loads((BASE/'20261117T051249Z-bidirectional/CATALOG.json').read_text())['ports']
 ports=[dict(root=r,cell_type=catalog[r]['cell_type'],group=catalog[r]['group'],side=catalog[r]['side']) for r in roots]
 verify_sources(sources);write(out/'RESULT.json',dict(family=family,variant=variant,seed=SEEDS[variant],start=start,action=start if family.startswith('pivot') else action,off=off,
   period=period,steps=20000,new_neural_ticks=20000,sha256=sha(out/'TRACE.npz'),ports=ports))
 print(family,variant,'actual neck trace saved',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('split',choices=('dev','held'));a=p.parse_args()
 for v in ((0,1) if a.split=='dev' else (2,)):
  for f in FAMILIES:generate(f,v)
