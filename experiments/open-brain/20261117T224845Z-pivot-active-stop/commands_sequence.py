from commands import *

def generate_sequence(family):
 variant=7;out=HERE/'COMMANDS'/f'{family}_{variant}';out.mkdir(parents=True,exist_ok=False)
 start=2400;action=8200;off=15000;period=63;brake_off=17000
 schedule=[('forward',None,start,off),(GROUP[family],None,action,brake_off)]
 case=SimpleNamespace(name='bluebell',schedule=lambda:schedule);neck=ground.Ground(case,False,None)
 sources=freeze([NECK/'PARAMETERS.json',NECK/'READOUT.json',HERE/'RELEASE_SEQUENCE_PREREG.zh-CN.md'])
 write(out/'INPUTS.json',dict(source_sha256=sources,schedule=schedule,variant=variant,steps=20000))
 cmd=[];sent=[];arr=[]
 for tick in range(20000):
  s,a=neck.step(tick,period,np.zeros(2));cmd.append(neck.command.copy());sent.append(s.copy());arr.append(a.copy())
 roots=list(neck.circuit.port_roots);catalog=json.loads((BASE/'20261117T051249Z-bidirectional/CATALOG.json').read_text())['ports']
 np.savez_compressed(out/'TRACE.npz',command=np.array(cmd),emitted=np.array(sent),arrived=np.array(arr),roots=np.array(roots))
 ports=[dict(root=r,cell_type=catalog[r]['cell_type'],group=catalog[r]['group'],side=catalog[r]['side']) for r in roots]
 verify_sources(sources);write(out/'RESULT.json',dict(family=family,variant=variant,seed=9337,start=start,action=action,off=off,brake_off=brake_off,period=period,steps=20000,new_neural_ticks=20000,ports=ports,sha256=sha(out/'TRACE.npz')))
 print(family,'sequenced release source saved',flush=True)
if __name__=='__main__':
 for f in ('stop60','stop74a','stop74b'):generate_sequence(f)
