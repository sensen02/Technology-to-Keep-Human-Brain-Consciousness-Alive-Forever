import numpy as np
from fixed_neck import FixedCircuit, HERE, full_catalog, write
catalog=full_catalog()
records={}
for start in (-70.,-52.):
 model=FixedCircuit(catalog)
 root=next(r for r,p in model.ports.items() if p['group']=='halt_web_a' and p['side']=='left')
 target=max(model.ports[root]['outgoing'],key=lambda x:x[1])[0]
 index=model.index[target]
 drive=np.zeros((1,len(model.roots)),dtype=np.float32)
 mask=np.zeros(len(model.ports),dtype=bool)
 values=[]
 for tick in range(300):
  drive.fill(0.)
  if tick==5:
   drive[0,model.index[root]]=68.75
  model._voltage[0,index]=start
  model.step(drive,mask)
  values.append([*model.last_conductance[:,0,index],*model.last_current[:,0,index]])
 values=np.asarray(values)
 peak=int(np.argmax(values[:,:3].sum(1)))
 i=values[peak,3:].sum()
 assert i>0 if start==-70. else i<0
 records[str(start)]=dict(peak_conductance_ns=values[peak,:3].sum(),current_pa=float(i))
write(HERE/'REVERSAL_CHECK.json',records)
print(records)
