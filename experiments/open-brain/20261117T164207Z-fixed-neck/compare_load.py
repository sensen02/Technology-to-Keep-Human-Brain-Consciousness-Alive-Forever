import numpy as np
from fixed_neck import FixedCircuit,LegacyCircuit,full_catalog
root='720575941685691375'
for C in [LegacyCircuit,FixedCircuit]:
 c=C(full_catalog())
 indices=c.validate_inputs([r for r,n in c.ports[root]['incoming_receptors']],'body')
 stimulus=np.zeros((1,len(c.roots)),dtype=np.float32)
 mask=np.zeros(len(c.ports),dtype=bool)
 peak=-1e9;total=0
 for tick in range(3000):
  stimulus.fill(0.)
  if tick%23==0:stimulus[0,indices]=68.75
  e,a,s,v=c.step(stimulus,mask)
  peak=max(peak,float(v[0,c.index[root]]));total+=e[0,c.port_index[root]]
 print(C.__name__,peak,total)
