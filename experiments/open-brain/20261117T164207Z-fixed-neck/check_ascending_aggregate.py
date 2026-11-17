"""Probe weak ascending relays with sustained receptor input and exact restoration."""
import numpy as np
from fixed_neck import FixedCircuit, LegacyCircuit, HERE, full_catalog, write, hash_file

catalog=full_catalog()
weak=[root for root,row in __import__('json').loads((HERE/'DYNAMICS_VERIFIED.json').read_text())['all_ports']['ports'].items()
      if not row['fires_under_test_stimulus']]
output={}
for root in weak:
    result={}
    for cls in (LegacyCircuit,FixedCircuit):
        model=cls(catalog)
        columns=model.validate_inputs([r for r,n in model.ports[root]['incoming_receptors']],'body')
        drive=np.zeros((1,len(model.roots)),dtype=np.float32)
        idx=model.port_index[root]
        source=0;arrival=0;after=0
        vmin=vmax=-52.
        for tick in range(4000):
            drive.fill(0.)
            if tick<3000 and tick%23==0:
                drive[0,columns]=68.75
            cut=np.array([r==root and 1200<=tick<1900 for r in model.port_roots])
            sent,arrived,spikes,voltage=model.step(drive,cut)
            source+=int(sent[0,idx]);arrival+=int(arrived[0,idx])
            if tick>=1920:
                after+=int(arrived[0,idx])
            if 1200<=tick<1900:
                assert not arrived[0,idx]
            vmin=min(vmin,float(voltage[0,model.index[root]]))
            vmax=max(vmax,float(voltage[0,model.index[root]]))
        result[cls.__name__]=dict(emitted=source,arrived=arrival,arrived_after_restore=after,
             min_mv=vmin,max_mv=vmax,stimulus_period_ms=2.3,stimulus_duration_ms=300)
    output[root]=dict(cell_type=model.ports[root]['cell_type'],group=model.ports[root]['group'],models=result)
    print(root,output[root],flush=True)
write(HERE/'ASCENDING_SUSTAINED.json',dict(ports=output,runtime_sha256=hash_file(HERE/'fixed_neck.py'),
       params_sha256=hash_file(HERE/'PARAMETERS.json'),threshold_adjusted=False))
