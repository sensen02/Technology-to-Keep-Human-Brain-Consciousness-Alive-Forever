"""Pin the tested calibration, full-brain inputs, structural proofs and frozen body."""
import json
from pathlib import Path
import sys

from overlap import BIDIR, HERE, RECEIVING, WEIGHT, digest, first, read, study, write
from brain_events import FOUNDATION
import torch
import mujoco


def freeze(out):
    timing = read(HERE/'BRAIN/181_off/INPUT.json')['equivalent_ring']
    assert timing['exact_all_arrays'] and timing['active_ticks']>0
    catalog=read(BIDIR/'CATALOG.json')['ports']
    admission=read(BIDIR/'EON/RESULT.json')['admitted']
    probes=read(HERE/'PROBE/off/RESULT.json')['probes']
    files={}
    binding=[]
    for root,wire in admission.items():
        row=catalog[root]
        if row['group'] not in RECEIVING:
            continue
        mesh=BIDIR/'geometry'/row['object']['filename']
        assert digest(mesh)==row['object']['sha256']
        files[str(mesh.resolve())]=row['object']['sha256']
        selected=probes[row['group']+'/'+row['side']]
        edge_weights=dict(row['outgoing'])
        assert all(edge_weights[target]>0 for target in selected)
        binding.append(dict(type=row['cell_type'],side=row['side'],flywire_v783=wire,banc_v888=root,
                            swc=str(mesh),crossing_children=row['geometry']['crossing_children'],
                            vnc_targets={target:edge_weights[target] for target in selected}))
    assert len(binding)==6
    parent=read(study.HERE/'PHASE/IDENTITY.json')['sha256']
    for name,value in parent.items():
        assert digest(Path(name))==value,name
    files.update(parent)
    paths=[HERE/item for item in ('brain_events.py','overlap.py','audit_overlap.py','run_brain.sh',
           'run_probe.sh','run_physics.sh','PROTOCOL.zh-CN.md','CALIBRATION.json')]
    paths.extend((FOUNDATION/'BENCH/INPUTS.json',BIDIR/'CATALOG_PRE_ABDOMEN.json',BIDIR/'EON/RESULT.json',
                  BIDIR/'CATALOG.json',first.MODEL_FILE,WEIGHT))
    for source in HERE.glob('BRAIN/*/INPUT.json'):
        paths += [source,source.parent/'BRAIN.npz',source.parent/'RESULT.json']
        entry=read(source.parent/'RESULT.json')
        assert entry['input_sha256']==digest(source) and entry['trace_sha256']==digest(source.parent/'BRAIN.npz')
    for path in paths:
        files[str(path.resolve())]=digest(path)
    write(out,dict(source_brain='Eon FlyWire v783',neck='BANC v888',bindings=binding,
                   parity=timing,python=sys.version,torch=torch.__version__,mujoco=mujoco.__version__,
                   sha256=files))


if __name__=='__main__':
    freeze(Path(sys.argv[1]))
