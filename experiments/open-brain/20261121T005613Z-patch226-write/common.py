from pathlib import Path
import json,sys,numpy as np
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
HERE=Path(__file__).resolve().parent
OMNI=Path('/run/media/sensen/Data1/omnisim')
sys.path.insert(0,str(OMNI))
FULL=Path('/run/media/sensen/Data1/omnisim-results/implant-site/banc-neck/patch-pitch5/full5')
from omnisim.modules.banc_neck import load_fixed_neck
from omnisim.modules.electrode_observation import AxonProbe, ElectrodeObservation
read=lambda p:json.loads(Path(p).read_text())
def write(name, v):
    (HERE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
IO=read(ROOT/'experiments/open-brain/20261120T155833Z-body-surrogate/IO.json')
AN=IO['an_roots_left_right']
geom=dict(np.load(FULL/'acute-state.npz'))
live=np.flatnonzero(geom['live'])
def contacts(key):
    return (geom[key+'_base_m']+geom[key+'_axis']*geom[key+'_length_um'][:,None]*1e-6)[live]
tips=contacts('actual')
paths=load_fixed_neck(ROOT)[0]
assert [p.root for p in paths]==geom['registered_roots'].tolist()
roots=[p.root for p in paths]
target=[roots.index(t) for t in AN]
arms=['silent','left','right','both']
def patterns():
    sel=read(HERE/'selection.json')['sides']
    pat=np.zeros((4,len(live)))
    for j,row in enumerate(sel,1):
        a,b=row['live_indices']
        pat[j,[a,b]]=np.array([1,-1])*row['working_A']
    pat[3]=pat[1]+pat[2]
    return pat
