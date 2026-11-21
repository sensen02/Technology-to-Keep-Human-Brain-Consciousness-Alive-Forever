"""Turn the teacher's Eon output into causal 226-needle observations before training."""
from pathlib import Path
import sys,json
import numpy as np
from dataclasses import replace
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'20261121T012712Z-body226-distal-cut'
sys.path.insert(0,str(PARENT))
import record as r
def run(folders):
    b=HERE.parent/'20261117T051249Z-bidirectional'
    groups=r.read(b/'CATALOG.json')['ports']
    admitted=r.read(b/'EON/RESULT.json')['admitted']
    order=[k for k in admitted if groups[k]['group'] not in ('takeoff','wing_amplitude','landing','flight_steer')]
    assert set(order)==set(r.IO['dn_roots']) and len(order)==17
    indices=[order.index(root) for root in r.IO['dn_roots']]
    c=np.load(PARENT/'calibration.npz');basis=c['basis'];unit=c['unit'];sigma=c['sigma']
    front=r.FrontEnd(np.sqrt(2*np.pi)*.15e-6)
    sel=[r.roots.index(k) for k in r.IO['dn_roots']]
    cut=[]
    with np.load(PARENT/'extensions.npz') as ext:
        for k in sel:
            path=r.paths[k];parts=[ext[key] for key in ext.files if key.startswith(path.root+'_')]
            shared=[]
            for pts in zip(*parts):
                if not np.allclose(pts,pts[0],atol=1e-13,rtol=0):break
                shared.append(pts[0])
            pts=path.points_m
            if len(shared)>1:pts=np.vstack((pts,np.array(shared)[1:]))
            cut.append(replace(path,points_m=pts))
    mask=~r.patterns().any(0);assert np.array_equal(r.live[mask],c['read_site_ids'])
    waves=(r.raw_bank(cut,r.contacts('actual')[mask],front)@basis)*r.geom['route_gain'][sel,None,None].astype('f')
    width=r.WIDTH;gram=np.zeros((2*width-1,len(unit),len(unit)),'f')
    for lag in range(width):
        dot=unit[:,lag:].reshape(len(unit),-1)@unit[:,:width-lag].reshape(len(unit),-1).T
        gram[width-1+lag]=dot;gram[width-1-lag]=dot.T
    for folder in folders:
        rng=np.random.default_rng(175+int(folder.name.split('-')[1]))
        with np.load(folder/'NECK.npz') as z:source=z['emitted'][:,indices]
        voltage=r.mix(source,waves)
        for i in range(0,len(voltage),8000):
            n=min(8000,len(voltage)-i)
            voltage[i:i+n]+=(front.noise(n,int(mask.sum()),rng)[0]@basis)*1e6
        dn=r.sort(r.scores(voltage,unit),gram,sigma,r.read(PARENT/'calibration.json')['threshold'])[0][:,:17]
        np.savez_compressed(folder/'electrode.npz',counts=dn.reshape(-1,50,17).sum(1),roots=r.IO['dn_roots'],voltage_excerpt_uV=voltage[:400])
        print(folder,int(source.sum()),int(dn.sum()),flush=True)
if __name__=='__main__':run([Path(v) for v in sys.argv[1:]])
