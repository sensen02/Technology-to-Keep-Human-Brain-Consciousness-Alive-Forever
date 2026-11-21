"""Independent RGB and floor readout check; score existing saved artifacts without changing the student."""
from pathlib import Path
import io,json,zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from optics import P,W,H,reconstruct
from fit_feeding import OUT,PRE,ObservedBody,torch
def verify(seed):
    d=OUT/f'aligned-{seed}'
    with np.load(d/'camera_cal.npz') as cal,np.load(d/'external.npz') as z,np.load(d/'floor.npz') as fl,zipfile.ZipFile(d/'cameras.zip') as frames:
        pos=z['markers_mm'];pressure=z['force_uN'];n=len(pos)
        assert n==len(frames.namelist())==len(fl['offsets'])-1
        assert len(fl['cells']) and np.isfinite(fl['cells']).all()
        total=[]
        with np.load(d/'electrode.npz') as ele:assert len(ele['counts'])==n
        for t in [1,n//2,n-1]:
            im=np.asarray(Image.open(io.BytesIO(frames.read(f'{t:04d}.png'))))
            points,views=reconstruct(im,cal['offsets'][t])
            assert np.allclose(points[3:21],pos[t,3:],equal_nan=True,atol=1e-5)
            fp=np.zeros(6)
            for x,y,f in fl['cells'][fl['offsets'][t]:fl['offsets'][t+1]]:
                distance=np.linalg.norm(points[5:21:3,:2]-.125*np.array([x,y]),axis=1)
                if np.isfinite(distance).any() and np.nanmin(distance)<.25:fp[np.nanargmin(distance)]+=f
            assert np.allclose(fp,pressure[t],atol=1e-4)
        blank,_=reconstruct(np.zeros_like(im),cal['offsets'][0]);assert np.isnan(blank).all()
        total=[float(fl['cells'][i:j,2].sum()) for i,j in zip(fl['offsets'][:-1],fl['offsets'][1:])]
        if seed==88207:
            fig,axes=plt.subplots(2,1,figsize=(7.8,5),layout='constrained',height_ratios=[1,1.1])
            sample=n//2
            tile=np.asarray(Image.open(io.BytesIO(frames.read(f'{sample:04d}.png'))))
            axes[0].imshow(tile[:H,:2*W]);axes[0].set_title(f'Two of eight simulated RGB views at {(sample+1)*.005:.3f} s');axes[0].axis('off')
            for i,k in enumerate(('LF','LM','LH','RF','RM','RH')):axes[1].plot(np.arange(n)*.005+.005,pressure[:,i],lw=1,label=k)
            first=json.loads((d/'RESULT.json').read_text())['first_intake_s']
            axes[1].axvline(first,color='black',ls='--',lw=1,label='First intake')
            axes[1].set(xlabel='Teacher time (s)',ylabel='Assigned vertical load (µN)')
            axes[1].legend(fontsize=7,ncol=4);fig.savefig(OUT/'sensor-evidence.png',dpi=145)
    return dict(frames=n,complete=int(np.isfinite(pos).all((1,2)).sum()),force_peak_uN=max(total),mean_assigned_uN=float(pressure.sum(1).mean()),assigned_fraction=json.loads((d/'optical.json').read_text())['assigned_force_fraction'])
if __name__=='__main__':
    o={str(s):verify(s) for s in (88207,88209,88211)}
    net=ObservedBody(OUT/'student/weights.pt');prior=ObservedBody(PRE/'weights.pt')
    assert all(torch.equal(a,b) for a,b in zip(net.raw.feedback.parameters(),prior.raw.feedback.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(net.raw.motion.parameters(),prior.raw.motion.parameters()))
    (OUT/'verified.json').write_text(json.dumps(o,indent=2));print(o)
