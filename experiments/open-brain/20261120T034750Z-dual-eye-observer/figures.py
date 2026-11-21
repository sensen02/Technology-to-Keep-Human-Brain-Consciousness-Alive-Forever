import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

d=Path(sys.argv[1])
s=np.load(d/'compound.npz')
poses={int(k):v for k,v in json.loads((d/'estimated.json').read_text()).items() if v is not None}
walk=np.array([k for k in poses if k<=315]); t=2*walk
central=np.array([poses[k]['pos_mm'] for k in walk])
axes3=np.array([poses[k]['forward'] for k in walk])
yaw=np.degrees(np.unwrap(np.arctan2(axes3[:,1],axes3[:,0])))
pitch=np.degrees(np.arcsin(np.clip(axes3[:,2],-1,1)))
fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
ax=axes[0,0]
base=central[0,:2]
relative=central[:,:2]-base
ax.plot(*relative.T,'o-',ms=3,color='#1e5775')
ax.quiver(*relative.T,*axes3[:,:2].T,angles='xy',scale_units='xy',scale=3.5,color='#245e57')
ax.set_title('Head track and forward axis');ax.set_xlabel('X from start (mm)');ax.set_ylabel('Y from start (mm)');ax.set_aspect('equal')
ax=axes[0,1]
ax.plot(t,yaw,'o-',ms=3,label='Yaw');ax.plot(t,pitch,'s-',ms=3,label='Pitch')
ax.set_title('Walking head orientation');ax.set_xlabel('Time (ms)');ax.set_ylabel('Angle (°)');ax.legend()
a=json.loads((d/'AUDIT.json').read_text())
metrics={e['frame']:e for e in a['per_frame']}
ax=axes[1,0]
ax.plot(t,[metrics[k]['head_um'] for k in walk],'o-',ms=3,color='#1e5775')
ax.set_title('Head position error');ax.set_xlabel('Time (ms)');ax.set_ylabel('3D position error (µm)')
ax=axes[1,1]
for eye,color in [('L','#1e5775'),('R','#b55330')]:
    ax.plot(t,[metrics[k][eye+'_axis_deg'] for k in walk],'o-',ms=3,color=color,label=eye)
ax.set_title('Optical-axis error');ax.set_xlabel('Time (ms)');ax.set_ylabel('Angle error (°)');ax.legend()
for ax in axes.ravel():ax.grid(alpha=.18)
fig.savefig(d/'HEAD_POSE.png',dpi=150);plt.close(fig)
frames=[211,317,318]
fig,axes=plt.subplots(len(frames),5,figsize=(12.8,7.6),layout='constrained')
for row,k in enumerate(frames):
    arr=s['readout'][np.where(s['frames']==k)[0][0]].sum(-1)
    pix=s['ommatidia_id_map'].astype(int)
    render=lambda e:np.where(pix>0,arr[e][np.maximum(pix-1,0)],np.nan)
    for col,name in enumerate(['external','L','R','compound_l','compound_r']):
        ax=axes[row,col];ax.set_axis_off()
        if name=='external':
            im=np.array(Image.open(d/'external'/f'{k}_H02_NW.png'))
            ax.imshow(im)
            heading='Walk, 422 ms' if k<=315 else ('Head-only +25°' if k==317 else 'Head-only −25°')
            ax.set_title(heading,fontsize=10,loc='left')
        elif name in 'LR':
            ax.imshow(Image.open(d/'eyes'/f'{k}_{name}.png'))
        else:
            ax.imshow(render(0 if name.endswith('l') else 1),cmap='gray',vmin=0,vmax=.68)
        if row==0 and col:
            ax.set_title(['Left eye','Right eye','Left compound','Right compound'][col-1],fontsize=10)
fig.savefig(d/'DUAL_EYES.png',dpi=155)
