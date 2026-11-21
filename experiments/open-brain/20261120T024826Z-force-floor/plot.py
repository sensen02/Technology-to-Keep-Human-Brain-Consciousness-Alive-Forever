import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
d=Path(__file__).resolve().parent
t=np.arange(1,316)*2
loads=np.array([float(r['fz_uN']) for r in csv.DictReader((d/'insert/feet.csv').open())]).reshape(-1,6)
truth=np.load(d/'insert/private_walk.npz')['forces'][:,:6]
leg='LF LM LH RF RM RH'.split()
pal=['#b74149','#b46d22','#8e790b','#008e73','#2268a5','#94427e']
fig,axes=plt.subplots(3,2,figsize=(11,6.8),sharex=True,sharey=True,layout='constrained')
for k,ax in enumerate(axes.T.flat):
    ax.axhline(0,color='#819299',lw=.7)
    ax.plot(t,truth[:,k],color='#444',lw=1.8,alpha=.65)
    ax.plot(t,loads[:,k],color=pal[k],lw=1)
    ax.set_title(leg[k],loc='left',fontweight='bold',fontsize=10)
    ax.grid(alpha=.16)
fig.supxlabel('Time (ms)');fig.supylabel('Signed vertical force (µN)')
fig.suptitle('Six-foot loading on the sand-replica insert  |  500 Hz')
axes[0,0].legend([Line2D([],[],color='#444'),Line2D([],[],color=pal[0])],['Solver reference','Optical estimate'],fontsize=8)
fig.savefig(d/'LOADS.png',dpi=155)
fig,ax=plt.subplots(2,2,figsize=(10.4,6.9),layout='constrained')
idx=212
with np.load(d/'insert/external/H02_NW.npz') as a:
    pix=a['background']^np.bitwise_xor.reduce(a['delta'][:idx+1])
    rgb=a['palette'][pix]
ax[0,0].imshow(rgb);ax[0,0].set_title('External camera H02  |  426 ms');ax[0,0].axis('off')
r=[p for p in csv.DictReader((d/'insert/patches.csv').open()) if int(p['frame'])==idx]
xy=np.load(d/'insert/observed_feet_mm.npy')[idx]
for k,(x,y,z) in enumerate(xy):
    ax[0,1].scatter(x,y,s=55,facecolors='none',edgecolors=pal[k])
    ax[0,1].annotate(leg[k],(x,y),xytext=(5,6),textcoords='offset points',fontsize=9)
for p in r:
    f=float(p['fz_uN'])
    ax[0,1].scatter(float(p['x_mm']),float(p['y_mm']),s=22*abs(f)+8,marker='+',color='#222')
ax[0,1].set_title('Foot markers and measured load centres');ax[0,1].set_aspect('equal');ax[0,1].set_xlabel('x (mm)');ax[0,1].set_ylabel('y (mm)')
for j,name in enumerate(['punches','merged']):
    p=list(csv.DictReader((d/name/'patches.csv').open()))
    for col in sorted(set(round(float(row['x_mm']),2) for row in p)):
        pp=[u for u in p if round(float(u['x_mm']),2)==col]
        ax[1,j].scatter([float(u['time_ms']) for u in pp],[float(u['fz_uN']) for u in pp],s=6)
    ax[1,j].set_ylim(-.4,6.8);ax[1,j].grid(alpha=.2);ax[1,j].set_xlabel('Time (ms)');ax[1,j].set_ylabel('Detected load (µN)')
ax[1,0].set_title('0.50 mm probe spacing: two footprints')
ax[1,1].set_title('0.10 mm probe spacing: one footprint')
fig.savefig(d/'FLOOR.png',dpi=155)
