"""Saved-array supplementary command/roll plot, no simulation or weight edits."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/sensen/Desktop/engine修复验收/s116-offline-roll-channels/20261024T063500Z')
paths=[Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z/dt25us/TRAJECTORY.npz'),Path('/home/sensen/Desktop/engine修复验收/s115-extended-initialization/20261024T062500Z/init80ms/TRAJECTORY.npz')]
z=[]
for p in paths:
 with np.load(p) as a:z.append({k:a[k] for k in a.files})
fig,ax=plt.subplots(5,1,figsize=(13,15),sharex=True)
for n,a in zip(('40ms','80ms'),z):
 t=a['time']*1000; raw=(a['features']*a['weights']).sum(2)
 for j in (2,4,5):
  po=int(a['actuator_pool'][j]);ax[0].step(np.arange(7)*.5,np.r_[raw[:,po],raw[-1,po]],where='post',label=f'{n} act{j} raw')
  ax[1].step(np.arange(7)*.5,np.r_[a['applied'][:,j],a['applied'][-1,j]],where='post',label=f'{n} act{j} actual clipped')
 ax[2].plot(t,a['qpos'][:,9]-.6236,label=n+' q-upper')
 ax[3].plot(t,a['qvel'][:,8],label=n+' velocity')
 ax[4].plot(t,a['actuator_force'][:,2]*a['actuator_moment'][:,2,8],label=n+' actual act2 torque8')
ax[0].axhline(1,color='k',linestyle=':')
ax[2].axhline(0,color='k',linestyle=':')
for v in ax:
 for t in (1.7,1.925,2.625,2.65,2.825):v.axvline(t,color='grey',alpha=.25)
 v.grid(alpha=.2);v.legend(fontsize=8,ncol=2)
for v,s in zip(ax,('Unclipped drive','Clipped command','Roll q-upper (rad)','Roll velocity (rad/native time)','Act2 actual torque toward upper')):v.set_ylabel(s)
ax[-1].set_xlabel('Physical time (ms); lines: load failures1.7/1.925, limit failures2.625/2.65, peak2.825')
fig.suptitle('S116 saved-data driver–actual torque–limit timing; W unchanged, no simulation; units NOT_ESTABLISHED')
fig.tight_layout();fig.savefig(P/'DRIVE_COMMAND_TORQUE_LIMIT.png',dpi=150);plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(P/'SUPPLEMENT_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in [*paths,Path(__file__).resolve(),P/'DRIVE_COMMAND_TORQUE_LIMIT.png']},indent=2))
print('supplement plotted without physics')
