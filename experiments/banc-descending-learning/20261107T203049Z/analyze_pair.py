"""Independent saved-array comparison of matched DN-zero vs DN-on physical pair; no simulation."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z');PHY=OUT/'physical';W=24.446354755878005
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
names=['dn_zero','dn_on']
res={};body={};native={}
for n in names:
    p=PHY/n;r=json.loads((p/'RESULT.json').read_text());b=load(p/'BODY.npz');nt=load(p/'NATIVE.npz')['rows'];pol=load(p/'POLICY.npz')
    res[n]=r;body[n]=b;native[n]=nt
    rate=np.asarray(pol['predicted_rate'],float)
    speeds=nt[:,7:13].sum(0)/np.maximum(nt[:,1:7].sum(0),1e-15)
    res[n]|={'mean_abs_rate':float(np.abs(rate).mean()),'rate_rms':float(np.sqrt(np.mean(rate**2))),'theta_nonzero':int(np.count_nonzero(load(p/'ACTOR.npz')['theta'])),'slip_speed':speeds.tolist(),'support_mean_W':float(nt[:,1:7].sum(1).mean()),'support_min_W':float(nt[:,1:7].sum(1).min()),'nonfoot_max':float(nt[:,7].max()),'com_z_min':float(nt[:,10].min()),'com_dz':float(b['com'][-1,2]-b['com'][0,2])}
# DN command actually differs only at the two named ports
for n in names:
    c=load(PHY/n/'DN_INPUT.npz');amps=float(np.max(c['current']));res[n]['dn_amplitude_nA']=amps;res[n]['dn_ports_active']=int(np.count_nonzero(c['current']))
diff=float(np.max(np.abs(load(PHY/'dn_on'/'ACTOR.npz')['theta']-load(PHY/'dn_zero'/'ACTOR.npz')['theta'])))
assert diff==0,'Actors must be identical across pair'
for key in ('qpos','qvel','com','time'):
    if key in body['dn_zero']:
        res.setdefault('body_max_abs_diff',{})[key]=float(np.max(np.abs(body['dn_zero'][key]-body['dn_on'][key])))
res['body_max_abs_diff']['native_foot_force']=float(np.max(np.abs(native['dn_zero'][:,1:7]-native['dn_on'][:,1:7])))-0.
res['actor_identical']=diff==0
res['data_sha256']={n:{f:hashlib.sha256((PHY/n/f).read_bytes()).hexdigest() for f in ['BODY.npz','NATIVE.npz','POLICY.npz','COMMANDS.npz','SLIP.npz']} for n in names}
res['notes']='Saved arrays only, no new steps. DNp09-current is an engineering probe, not a brain-generated command or proven optogenetic equivalent. Pair isolates the descending current: same body, same Actor, same core, same checkpoint.'
with (OUT/'PHYSICAL_PAIR_COMPARISON.json').open('x') as f:json.dump(res,f,indent=2,allow_nan=False)
fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for n,ls in zip(names,['-','--']):
    b=body[n];t=b['time']-b['time'][0]
    ax[0].plot(t,b['com'][:,0],ls,label=n);ax[0].plot(t,b['com'][:,1],ls,alpha=.6);ax[0].set(xlabel='release s',ylabel='COM x/y native',title='Body COM');ax[0].legend()
    nt=native[n];tn=nt[:,0]-nt[0,0]
    ax[1].plot(tn,nt[:,1:7].sum(1)/W,ls,label=n);ax[1].set(xlabel='release s',ylabel='foot load / W',title='Foot support');ax[1].legend()
    mn=np.asarray(load(PHY/n/'POLICY.npz')['mn_rates_hz']);tm=np.arange(mn.shape[0])*.005
    ax[2].plot(tm,mn.mean(1),ls,label=n);ax[2].set(xlabel='release s',ylabel='mean MN rate Hz',title='Local MN activity');ax[2].legend()
fig.suptitle('Matched DN probe pair | identical Actor, only named descending current differs')
fig.savefig(OUT/'PHYSICAL_PAIR.png',dpi=150);plt.close(fig)
print(json.dumps({k:res[k] for k in ('actor_identical','body_max_abs_diff')},indent=2))
print(json.dumps({n:{k:res[n][k] for k in ('dn_amplitude_nA','mean_abs_rate','support_mean_W','nonfoot_max','com_z_min','slip_speed')} for n in names},indent=2))
