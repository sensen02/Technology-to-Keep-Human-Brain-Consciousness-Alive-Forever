"""Offline S283 checks and outcome metrics. No simulator imports."""
from pathlib import Path
import argparse,json,csv,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REF=Path('/home/sensen/Desktop/engine修复验收/s277-silence-attribution/20261027T030000000000Z')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);d=p.parse_args().output;pre=json.loads((d/'PREREG.json').read_text());meta=json.loads((d/'MODEL_METADATA.json').read_text());checks=[];metrics={}
 def check(n,x,**v):checks.append(dict(name=n,passed=bool(x),**v))
 for path,h in pre['source_hashes'].items():check('hash '+path,hashlib.sha256(Path(path).read_bytes()).hexdigest()==h)
 pars={a:json.loads((d/a/'PARAMETERS.json').read_text()) for a in pre['arms']};check('only tau differs',[k for k in pars['baseline'] if pars['baseline'][k]!=pars['tau_equal'][k]]==['inh_tau_ms']);check('tau equal',pars['tau_equal']['inh_tau_ms']==pars['tau_equal']['exc_tau_ms']==3.)
 fig,ax=plt.subplots(4,2,figsize=(15,15));target=meta['target_index'];floor=meta['floor'];gl={int(k):v for k,v in meta['geom_leg'].items()};legs=('lf','lm','lh','rf','rm','rh')
 for arm in pre['arms']:
  a=np.load(d/arm/'TRAJECTORY.npz');n=np.load(d/arm/'NEURAL.npz');ctrl=np.load(d/arm/'CTRL.npz')['ctrl'];rows=list(csv.DictReader((d/arm/'STATE.csv').open()));t=a['time']*1000
  check(arm+' full shape',a['qpos'].shape==(4001,49) and ctrl.shape==(4000,90) and n['spikes'].shape[0]==800);check(arm+' time',np.allclose(t,np.arange(4001)*.1,rtol=0,atol=1e-8));check(arm+' no external loads',not a['external_applied'].any() and not a['qfrc_applied'].any());check(arm+' command applied',np.array_equal(ctrl,a['ctrl'][1:]))
  raw=n['raw_rates_hz'].reshape(800,90);r=np.zeros(90);filtered=[]
  for k,sample in enumerate(raw):r=np.exp(-.0005/.005)*r+(1-np.exp(-.0005/.005))*sample;filtered.append(r.copy())
  filtered=np.array(filtered);check(arm+' rate reconstruction',np.allclose(filtered[1::2],n['rates_hz'],rtol=0,atol=1e-10))
  if arm=='baseline':
   ref=np.load(REF/'CTRL_S277.npz',allow_pickle=True)['ctrl'];check('baseline CTRL exact',np.array_equal(ctrl[:2000],ref));rr=list(csv.DictReader((REF/'STATE_S277.csv').open()))
   for k in pre['gate_columns']:check('baseline '+k+' exact',all(float(rows[i][k])==float(rr[i][k]) for i in range(200)))
  contacts=json.loads((d/arm/'CONTACTS.json').read_text());ff=np.zeros((4001,6,3));body=np.zeros(4001,int)
  for i,cs in enumerate(contacts):
   for c in cs:
    g1,g2,dist=int(c[0]),int(c[1]),c[2]
    if floor not in (g1,g2):continue
    other=g2 if g1==floor else g1;force=(1 if g1==floor else -1)*np.array(c[6:15]).reshape(3,3).T@np.array(c[15:18])
    if other in gl:ff[i,legs.index(gl[other])]+=force
    elif dist<0:body[i]+=1
  check(arm+' raw force reconstruction',np.allclose(ff,a['foot_force_world'],rtol=0,atol=2e-12));check(arm+' body contact reconstruction',np.array_equal(body,a['body_penetrating_contacts']))
  def first(mask):ix=np.flatnonzero(mask);return None if not len(ix) else float(t[ix[0]])
  ts=n['time_ms'][n['spikes'].any(1)];q=a['qpos'];orient=2*np.arccos(np.clip(np.abs(np.sum(q[:,3:7]*q[0,3:7],axis=1)),0,1));disp=np.linalg.norm(q[:,:2]-q[0,:2],axis=1)
  feet=[]
  for leg in legs:
   ids=[i for i,b in enumerate(meta['body_names']) if str(b).lower().startswith(leg) and 'tars' in str(b).lower()];feet.append(ids[-1] if ids else -1)
  slip={leg:float(np.linalg.norm(a['body_xpos'][:,bid,:2]-a['body_xpos'][0,bid,:2],axis=1).max()) if bid>=0 else None for leg,bid in zip(legs,feet)}
  metrics[arm]={'target_spike_count0_200':int(n['spikes'][n['time_ms']<=200].sum()),'target_spike_count200_400':int(n['spikes'][n['time_ms']>200].sum()),'target_last_spike_ms':float(ts[-1]) if len(ts) else None,'target_max_interspike_gap_ms':float(np.diff(ts).max()) if len(ts)>1 else None,'target_silence_to400_ms':float(400-ts[-1]) if len(ts) else 400.,'target_spike_times_ms':ts.tolist(),'first_body_penetration_ms':first(body>0),'legacy_first_body_contact_ms':next((float(r['t_ms']) for r in rows if int(r['n_body_ground'])>0),None),'COM_dz400_native':float(a['com'][-1,2]-a['com'][0,2]),'root_dz400_native':float(q[-1,2]-q[0,2]),'root_horizontal400_native':float(disp[-1]),'orientation_max_rad':float(orient.max()),'joint_limit_max_rad':float(a['max_limit'].max()),'foot_force_z400_native':ff[-1,:,2].tolist(),'foot_load_min_native':float(ff[:,:,2].sum(1).min()),'foot_xy_displacement_max_native':slip,'slip_caveat':'absolute foot XY displacement, not stance-only slip; no walking inference','support_claim':'NOT_ESTABLISHED'}
  ax[0,0].plot(t,a['com'][:,2]-a['com'][0,2],label=arm);ax[0,1].plot(t,ff[:,:,2].sum(1),label=arm);ax[1,0].plot(t,body,label=arm);ax[1,1].plot(t,orient,label=arm);ax[2,0].plot(n['time_ms'],n['ge'][:,0],label=arm+' gE');ax[2,0].plot(n['time_ms'],n['gi'][:,0],ls='--',label=arm+' gI');ax[2,1].scatter(ts,np.full(len(ts),0 if arm=='baseline' else 1),s=8,label=arm);ax[3,0].plot(np.arange(1,401),n['rates_hz'][:,target],label=arm);ax[3,1].plot(t,a['max_limit'],label=arm)
 for panel,title in zip(ax.flat,['COM dz native','World total foot Fz native','Body penetrating contacts','Root orientation change rad','Target conductance uS','Target actual spikes','Target filtered rate Hz','Joint limit excess rad']):panel.set_title(title);panel.grid(alpha=.3);panel.legend(fontsize=7);panel.set_xlim(0,400)
 fig.suptitle('S283 uncalibrated all_empty tau sensitivity | original 0.1ms grid, 400ms\nNot annotation network evidence; no standing/walking claim');fig.tight_layout();fig.savefig(d/'OUTCOMES.png',dpi=140);plt.close(fig)
 check('no simulator imported','mujoco' not in sys.modules);result={'passed':all(c['passed'] for c in checks),'checks':checks,'metrics':metrics,'scope':'data consistency, not support success','new_simulation_calls':0};(d/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'failed':[c['name'] for c in checks if not c['passed']],'metrics':metrics},ensure_ascii=False,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
