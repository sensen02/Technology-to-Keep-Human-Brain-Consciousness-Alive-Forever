"""Saved-only node6047 event->decoder->write->activation audit."""
from pathlib import Path
import json,sys,inspect,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.embodied.motor_output import MotorOutput
P=Path(__file__).parent
OLD=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
NEW=Path('/home/sensen/Desktop/engine修复验收/s3-motor-threshold-minus2/20261027T205000Z')
TARGET=Path('/home/sensen/Desktop/engine修复验收/s3-silent-motor-drive/20261027T204100Z/TARGETS.json')
O=Path('/home/sensen/Desktop/engine修复验收/s3-node6047-chain/20261027T210100Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def first(a,b,t):
 mask=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b;ix=np.flatnonzero(mask);return dict(index=int(ix[0]),time_s=float(t[ix[0]])) if len(ix) else None

def main():
 O.mkdir(parents=True,exist_ok=False);sources=[TARGET,NEW/'MAPPING.json',P/'engine/embodied/motor_output.py',P/'run_neural_muscle_closure.py',Path(__file__)]
 for root,prefix in [(OLD,'normal/'),(NEW,'')]:
  man=json.loads((root/'MANIFEST_SHA256.json').read_text())
  for name in ['BODY.npz','INTERVALS.npz','RECORDS.json']:
   path=root/(prefix+name);assert sha(path)==man[prefix+name];sources.append(path)
 sources.append(NEW/'ALL_NODE_TICK_EVENTS.npz');assert sha(sources[-1])==json.loads((NEW/'MANIFEST_SHA256.json').read_text())['ALL_NODE_TICK_EVENTS.npz']
 hashes={str(p):sha(p) for p in sources};dump(O/'PREREGISTRATION.json',dict(input_code_sha256=hashes,scope='saved-only interface chain; no neural/body',node=6047,decoder_validation='actual MotorOutput.consume reconstructed probe ticks, assert exact all actuator output vs decoded_next_controls and record rates',time_rules='interval[start,end] emits tick at end; decode after body interval then pending written at next start; first saved post-write one12.5us step later',missing='body ctrl at exact write boundary not saved; previous sample belongs prior step, infer write from source and next saved command'))
 mapping=json.loads((NEW/'MAPPING.json').read_text());target=json.loads(TARGET.read_text());rows=[r for r in target['source_11_muscles'] if 6047 in r['local_indices']];assert rows;dump(O/'SHARED_POOL_RELATIONS.json',[dict(actuator=r['actuator'],actuator_index=r['actuator_index'],members=r['local_indices'],member_count=len(r['local_indices'])) for r in rows]);data={};traces={};spikes={}
 for label,path in [('historical',OLD/'normal'),('minus2mV',NEW)]:
  packed=np.load(path/'BODY.npz');z={k:packed[k] for k in packed.files};iv=np.load(path/'INTERVALS.npz');rec=json.loads((path/'RECORDS.json').read_text());sp=np.zeros((500,mapping['n_neurons']),bool)
  for k,r in enumerate(rec['records']):
   for j,col in r['probe_raw_spike_events_step_column']:sp[k*10+j,rec['probe_local_indices'][col]]=True
  motor={i for r in mapping['rows'] for i in r['local_indices']};assert motor.issubset(set(rec['probe_local_indices']))
  if label=='minus2mV':np.testing.assert_array_equal(sp[:,sorted(motor)],np.load(NEW/'ALL_NODE_TICK_EVENTS.npz')['spikes'][:,sorted(motor)])
  dec=MotorOutput(mapping);trace=[]
  for k in range(50):
   ctrl,rr=dec.consume(sp[k*10:(k+1)*10],spike_dt_s=.0005,start_s=k*.005,end_s=(k+1)*.005)
   np.testing.assert_array_equal(ctrl,iv['decoded_next_controls'][k]);np.testing.assert_array_equal(rr['rates_hz'],rec['records'][k]['decoder']['rates_hz'])
   for r in rows:
    ai=r['actuator_index'];idx=r['local_indices'];trace.append(dict(window=k,start_s=k*.005,end_s=(k+1)*.005,actuator=r['actuator'],pool_members=len(idx),pool_events=int(sp[k*10:(k+1)*10,idx].sum()),tick_pool_rate_hz=sp[k*10:(k+1)*10,idx].mean(1)/.0005,filtered_rate_hz=rr['rates_hz'][ai],raw_command=rr['rates_hz'][ai]/dec.config.rate_for_full_control_hz,clipped_command=float(ctrl[ai]),saved_applied_next_interval=float(iv['commands'][k+1,ai]) if k<49 else None))
   if k<49:np.testing.assert_array_equal(ctrl,iv['commands'][k+1])
  data[label]=z;traces[label]=trace;spikes[label]=sp
 dump(O/'DECODER_TRACE.json',traces)
 old=data['historical'];new=data['minus2mV'];ais=[r['actuator_index'] for r in rows];diffs={k:first(old[k],new[k],old['time']) for k in ['qpos','qvel','activation','command']};diffs['target_command']=first(old['command'][:,ais],new['command'][:,ais],old['time']);diffs['target_activation']=first(old['activation'][:,ais],new['activation'][:,ais],old['time']);diffs['target_pool_tick']=first(spikes['historical'][:,sorted({i for r in rows for i in r['local_indices']})],spikes['minus2mV'][:,sorted({i for r in rows for i in r['local_indices']})],np.arange(1,501)*.0005)
 oldiv=np.load(OLD/'normal/INTERVALS.npz');newiv=np.load(NEW/'INTERVALS.npz');diffs['full_network_window_counts']=first(oldiv['spikes'],newiv['spikes'],np.arange(1,51)*.005);oldrec=json.loads((OLD/'normal/RECORDS.json').read_text());newrec=json.loads((NEW/'RECORDS.json').read_text());probe=oldrec['probe_local_indices'];assert probe==newrec['probe_local_indices'];diffs['observed_probe_tick_events']=first(spikes['historical'][:,probe],spikes['minus2mV'][:,probe],np.arange(1,501)*.0005);dump(O/'FIRST_DIFFERENCES.json',diffs)
 mask=old['time']<=.04+1e-12;np.savez_compressed(O/'SHORT_WINDOW_CHAIN.npz',time_s=old['time'][mask],historical_command=old['command'][mask][:,ais],new_command=new['command'][mask][:,ais],historical_activation=old['activation'][mask][:,ais],new_activation=new['activation'][mask][:,ais],event_end_s=np.arange(1,81)*.0005,historical_target_pool_ticks=spikes['historical'][:80,sorted({i for r in rows for i in r['local_indices']})],new_target_pool_ticks=spikes['minus2mV'][:80,sorted({i for r in rows for i in r['local_indices']})])
 fig,axs=plt.subplots(5,1,figsize=(15,14));dt=.0005;decay=np.exp(-dt/.005)
 for label,style in [('historical','--'),('minus2mV','-')]:
  for r in rows:
   idx=r['local_indices'];ai=r['actuator_index'];ss=spikes[label][:80,idx];tm=np.arange(1,81)*.5;axs[0].step(tm,ss.sum(1),where='post',ls=style,label=label+' pool');axs[1].step(tm,ss.mean(1)/dt,where='post',ls=style,label=label);tt=[v for v in traces[label] if v['actuator']==r['actuator'] and v['end_s']<=.04+1e-12];axs[2].step([v['end_s']*1000 for v in tt],[v['clipped_command'] for v in tt],where='post',ls=style,label=label);zz=data[label];axs[3].plot(zz['time'][mask]*1000,zz['command'][mask,ai],ls=style,label=label);axs[4].plot(zz['time'][mask]*1000,zz['activation'][mask,ai],ls=style,label=label)
 for ax,title in zip(axs,['Pool events by neural tick','Mean pool rate Hz','Decoded clipped pending command (valid at window end)','Saved applied command (first new sample20.0125ms)','Actual activation']):
  ax.set_title(title);ax.axvline(18,color='gray',ls=':');ax.axvline(20,color='red',ls=':');ax.legend();ax.grid(alpha=.2);ax.set_xlabel('physical / neural ms')
 fig.suptitle('Node6047 event -> pool mean/filter -> pending -> applied -> activation / no new dynamics');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'NODE6047_CHAIN.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S3_NODE6047_CHAIN.png').write_bytes((O/'NODE6047_CHAIN.png').read_bytes());assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,decoder_consume_calls=100,models=0,forward=0,body_step=0,neural_step=0,failures=[],missing=['historical nonprobe full-network per-tick raster','exact-boundary ctrl sample before first step; source order documented']));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(diffs));print(json.dumps([v for v in traces['minus2mV'] if v['window']==3],default=lambda x:x.tolist()));print('decay',decay)
if __name__=='__main__':main()
