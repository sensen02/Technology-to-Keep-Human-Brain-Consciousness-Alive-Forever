"""Independent recorded-event K5 preclip decoder audit; no body/neural simulation."""
from pathlib import Path
import json,math,hashlib,csv,collections
import numpy as np
R=Path(__file__).resolve().parents[1]
B=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-pool-size-decoder-audit/20261102T064123Z')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False)
 inputs=[Path(__file__),B/'RUN/MAPPING.json',B/'NEURAL_RASTER.npz',B/'RUN/PREWARM_NEURAL_RECORDS.json',B/'RUN/NEURAL_RECORDS.json',R/'engine/embodied/motor_output.py'];hashes={str(p):sha(p) for p in inputs}
 dump(O/'PREREGISTRATION.json',dict(scope='Recorded-event arithmetic only, not body or neural simulation',source_sha256=hashes,formula='g_i=min(1,presentMNcount/5) for nonempty,0empty; apply to raw filteredrate beforeclip exactly as existing MotorOutput. Identicalpool gains equal, no bias or newstate.',pulse='One observed hypotheticalevent from1member ofpresentN,dt.5ms tau5ms scale100Hz; report first/secondhalfstep placement in1mswindow; puretest not injected into body.',limits='K5 is declared engineering calibration, not measured NMJphysiology. Fixedoldspikes imply no bodyfeedback; anytruebodycandidate hasnewneuralhistory.'))
 mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];n=np.array([len(r['local_indices']) for r in mapping]);gain=np.minimum(1.,n/5.);assert len(n)==90 and np.count_nonzero(n)==74
 old=np.load(B/'NEURAL_RASTER.npz')['spikes'][:2300];records=json.loads((B/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((B/'RUN/NEURAL_RECORDS.json').read_text())[:50];assert len(records)==1150
 rate=np.zeros(90);decay=math.exp(-.0005/.005);previousold=np.zeros(90);previousnew=np.zeros(90);oldnext=[];newnext=[];oldapplied=[];newapplied=[];rates=[]
 for window in range(1150):
  oldapplied.append(previousold.copy());newapplied.append(previousnew.copy())
  for sample in old[2*window:2*window+2]:
   instantaneous=np.array([sample[r['local_indices']].mean()/.0005 if r['local_indices'] else 0. for r in mapping]);rate=decay*rate+(1-decay)*instantaneous
  previousold=np.clip(rate/100,0,1);previousnew=np.clip(gain*rate/100,0,1);oldnext.append(previousold.copy());newnext.append(previousnew.copy());rates.append(rate.copy())
  actual=records[window];assert np.array_equal(oldapplied[-1],actual['applied_command']) and np.array_equal(previousold,actual['decoded_next_command']) and np.array_equal(rate,actual['decoder']['rates_hz'])
 oldapplied=np.array(oldapplied);newapplied=np.array(newapplied);rates=np.array(rates);oldnext=np.array(oldnext);newnext=np.array(newnext)
 groups={};rows=[]
 for i,row in enumerate(mapping):
  key=tuple(row['local_indices'])
  if key:
   if key in groups:assert gain[i]==gain[groups[key]]
   else:groups[key]=i
  raw_first=0. if not n[i] else (1-decay)/(.0005*n[i])*decay/100
  raw_second=0. if not n[i] else (1-decay)/(.0005*n[i])/100
  rows.append(dict(actuator=row['actuator'],presentMN=int(n[i]),gain=float(gain[i]),status=row['status'],old_singlepulse_firsthalf_cmd=min(raw_first,1.),new_singlepulse_firsthalf_cmd=min(gain[i]*raw_first,1.),old_singlepulse_secondhalf_cmd=min(raw_second,1.),new_singlepulse_secondhalf_cmd=min(gain[i]*raw_second,1.)))
 with (O/'POOL_GAIN_AND_PULSE.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 clip_post=gain*oldnext;different=int(np.count_nonzero(clip_post!=newnext));assert different>0
 np.savez_compressed(O/'RECORDED_EVENT_DECODER_COMPARISON.npz',gains=gain,present_counts=n,old_filtered_rates_hz=rates,old_next=oldnext,K5_next=newnext,old_applied=oldapplied,K5_applied=newapplied,wrong_postclip_scaling=clip_post)
 result=dict(passed=True,source_sha256=hashes,recorded_ticks=2300,recorded_windows=1150,present_pool_counts=dict(collections.Counter(map(int,n))),unique_present_pools=len(groups),mapped_actuators=74,unresolved16gain0=int((gain==0).sum()),attenuated_mapped_actuators=int(((gain<1)&(n>0)).sum()),preclip_vs_postclip_different_elements=different,old_actual_record_reconstructed_exact=True,windows={name:dict(old_applied_saturated=int((oldapplied[sl]>=1).sum()),K5_applied_saturated=int((newapplied[sl]>=1).sum()),total_elements=int(oldapplied[sl].size)) for name,sl in [('prewarm',slice(0,1100)),('release50',slice(1100,1150))]},small2MN_onepulse=dict(old_firsthalf_cmd=min((1-decay)/(.0005*2)*decay/100,1),new_firsthalf_cmd=min(.4*(1-decay)/(.0005*2)*decay/100,1)),calls=dict(model_compile=0,network=0,body_step=0,neural_step=0,training=0),limits='Recorded-event replay oforiginalneutralhistory. Newcandidatephysicalrunmusthaveitsownhistory; saturationcounts aredescriptive,not supportimprovement.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
