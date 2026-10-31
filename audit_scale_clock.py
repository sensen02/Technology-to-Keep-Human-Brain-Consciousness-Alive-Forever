"""S158 offline scale-clock audit. No MuJoCo or neural imports/calls."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,xml.etree.ElementTree as ET
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ARMS={'S154':Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z'),'S157':Path('/home/sensen/Desktop/engine修复验收/s157-foreleg-gain/20261024T221335645450Z')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 p=Path('/home/sensen/Desktop/engine修复验收/s158-scale-clock')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p.mkdir(parents=True)
 inputs=[XML,Path(__file__),*[r/n for r in ARMS.values() for n in ['SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py','init80ms/TRAJECTORY.npz','init80ms/NATIVE_MODEL_ARRAYS.npz','init80ms/MODEL_PROVENANCE.json','init80ms/RESULT.json']]]
 hashes={str(f):sha(f) for f in inputs};(p/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2));(p/'PREANALYSIS.json').write_text(json.dumps(dict(scope='saved arrays/XML/source only; no model or neural execution',events='load peak, COMaz extrema, first postpeak load below1 and .8, maximum penetration',unit_rule='do not infer SI from geometry; source attribution separate'),indent=2))
 tree=ET.parse(XML).getroot();rows=[]
 for node in tree.iter():
  for k,v in node.attrib.items():
   if k in ['scale','pos','mass','diaginertia','gravity','timestep','solref','solimp','solreflimit','solimplimit','dynprm','gainprm','biasprm','lengthrange','stiffness','damping','armature']:
    rows.append(dict(source=str(XML),tag=node.tag,name=node.get('name',node.get('class','')),field=k,value=v,status='XML explicit numeric; absolute physical calibration NOT_ESTABLISHED'))
 for name,r in ARMS.items():
  m=load(r/'init80ms/NATIVE_MODEL_ARRAYS.npz');a=load(r/'init80ms/TRAJECTORY.npz');assert np.allclose(a['time'],np.arange(len(a['time']))*25e-6,rtol=0,atol=1e-14);assert np.all(m['actuator_dynprm'][:,0]==.001) and np.all(m['actuator_dynprm'][:,1]==.0004)
  for k in ['body_mass','body_inertia','actuator_lengthrange','actuator_gainprm','actuator_biasprm','actuator_dynprm']:
   rows.append(dict(source=str(r/'init80ms/NATIVE_MODEL_ARRAYS.npz'),tag=name,name=k,field='min/max',value=json.dumps([float(m[k].min()),float(m[k].max())]),status='native compiled snapshot numeric'))
 with (p/'FIELD_SOURCES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,axs=plt.subplots(2,2,figsize=(12,9));summary={};events=[];series=[]
 for name,r in ARMS.items():
  a=load(r/'init80ms/TRAJECTORY.npz');t=a['time'];loadratio=a['foot_force'][:,:,2].sum(1)/abs(a['gravity_force_world'][:,2]);penetration=np.array([max(0.,-float(a['contact_dist'][i,:int(a['contact_ncon'][i])].min())) for i in range(len(t))]);peak=int(loadratio.argmax());indices={'load_peak':peak,'COMaz_peak':int(a['com_acceleration_world'][:,2].argmax()),'COMaz_min':int(a['com_acceleration_world'][:,2].argmin()),'max_penetration':int(penetration.argmax()),'first_fail':len(t)-1};below=np.flatnonzero((np.arange(len(t))>peak)&(loadratio<1));indices['first_postpeak_load_below1']=int(below[0]) if len(below) else None
  for event,i in indices.items():
   if i is not None:events.append(dict(arm=name,event=event,index=i,time_s=float(t[i]),body_steps=i,activation_tau_units=float(t[i]/.001),deactivation_tau_units=float(t[i]/.0004),sensory_period_units=float(t[i]/.005),neural_tick_units=float(t[i]/.0005),contact_solref_nominal_units=float(t[i]/.002),load_ratio=float(loadratio[i]),penetration=float(penetration[i]),COMaz=float(a['com_acceleration_world'][i,2])))
  summary[name]=dict(samples=len(t),mass_total=float(a['body_mass'][0].sum()),gravity_norm=float(abs(a['gravity_force_world'][0,2])/a['body_mass'][0].sum()),max_penetration=float(penetration.max()),COMaz_minmax=[float(a['com_acceleration_world'][:,2].min()),float(a['com_acceleration_world'][:,2].max())],max_abs_muscle_force=float(abs(a['actuator_force']).max()),tendon_length_minmax=[float(a['actuator_length'].min()),float(a['actuator_length'].max())],tendon_velocity_minmax=[float(a['actuator_velocity'].min()),float(a['actuator_velocity'].max())])
  axs[0,0].plot(t*1000,loadratio,label=name);axs[0,1].plot(t*1000,penetration,label=name);axs[1,0].plot(t*1000,a['com_acceleration_world'][:,2],label=name)
  for i in range(len(t)):series.append(dict(arm=name,index=i,time_s=float(t[i]),load_ratio=float(loadratio[i]),penetration=float(penetration[i]),COMaz=float(a['com_acceleration_world'][i,2])))
 for filename,data in [('EVENT_TIMES.csv',events),('SAVED_TIME_SERIES.csv',series)]:
  with (p/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 names=['body dt','deact tau','neural tick','act tau','contact nominal','failure','sensory refresh','PRE neural only'];vals=[.025,.4,.5,1,2,2.1,5,80];axs[1,1].barh(names,vals);axs[1,1].set_xscale('log');axs[1,1].set_xlabel('Declared simulation time (ms); PRE has no body steps')
 for ax,title in zip(axs.flat,['Foot load / weight','Maximum penetration (native length)','Mass-weighted COM az (native length/time²)','Clock/time-scale comparison']):ax.set_title(title)
 for ax in axs.flat[:3]:ax.set_xlabel('Physical time (ms)');ax.legend()
 axs[0,0].axhline(.8,color='r');fig.tight_layout();fig.savefig(p/'SCALE_CLOCK.png',dpi=150)
 (p/'SUMMARY.json').write_text(json.dumps(dict(arms=summary,events=events,calls=dict(model_load=0,forward=0,step=0,neural_ticks=0),nominal_time_scales_s=dict(body_dt=.000025,neural_tick=.0005,sensory=.005,act_tau=.001,deact_tau=.0004,contact_solref=.002,PRE_neural_only=.08),contact_caveat='positive solref timeconst/dampratio mode; not scalar stiffness. solimp and coupled constraint effective inertia matter; do not infer universal eigenperiod from nominal .002',biological_calibration=False),indent=2))
 assert hashes=={str(f):sha(f) for f in inputs};(p/'INPUT_HASHES_AFTER.json').write_text(json.dumps(hashes,indent=2));print(p)
if __name__=='__main__':main()
