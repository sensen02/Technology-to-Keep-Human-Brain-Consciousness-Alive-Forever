"""Saved-state derived-model component and ground force comparison."""
import inspect,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import compare_postclip_half_command as parent
from run_double_muscle_force import O,BASE,c,sha,dump,scale_model,restore,components

def main():
 # Reuse established all-last50ms world-force reconstruction; scale compiled model only for new arm.
 src=inspect.getsource(parent.main).replace("'scale0p5'","'force_x2'").replace('scale0p5','force_x2').replace('scale1','force_x1').replace('Fresh half postclip command closure vs saved baseline / NOT stable-support certification','Muscle active AND bias capacity x2 / NOT calibrated force or standing certification').replace("Path('/home/sensen/Desktop/S2_HALF_COMMAND_COMPARISON.png')","Path('/home/sensen/Desktop/S2_MUSCLE_FORCE_DOUBLE_BODY.png')")
 src=src.replace("z=np.load(p/'BODY.npz');iv=", "\n  if label=='force_x2':scale_model(m,2)\n  z=np.load(p/'BODY.npz');iv=")
 ns=dict(vars(parent),O=O,BASE=BASE,scale_model=scale_model);exec(compile(src,__file__,'exec'),ns);ns['main']()
 ledger={};calls=0;models=0;fig,axs=plt.subplots(3,2,figsize=(15,12))
 mapping=json.loads((O/'MAPPING.json').read_text());motor=sorted({i for r in mapping['rows'] for i in r['local_indices']})
 for label,p in [('force_x1',BASE/'normal'),('force_x2',O)]:
  m=c.cp.load_model();models+=1
  if label=='force_x2':scale_model(m,2)
  z=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');rows=[]
  for idx in range(0,20001,400):
   d=restore(m,z,idx);calls+=1;a,b=components(m,d);assert np.allclose(a+b,d.actuator_force,atol=1e-10,rtol=1e-12);rows.append(dict(time=float(d.time),active=a,bias=b,total=d.actuator_force.copy(),command=d.ctrl.copy(),activation=d.act.copy()))
  ledger[label]=rows;ts=[r['time'] for r in rows];axs[0,0].plot(ts,[sum(r['active']) for r in rows],label=label);axs[0,1].plot(ts,[sum(r['bias']) for r in rows],label=label);axs[1,0].plot(np.arange(1,51)*.005,iv['spikes'][:,motor].sum(1),label=label);axs[1,1].plot(z['time'],z['thorax_position'][:,2],label=label);axs[2,0].plot(z['time'],z['com_velocity'][:,2],label=label);axs[2,1].plot(z['time'],z['nonfoot_floor_count'],label=label)
 for ax,title in zip(axs.flat,['Signed active model-force sum / not support','Signed bias model-force sum / not support','Unique mapped motor events /5ms','Thorax height / model units','COM vertical velocity','Nonfoot plane contact count']):ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Effective muscle force capacity x2 (active AND bias) / real feedback / not physiological calibration');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'FORCE_RECRUITMENT_BODY.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_MUSCLE_FORCE_DOUBLE.png').write_bytes((O/'FORCE_RECRUITMENT_BODY.png').read_bytes());dump(O/'COMPONENTS_90_AT_51_STATES.json',dict(actuator_names=[r['actuator'] for r in mapping['rows']],states=ledger));dump(O/'COMPONENT_ANALYSIS_EXECUTION.json',dict(exit_code=0,model_load=models,mj_forward=calls,mj_step=0,neural_step=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
