"""Reuse saved-data comparison, add actual vs encoded observations; no simulation."""
import inspect,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import compare_fast_sensory_only as parent
from run_fixed_initial_sensory import O,BASE,sha,dump

def main():
 src=inspect.getsource(parent.main).replace('list(range(0,500,2))','list(range(0,500,10))').replace("'sensory1ms'","'fixed_initial'").replace('sensory1ms','fixed_initial').replace('actual1ms refreshed current sum','fixed initial encoded current').replace('Sensory1ms ONLY vs saved5ms / neural500us command5ms / NOT standing certification','Fixed initial sensory vs dynamic5ms / NOT full feedback or standing certification').replace('refresh_count=250','refresh_count=50').replace("Path('/home/sensen/Desktop/S2_SENSORY1MS_COMMAND5MS.png')","Path('/home/sensen/Desktop/S2_FIXED_INITIAL_SENSORY_BODY.png')")
 ns=dict(vars(parent),O=O,BASE=BASE);exec(compile(src,__file__,'exec'),ns);ns['main']()
 log=json.loads((O/'OBSERVATION_ACTUAL_VS_ENCODER.json').read_text());ref=json.loads((O/'SENSORY_REFRESH.json').read_text());assert len(log)==len(ref)==50
 assert all(r['encoder_observation']==log[0]['encoder_observation'] for r in log);assert any(r['actual_observation']!=r['encoder_observation'] for r in log[1:]);assert all(np.array_equal(r['current'],ref[0]['current']) for r in ref)
 early=json.loads((O/'EARLY_LEG_RECRUITMENT.json').read_text());fig,axs=plt.subplots(4,1,figsize=(16,15));time=np.array([r['time'] for r in log])
 for leg in ['lf','lm','lh','rf','rm','rh']:
  keys=list(log[0]['actual_observation'][leg]);key=next(k for k in keys if 'tibia_joint_dev' in k);axs[0].plot(time,[r['actual_observation'][leg][key] for r in log],label=leg+' actual');axs[0].plot(time,[r['encoder_observation'][leg][key] for r in log],'--',alpha=.5)
 axs[0].set_title('Fixed arm: actual tibia deviation vs encoder observation (dashed frozen)')
 old=np.load(BASE/'normal/INTERVALS.npz');new=np.load(O/'INTERVALS.npz');axs[1].plot(np.arange(50)*.005,old['sensory_current_nA'].sum(1),label='dynamic current sum');axs[1].plot(time,[sum(r['current']) for r in ref],label='fixed current sum');axs[1].set_title('Actual network encoded input sum / nA, no added bias')
 for label,z in [('dynamic',np.load(BASE/'normal/BODY.npz')),('fixed',np.load(O/'BODY.npz'))]:
  axs[2].plot(z['time'],z['command'].max(1),label=label+' maximum command');axs[3].plot(z['time'],z['thorax_position'][:,2],label=label+' thorax')
 axs[2].set_title('Recruitment readout: applied maximum command /5ms');axs[3].set_title('Same saved physical trajectory: thorax height / model units')
 for ax in axs:ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Fixed initial sensory diagnostic / actual observation != encoded observation / NOT feedback success');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'ACTUAL_ENCODER_RESPONSE.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_FIXED_INITIAL_SENSORY.png').write_bytes((O/'ACTUAL_ENCODER_RESPONSE.png').read_bytes());dump(O/'FIXED_MODE_VERIFICATION.json',dict(exit_code=0,actual_observation_changes=True,encoder_observation_constant=True,current_constant_exact=True,refresh_count=50,model_forward_body_neural_calls=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
