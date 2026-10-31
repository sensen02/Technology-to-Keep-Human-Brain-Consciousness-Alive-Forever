"""Saved only sensory x2 analysis, no physical or neural calls."""
import inspect,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import compare_fast_sensory_only as parent
from run_sensory_current_double import O,BASE,sha,dump

def main():
 src=inspect.getsource(parent.main).replace('list(range(0,500,2))','list(range(0,500,10))').replace("'sensory1ms'","'sensory_x2'").replace('sensory1ms','sensory_x2').replace('actual1ms refreshed current sum','actual x2 current sum').replace('Sensory1ms ONLY vs saved5ms / neural500us command5ms / NOT standing certification','Sensory current x2 vs x1 /5ms input and command / NOT standing certification').replace('refresh_count=250','refresh_count=50').replace("Path('/home/sensen/Desktop/S2_SENSORY1MS_COMMAND5MS.png')","Path('/home/sensen/Desktop/S2_SENSORY_CURRENT_DOUBLE_BODY.png')").replace("np.testing.assert_array_equal(refresh[0]['current'],old[1]['sensory_current_nA'][0])","np.testing.assert_array_equal(refresh[0]['current'],old[1]['sensory_current_nA'][0]*2)")
 ns=dict(vars(parent),O=O,BASE=BASE);exec(compile(src,__file__,'exec'),ns);ns['main']()
 currents=json.loads((O/'CURRENT_RAW_REQUESTED_EFFECTIVE.json').read_text());assert len(currents)==50
 for r in currents:
  np.testing.assert_array_equal(np.array(r['raw_encoder_current'])*2,r['requested_current']);np.testing.assert_array_equal(r['requested_current'],r['effective_input_current'])
 mapping=json.loads((O/'MAPPING.json').read_text());locals=sorted({i for r in mapping['rows'] for i in r['local_indices']});details={};fig,axs=plt.subplots(3,2,figsize=(15,12))
 for label,p in [('x1',BASE/'normal'),('x2',O)]:
  iv=np.load(p/'INTERVALS.npz');z=np.load(p/'BODY.npz');rec=json.loads((p/'RECORDS.json').read_text());cur=iv['sensory_current_nA'];details[label]=dict(input_min=float(cur.min()),input_max=float(cur.max()),total_network_spikes=int(iv['spikes'].sum()),unique_motor_node_spikes=int(iv['spikes'][:,locals].sum()),decoder_saturated_windows_by_actuator=[dict(actuator=r['actuator'],windows=sum(bool(s['decoder']['saturated'][j]) for s in rec['records'])) for j,r in enumerate(mapping['rows'])],command_peak_by_actuator=z['command'].max(0),activation_peak_by_actuator=z['activation'].max(0))
  axs[0,0].plot(np.arange(50)*.005,cur.sum(1),label=label);axs[0,1].plot(np.arange(1,51)*.005,iv['spikes'][:,locals].sum(1),label=label);axs[1,0].plot(z['time'],z['command'].max(1),label=label);axs[1,1].plot(z['time'],z['thorax_position'][:,2],label=label);axs[2,0].plot(z['time'],z['com_velocity'][:,2],label=label);axs[2,1].plot(z['time'],z['nonfoot_floor_count'],label=label)
 for ax,title in zip(axs.flat,['Sensory effective current sum /nA','Unique mapped motor-node events /5ms','Maximum applied command','Thorax height /model units','COM vertical velocity','Nonfoot plane contact count']):ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.set_xlabel('physical seconds')
 fig.suptitle('Dynamic sensory current x2 / initial AND subsequent drive changed / NOT physiological calibration');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'CURRENT_MOTOR_BODY_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_SENSORY_CURRENT_DOUBLE.png').write_bytes((O/'CURRENT_MOTOR_BODY_COMPARISON.png').read_bytes());dump(O/'CURRENT_MOTOR_SATURATION.json',details);dump(O/'SCALING_VERIFICATION.json',dict(exit_code=0,requests_exactly_twice_raw_all50=True,effective_equals_requested_all50=True,subsequent_clip=False,analysis_model_forward_step_neural_calls=0));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps({k:{a:b for a,b in v.items() if a in ['input_min','input_max','total_network_spikes','unique_motor_node_spikes']} for k,v in details.items()}))
if __name__=='__main__':main()
