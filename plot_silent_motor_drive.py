"""Plot recorded replay only, no new neural execution."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_silent_motor_drive import O,dump

def main():
 r=json.loads((O/'TARGET_DRIVE_TIME_SERIES.json').read_text());tar=json.loads((O/'TARGETS.json').read_text());nodes=tar['unique_local_nodes'];t=np.array([v['end_ms'] for v in r]);legs=sorted({v['leg'] for v in tar['source_11_muscles']});fig,axs=plt.subplots(len(legs),3,figsize=(17,3.5*len(legs)),squeeze=False)
 for row,leg in enumerate(legs):
  ns=sorted({i for v in tar['source_11_muscles'] if v['leg']==leg for i in v['local_indices']})
  for node in ns:
   j=nodes.index(node);v=np.array([s['v_trial'][j] for s in r]);thr=np.array([s['threshold_mV'] for s in r]);axs[row,0].plot(t,v,label=str(node));axs[row,1].plot(t,thr-v,label=str(node));axs[row,2].plot(t,[s['exc_current_at_v_trial'][j]+s['inh_current_at_v_trial'][j] for s in r],label=str(node)+' synaptic net')
  axs[row,0].axhline(r[0]['threshold_mV'],color='k',ls='--');
  for ax,title in zip(axs[row],[leg+' V trial /mV',leg+' threshold gap /mV',leg+' synaptic I at trial V /nA']):ax.set_title(title);ax.legend(fontsize=6);ax.grid(alpha=.2);ax.axvline(38.7375,color='red',ls=':');ax.set_xlabel('replayed neural interval end ms')
 fig.suptitle('Fixed historical-input neural replay / counts and probe ticks match / full-network timing reference absent');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(O/'SILENT_MOTOR_DRIVE.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S3_SILENT_MOTOR_DRIVE.png').write_bytes((O/'SILENT_MOTOR_DRIVE.png').read_bytes());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());dump(O/'PLOT_EXECUTION.json',dict(exit_code=0,neural_steps=0,body_model_load=0,mj_forward=0,mj_step=0));print('plot saved')
if __name__=='__main__':main()
