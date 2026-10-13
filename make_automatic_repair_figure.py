from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.embodied.closure_bundle import validate_bundle
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/neural_muscle_closure'
bundle=validate_bundle(OUT);p=Path(bundle['path']); d=json.loads((p/'closure_results.json').read_text())['measured']; a=np.load(p/'closure_traces.npz',allow_pickle=False)
fig,axs=plt.subplots(1,2,figsize=(12,5))
for arm in ['normal','no_external_touch','sensory_cut','motor_cut','spike_output_silence']:
 r=d['episodes'][arm]['records'];t=[1000*x['time_end_s'] for x in r]
 axs[0].plot(t,[x['height_mm_declared'] for x in r],label=arm)
 axs[1].plot(t,a[arm+'_controls'].mean(axis=1),label=arm)
axs[0].set(title='Actual neural closure: still descending',xlabel='Time (ms)',ylabel='Thorax height (mm)')
axs[1].set(title='Mean commanded control (not activation)',xlabel='Interval end (ms)',ylabel='Mean normalized control')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Interface repairs verified; no autonomous standing claim')
fig.text(.5,.015,'Run '+bundle['run_id']+'; native positive control floor unchanged. Mean command hides shared-pool structure.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.045,1,.93));dest=OUT/'automatic_repair_progress.png';fig.savefig(dest,dpi=140,bbox_inches='tight');plt.close(fig);print(dest)
