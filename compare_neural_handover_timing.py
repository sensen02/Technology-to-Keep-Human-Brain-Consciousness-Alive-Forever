"""Saved-only baseline versus faster causal update comparison; not behavioral certification."""
from pathlib import Path
import argparse,json
import numpy as np

def summarize(p):
 with np.load(p/'BODY.npz') as f:b={k:f[k] for k in f.files}
 r=json.loads((p/'NEURAL_RECORDS.json').read_text());h=json.loads((p/'HANDOVER.json').read_text());switch=float(h['handover_absolute_time_s'])
 t=b['time'];neural=t>=switch-1e-10;nf=b['nonfoot_count'];on=np.flatnonzero(neural&(nf>0));ctrl=np.flatnonzero(neural&np.any(b['command']>0,axis=1));sp=next((x['relative_start_s'] for x in r if x['motor_spikes']>0),None)
 late=t>=t[-1]-.1;w=json.loads((p/'GAINS.json').read_text())['weight']
 assert not b['assist'][neural].any() and not b['external'].any() and not b['warning'].any()
 for k,x in enumerate(r):assert x['applied_source_interval']==k-1
 return dict(directory=str(p),switch_s=switch,neural_duration_s=float(t[-1]-switch),command_window_s=r[0]['relative_end_s']-r[0]['relative_start_s'],first_motor_spiking_window_relative_start_s=sp,first_saved_nonzero_command_relative_s=None if not len(ctrl) else float(t[ctrl[0]]-switch),first_nonfoot_relative_s=None if not len(on) else float(t[on[0]]-switch),motor_spikes=sum(x['motor_spikes'] for x in r),last100ms_nonfoot_fraction=float(np.mean(nf[late]>0)),last100ms_foot_load_ratio=float(np.mean(b['foot_force'][late,:,2].sum(1))/w),final_com_height=float(b['com'][-1,2]),neural_assist_zero=True,body_qpos=b['qpos'],body_time=t)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('baseline');ap.add_argument('trial');args=ap.parse_args();base=summarize(Path(args.baseline));trial=summarize(Path(args.trial))
 # The assisted trajectory must remain exactly identical up to withdrawal.
 aa=base.pop('body_qpos');at=base.pop('body_time');bb=trial.pop('body_qpos');bt=trial.pop('body_time');n=int(np.flatnonzero(at>=base['switch_s']-1e-10)[0])+1
 np.testing.assert_array_equal(at[:n],bt[:n]);np.testing.assert_array_equal(aa[:n],bb[:n])
 out=dict(baseline=base,trial=trial,warmup_trajectory_exactly_identical=True,scope='Causal timing contrast only, no new physics. Shorter test endpoint cannot prove sustained behavior beyond recorded duration. Actual muscle/torque capacity analyzed separately.')
 target=Path(args.trial)/'PARENT_TIMING_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
