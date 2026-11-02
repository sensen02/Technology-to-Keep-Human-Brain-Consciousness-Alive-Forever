"""Training-window reference diagnostics; no model fitting/control or policy."""
from pathlib import Path
import json,hashlib,math
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s4-invariant-motion-target/20261102T023600Z/INVARIANT_BENDING_TARGETS.npz')
O=Path('/home/sensen/Desktop/engine修复验收/s4-bending-reference-variation/20261102T042848Z')
def main():
 O.mkdir(parents=True,exist_ok=False)
 (O/'PREREGISTRATION.json').write_text(json.dumps(dict(scope='Offline reference baseline for first13source samples/100ms and full37/300ms training interpretation',source=str(S),source_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),rules='First13trainingtime samples; constant mean comparator uses only first13 angles. No phase warp/interpolation or physical inference. Comparator cannot generate control and is not deployed.',calls=dict(model=0,neural=0,body=0,optimizer=0)),ensure_ascii=False,indent=2)+'\n')
 z=np.load(S,allow_pickle=False);a=z['angles_rad'];mask=z['angle_valid'];time=z['time_relative_s'];assert a.shape==(37,6,2) and mask.all() and np.allclose(time,np.arange(37)/120,rtol=0,atol=1e-15)
 assert abs(time[12]-.1)<1e-15 and abs(time[-1]-.3)<1e-15
 mean=a[:13].mean(0);frozen=a[0];table=[]
 for li,leg in enumerate(z['leg_names']):
  for ki,name in enumerate(z['triplet_names']):
   training=a[:13,li,ki];validation=a[:,li,ki]
   table.append(dict(leg=str(leg),triplet=str(name),train_range_rad=float(np.ptp(training)),full_range_rad=float(np.ptp(validation)),train_sd_rad=float(np.std(training)),training_mean_rad=float(mean[li,ki]),frozen_first_train_MSE=float(np.mean((training-frozen[li,ki])**2)),frozen_first_full_MSE=float(np.mean((validation-frozen[li,ki])**2)),training_mean_train_MSE=float(np.mean((training-mean[li,ki])**2)),training_mean_full_MSE=float(np.mean((validation-mean[li,ki])**2)),after_training_prefix_mean_MSE=float(np.mean((a[13:,li,ki]-mean[li,ki])**2))))
 report=dict(passed=True,source=str(S),frames=37,training_frames=13,full_window_includes_training_prefix=True,triplet_statistics=table,overall_training_mean_train_MSE=float(np.mean((a[:13]-mean)**2)),overall_training_mean_full_MSE=float(np.mean((a-mean)**2)),frozen_first_train_MSE=float(np.mean((a[:13]-frozen)**2)),frozen_first_full_MSE=float(np.mean((a-frozen)**2)),limits='Recorded geometry angle variation only. A low pose error must be compared with movement and support; a constant source mean is an offline arithmetic baseline, never a neural/body controller. Full37 temporal extension is not new animal validation.')
 (O/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(O/'REFERENCE_COMPARATORS.npz',source_angles_rad=a,source_time_s=time,source_frame=z['source_frame'],training_mean_rad=mean,first_frame_rad=frozen,training_mask=np.arange(37)<13)
 print(json.dumps({k:v for k,v in report.items() if k!='triplet_statistics'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
