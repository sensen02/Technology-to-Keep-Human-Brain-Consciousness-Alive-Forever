"""Independent728actualRGB audit+TRAINrawpixelopponent statistics,no model/valforward."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ORIGINAL_AUDIT=ROOT/'landmark_neutral_v6/library/audit.py'
def build_source():
 src=ORIGINAL_AUDIT.read_text();src=src.replace("OLD=ROOT/'landmark_coverage_v5/library'","OLD=ROOT/'landmark_neutral_v6/library'").replace('VISION_NEUTRAL_HEAD_LIBRARY_V6','VISION_OPPONENT_HEAD_LIBRARY_V7')
 src=re.sub(r'672|616',lambda m:{'672':'728','616':'672'}[m.group(0)],src)
 marker=' result=dict(integrity_all_pass=all(checks.values())'
 block=""" # Pixel-only diagnostic limited to TRAIN rawimages; no policy or freshvalidation inference.
 trainstats=[]
 for row in m['rows']:
  if row['split']!='train':continue
  image=rgb[row['frame_id']].astype(np.float64)/255.;op=np.maximum(image[...,0]-.5*image[...,1]-.5*image[...,2],0.)
  trainstats.append(dict(frame_id=row['frame_id'],config_id=row['config_id'],pose_id=row['pose_id'],stimulus=row['stimulus'],opponent_mean=float(op.mean()),opponent_max=float(op.max()),opponent_abs_mean=float(np.abs(op).mean()),opponent_nonzero_fraction=float(np.mean(op>0)),opponent_gt0_01_fraction=float(np.mean(op>.01)),raw_rgb_sha256=row['rgb_sha256']))
 groups={}
 for stimulus in ('off','center','left','right','occluded_left','occluded_right','occluder_only'):
  rr=[r for r in trainstats if r['stimulus']==stimulus];means=np.array([r['opponent_mean'] for r in rr]);maxima=np.array([r['opponent_max'] for r in rr]);groups[stimulus]=dict(actual_frames=len(rr),pixel_mean_median=float(np.median(means)),pixel_mean_max=float(means.max()),pixel_maximum=float(maxima.max()),nonzero_fraction_median=float(np.median([r['opponent_nonzero_fraction'] for r in rr])))
 diagnostic=dict(kind='TRAIN_RAW_RGB_OPPONENT_DISTRIBUTION_ONLY',fixed_pixel_formula='relu(R-.5G-.5B),RGB/255',rows=trainstats,groups=groups,training_actual_frames=len(trainstats),no_pixel_replacement=True,no_neutral_forcing=True,model_forward_calls=0,validation_model_forward_calls=0,Core_constructions=0,physics_steps=0,interpretation='Measuredgray/boxopponentmaybenonzero,reporttruthnotclamp. Thisisnotvisualbehaviourproof.')
 with (HERE/'TRAIN_OPPONENT_DIAGNOSTIC.json').open('x') as f:json.dump(diagnostic,f,indent=2);f.write('\\n')
"""
 if src.count(marker)!=1:raise ValueError('opponent audit insertion shape')
 src=src.replace(marker,block+marker)
 src=src.replace("result=dict(integrity_all_pass=all(checks.values()),","result=dict(integrity_all_pass=all(checks.values()),training_opponent_diagnostic=str(HERE/'TRAIN_OPPONENT_DIAGNOSTIC.json'),training_opponent_diagnostic_sha256=sha(HERE/'TRAIN_OPPONENT_DIAGNOSTIC.json'),")
 src=src.replace("str(HERE/'layouts.py'):sha(HERE/'layouts.py')","str(HERE/'layouts.py'):sha(HERE/'layouts.py'),str(ORIGINAL_AUDIT):sha(ORIGINAL_AUDIT)")
 return src
if __name__=='__main__':
 ns={'__name__':'_opponent_actualaudit','__file__':str(HERE/'audit.py'),'ORIGINAL_AUDIT':ORIGINAL_AUDIT};exec(compile(build_source(),str(HERE/'audit.py')+'::actual728andtrainpixelstats','exec'),ns);ns['main']()
