"""Generate only replacementweights and new frozen validation, originalX/Y/C exact."""
from data import *
def main():
 cpu1();guard()
 if (HERE/'DATASET.json').exists():raise FileExistsError('neveroverwritefrozen dataset')
 parent=read(OLD/'DATASET.json');assert parent['splits']['train']['sha256']==sha(OLD/'train.npz')
 meta={}
 for which,n in [('train',24),('validation',12)]:
  a,cfg=split(which);assert a['receptors'].shape==(1920,n,2) and a['targets'].shape==(1920,n);assert all(np.isfinite(v).all() for v in a.values());assert np.max(np.abs(a['weights'].sum(0)-1))<1e-12
  path=HERE/(which+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);meta[which]=dict(path=str(path),sha256=sha(path),episodes=n,configs=cfg)
 new=arrays(HERE/'train.npz');old=arrays(OLD/'train.npz');checks={}
 for name in ('receptors','targets','concentrations','time'):checks['original_'+name+'_bitexact']=np.array_equal(new[name],old[name])
 pulse=[j for j,cfg in enumerate(meta['train']['configs']) if cfg['motif'] in ('left_pulse','right_pulse')];others=[j for j in range(24) if j not in pulse]
 checks['exactly8pulse16other']=len(pulse)==8 and len(others)==16;checks['nonpulse16_weights_bitexact']=np.array_equal(new['weights'][:,others],old['weights'][:,others]);checks['all8pulse_weights_changed']=all(not np.array_equal(new['weights'][:,j],old['weights'][:,j]) for j in pulse)
 for j in pulse:
  mask=segment_masks(new['time'],meta['train']['configs'][j]);checks['episode'+str(j)+'_segments_exclusive_coverage']=bool(np.all(mask.sum(1)==1));mass=np.array([new['weights'][mask[:,k],j].sum() for k in range(4)]);checks['episode'+str(j)+'_registered_masses']=bool(np.max(np.abs(mass-np.array([.3,.2,.4,.1])))<1e-12)
 assert all(checks.values()),checks
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=meta,reuse_checks=checks,source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'generate_data.py',HERE/'data.py',OLD/'data.py',OLD/'train.npz',OLD/'DATASET.json',GEOMETRY)},old_validation_status='DEVELOPMENT_HISTORY_NOT_FRESH',new_validation_frozen_before_updates=True,duplicate_note='noodor deterministic histories repeat, not independent replications',only_change_train24='8pulse offlineweightarrays;16otherweights/X/Y/C/time bitexact',Core_constructions=0,model_loads=0))
 print(json.dumps(dict(train24=True,validation12=True,reuse_checks=len(checks),all_passed=True,Core_constructions=0)))
if __name__=='__main__':main()
