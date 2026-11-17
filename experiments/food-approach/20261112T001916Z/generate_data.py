"""Freeze exactold24allarrays and newV4/V5 validation before anyoptimizer step."""
from data import *
def main():
 cpu1();guard()
 if (HERE/'DATASET.json').exists():raise FileExistsError('dataalreadyfrozen')
 old=read(OLD/'DATASET.json');assert sha(OLD/'train.npz')==old['splits']['train']['sha256'];metadata={}
 for name,n in [('train',24),('validation',12)]:
  a,meta=split(name);assert a['receptors'].shape==(1920,n,2) and a['targets'].shape==(1920,n) and np.max(np.abs(a['weights'].sum(0)-1))<1e-12
  assert all(np.isfinite(v).all() for v in a.values());path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**a);metadata[name]=dict(path=str(path),sha256=sha(path),episodes=n,configs=meta)
 original=arrays(OLD/'train.npz');new=arrays(HERE/'train.npz');checks={name+'_bitexact':np.array_equal(new[name],original[name]) for name in original};assert all(checks.values())
 save(HERE/'DATASET.json',dict(kind=FORMAT,splits=metadata,pair_table=pair_table(),old_train_allarrays_checks=checks,source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'data.py',HERE/'generate_data.py',ORIGINAL/'data.py',OLD/'DATASET.json',OLD/'train.npz',GEOMETRY)},validation_frozen_before_updates=True,oldV2V3_validation='DEVELOPMENT_HISTORY_NOT_FRESH',duplicate_note='Deterministiczero-noodor histories repeated, not independentreplicates',Core_constructions=0,model_loads=0))
 print(json.dumps(dict(all_old_train_arrays_bitexact=True,checks=len(checks),train24=True,validation12=True,pairs=12,Core_constructions=0)))
if __name__=='__main__':main()
