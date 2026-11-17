"""Freeze24train/12validation before all optimizer updates; no model forward."""
from data import *
def main():
 cpu1();guard()
 if (HERE/'DATASET.json').exists():raise FileExistsError('never overwrite frozen long sequences')
 meta={}
 for name,n in [('train',24),('validation',12)]:
  arrays,configs=split(name);assert arrays['receptors'].shape==(1920,n,2) and arrays['targets'].shape==(1920,n) and arrays['weights'].shape==(1920,n)
  assert all(np.isfinite(v).all() for v in arrays.values()) and np.max(np.abs(arrays['targets']))<=.8+1e-12
  assert np.max(np.abs(arrays['weights'].sum(0)-1))<1e-12
  path=HERE/(name+'.npz')
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,**arrays);meta[name]=dict(path=str(path),sha256=sha(path),episodes=n,configs=configs)
 save(HERE/'DATASET.json',dict(kind=FORMAT,physical_steps=0,Core_constructions=0,splits=meta,source_hashes={str(p):sha(p) for p in (HERE/'PREREG.json',HERE/'data.py',HERE/'generate_data.py',GEOMETRY)},inputs='onlyreceptors2',target_metadata='environmentfactor/envelope offline only; NEVER deployment input',validation_frozen_before_updates=True,duplicate_note='Zero-noise no-odor histories repeat across configured timing/amplitude variants; intentionally balance motif, not independent replicate evidence',oldAB_status='developmentfailure_NOTfreshheldout'))
 print(json.dumps(dict(train=24,validation=12,steps=1920,Core_constructions=0,models_loaded=0)))
if __name__=='__main__':main()
