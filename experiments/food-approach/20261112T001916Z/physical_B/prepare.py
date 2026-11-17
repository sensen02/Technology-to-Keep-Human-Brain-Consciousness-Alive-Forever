"""Thin stdlib admission over candidate2 draft; no model/Core/array imports."""
from bindings import ROOT,CANDIDATE,AB,OLD,FORMAT,KIND,read,sha,require,module,clone,inherited_v2_proof

def main():
 old=module('_candidate3_inherited_prepare',OLD/'prepare.py');g=dict(old.__dict__)
 def scoped_read(path):
  # Candidate2 main reads AB/FROZEN; supply the complete inherited v2 ledger.
  return inherited_v2_proof() if path==AB/'FROZEN.json' else read(path)
 def scoped_fresh(path,value):
  value=dict(value)
  if path==ROOT/'INPUT_BINDINGS.json':
   value.update(kind=KIND,behavior_format=FORMAT,AB_protocol='FROZEN_PLUS_SUPPLEMENT_AND_BASELINE_VERIFY_NEUTRAL',summary_protocol='summarize_B_v2.build_source / effective receptor onset',inherited_frozen_supplement=str(AB/'FROZEN_SUPPLEMENT.json'),inherited_frozen_supplement_sha256=sha(AB/'FROZEN_SUPPLEMENT.json'),baseline_audit=str(AB.parent/'20261111T161007Z/combined_turning/VERIFY_neutral.json'))
  elif path==ROOT/'FROZEN.json':
   value['kind']=KIND+'_INPUTS'
   value['source_hashes']=dict(value['source_hashes']);value['source_hashes'][str(ROOT/'bindings.py')]=sha(ROOT/'bindings.py')
  old.fresh(path,value)
 g.update(ROOT=ROOT,CANDIDATE=CANDIDATE,AB=AB,read=scoped_read,fresh=scoped_fresh)
 clone(old.main,g)()
if __name__=='__main__':main()
