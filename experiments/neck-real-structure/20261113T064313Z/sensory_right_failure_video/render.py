"""Independent saved F_right_agree failure video, no oldsource modification or physics."""
from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent;EVIDENCE=HERE.parent/'sensory_evidence_repair'
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def namespace():
 sentinel=object();prior=sys.modules.get('contracts',sentinel)
 try:
  contracts=module('_rightfailure_frozen_repair_contracts',EVIDENCE/'contracts.py');sys.modules['contracts']=contracts;adapter=module('_rightfailure_frozen_repair_adapter',EVIDENCE/'adapter.py');ns=adapter.namespace('render.py')
 finally:
  if prior is sentinel:sys.modules.pop('contracts',None)
  else:sys.modules['contracts']=prior
 frozen_load=ns['load'];paths=[HERE/n for n in ('AMENDMENT_PREREG.json','render.py','check_codeonly.py','CODEONLY_CHECK.json','IMMUTABLE_SOURCE_IDENTITIES.json')]
 def load(condition):
  if condition not in ('F_right_agree','O_noodor'):raise ValueError('Onlyfifthcaseandoriginalbaseline permitted')
  checked=contracts.read(HERE/'CODEONLY_CHECK.json')
  if not checked['all_passed'] or not all(checked['checks'].values()):raise ValueError('Independentfifthentry check required')
  contracts.unchanged(checked['source_hashes']);frozen=contracts.read(HERE/'IMMUTABLE_SOURCE_IDENTITIES.json');contracts.unchanged(frozen['source_hashes']);case=frozen_load(condition)
  extra={str(p):contracts.sha(p) for p in paths};extra.update(frozen['source_hashes']);case['source_hashes'].update(extra);return case
 ns.update(HERE=HERE,VIDEO_IDS=('F_right_agree',),load=load)
 return ns

def main():namespace()['main']()
if __name__=='__main__':main()
