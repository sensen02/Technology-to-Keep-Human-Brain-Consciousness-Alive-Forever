"""Isolatedvideo-only missinghashlib namespace repair; exact savedrenderer code preserved."""
from pathlib import Path
import importlib.util,sys,hashlib,json
HERE=Path(__file__).resolve().parent;EVIDENCE=HERE.parent/'sensory_evidence_repair';VIDEO_IDS=('V_left_to_right','O_left','F_left_agree','O_left_odorcut','F_right_agree')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def namespace():
 sentinel=object();prior=sys.modules.get('contracts',sentinel)
 try:
  c=module('_sensoryrender_recovery_frozen_contracts',EVIDENCE/'contracts.py');sys.modules['contracts']=c;adapter=module('_sensoryrender_recovery_frozen_adapter',EVIDENCE/'adapter.py');ns=adapter.namespace('render.py')
 finally:
  if prior is sentinel:sys.modules.pop('contracts',None)
  else:sys.modules['contracts']=prior
 frozen_load=ns['load']
 def load(condition):
  if condition not in VIDEO_IDS+('O_noodor',):raise ValueError('Fixedfivevideosandoriginalbaselineonly')
  checked=c.read(HERE/'CODEONLY_CHECK.json');frozen=c.read(HERE/'SOURCE_IDENTITIES.json')
  if not checked['all_passed'] or not all(checked['checks'].values()):raise ValueError('Recovery source and globalname proof required')
  c.unchanged(checked['source_hashes']);c.unchanged(frozen['source_hashes']);case=frozen_load(condition);extra={str(HERE/n):c.sha(HERE/n) for n in ('PREREG.json','render.py','check_codeonly.py','CODEONLY_CHECK.json','SOURCE_IDENTITIES.json')};extra.update(frozen['source_hashes']);case['source_hashes'].update(extra);return case
 ns.update(HERE=HERE,VIDEO_IDS=VIDEO_IDS,load=load,hashlib=hashlib,json=json)
 return ns

def main():namespace()['main']()
if __name__=='__main__':main()
