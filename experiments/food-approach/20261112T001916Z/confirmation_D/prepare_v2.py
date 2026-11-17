"""Versioned side-only Dprepare; preserves original proposal/prepare/allC gate history."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def build_source():
 src=(ROOT/'prepare.py').read_text()
 for old,new in [('hashes=admission();','hashes=side_admission();'),("'check_codeonly.py')","'check_codeonly.py','AMENDMENT_SIDE_ONLY.json','side_admission.py','prepare_v2.py','CODEONLY_SIDE_CHECK.json')"),('C_allintegritystable=True,','C_allintegrity=True,C_allstable=False,side_only_admitted=True,front_qualified=False,side_amendment_sha256=sha(ROOT/\'AMENDMENT_SIDE_ONLY.json\'),')]:
  if src.count(old)!=1:raise ValueError('sideprepareexactadaptation '+old)
  src=src.replace(old,new)
 return src
if __name__=='__main__':
 from side_admission import side_admission
 ns={'__name__':'_D_side_prepare_v2','__file__':str(ROOT/'prepare_v2.py'),'side_admission':side_admission};exec(compile(build_source(),str(ROOT/'prepare_v2.py')+'::sideamendment','exec'),ns);ns['main']()
