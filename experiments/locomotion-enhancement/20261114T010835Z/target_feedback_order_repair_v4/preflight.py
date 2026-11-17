"""Only fullB12REFERENCE preflightRSS amended3072; trainingremains1800."""
from pathlib import Path
import ast,resource,time
ORIGINAL=Path(__file__).resolve().parent.parent/'target_feedback_order_repair_v3/preflight.py'
PREFLIGHT_RSS_LIMIT=3072.
TRAINING_RSS_LIMIT=1800.

def preflight_resource_gate(start,limit_s=5400.):
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
 if rss>PREFLIGHT_RSS_LIMIT:raise RuntimeError('FullB12referencepreflight RSS exceeds3072MiB')
 if time.monotonic()-start>limit_s:raise RuntimeError('Preflightwall5400sexceeded,noextension')
 return rss

def main():
 tree=ast.parse(ORIGINAL.read_text());removed=0
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom) and node.module=='learning':
   names=[a for a in node.names if a.name!='resource_gate'];removed+=len(node.names)-len(names);node.names=names
 if removed!=1:raise ValueError('Exactlyonepreflightresourceimportrequired')
 ast.fix_missing_locations(tree)
 ns={'__name__':'_v4_exact_v3_preflight_reference_only','__file__':__file__,'resource_gate':preflight_resource_gate}
 exec(compile(tree,str(ORIGINAL)+'::same_numeric_code_preflight3072only','exec'),ns)
 original_save=ns['save']
 def scoped_save(path,value):
  if Path(path).name=='PREFLIGHT.json':
   value=dict(value,resource_scope='fullB12reference_preflight_only',preflight_RSS_limit_MiB=3072,training_RSS_limit_MiB=1800,prior_v3_exit1_preserved=True)
  return original_save(path,value)
 ns['save']=scoped_save;ns['main']()
if __name__=='__main__':main()
