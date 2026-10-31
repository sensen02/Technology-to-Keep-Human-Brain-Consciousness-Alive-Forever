"""Bounded four-cache-root derivative of six-root runner; historical source unchanged."""
from pathlib import Path
import json,inspect,numpy as np
import run_required_motor_tier as prior
import derive_required_motor_tier as deriv
import run_neural_muscle_closure as c
from run_one_second_motor_pair import dump,sha
O=Path('/home/sensen/Desktop/engine修复验收/s2-four-motor-tier/20261027T174000Z')
ROOTS=[720575941391252886,720575941497037218,720575941511883985,720575941559306355]
# Reuse actual build and require graph reachability before allowing physical arm.
def checked_derive(required=(),base=None):
 out=deriv.derive_tier(required,base)
 if not required:return out
 assert list(required)==ROOTS
 n=base['n'];assert out['n']==n+4
 groups,_=c.channel_index_map(out['n'],out['tier']['root_ids']);seeds={int(i) for leg in groups.values() for channel in leg.values() for i in channel}
 W=(out['edges']['We']+out['edges']['Wi']).tocsc();reachable=set(seeds);front=list(seeds)
 while front:
  j=front.pop()
  for i in W.indices[W.indptr[j]:W.indptr[j+1]]:
   if int(i) not in reachable:reachable.add(int(i));front.append(int(i))
 states=[dict(root=int(out['tier']['root_ids'][i]),local=i,reachable_from_existing_sensory=i in reachable,direct_sensory=i in seeds) for i in range(n,n+4)]
 dump(O/'INPUT_REACHABILITY.json',states)
 if not any(r['reachable_from_existing_sensory'] for r in states):
  dump(O/'OFFLINE_STOP.json',dict(reason='none of four nodes reachable from existing sensory groups',body_runs=0));raise SystemExit(0)
 return out

def main():
 # In-memory minimal derivative, explicit replacements recorded for reproducibility.
 src=inspect.getsource(prior.main)
 src=src.replace("required=sorted({root for r in diff for root in r['unavailable_in_tier']});assert len(required)==6","required=ROOTS;assert len(required)==4")
 src=src.replace("n+6","n+4").replace("six exact roots","four exact roots").replace("range(n,n+6)","range(n,n+4)")
 # Duplicate/missing tests invoke original validator; actual extension invokes checked wrapper.
 src=src.replace('derive_tier(bad,base)','deriv.derive_tier(bad,base)')
 ns=dict(vars(prior));ns.update(O=O,ROOTS=ROOTS,__file__=__file__,derive_tier=checked_derive,deriv=deriv)
 exec(compile(src,__file__,'exec'),ns);ns['main']()
 (O/'DERIVED_MAIN.txt').write_text(src)
 dump(O/'DERIVATION.json',dict(parent=str(Path(prior.__file__)),parent_sha256=sha(Path(prior.__file__)),builder=str(Path(deriv.__file__)),builder_sha256=sha(Path(deriv.__file__)),prior_connection_evidence='/home/sensen/Desktop/engine修复验收/s2-required-motor-tier/20261027T170000Z/OFFLINE_SOURCE_CONNECTIONS.json',excluded_version_roots=[720575941434056663,720575941500775137],candidate_pool_used=False,only_factor='four cache nodes and sourced internal edges; historical pools unchanged'))
if __name__=='__main__':main()
