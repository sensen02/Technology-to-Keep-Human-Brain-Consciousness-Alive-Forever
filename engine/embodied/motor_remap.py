"""Preserve frozen pool weights by anatomical root identity, never local row order."""
import numpy as np
from .motor_translation import FiniteMotorTranslator


def remap_frozen_pool_weights(old_manifest, new_manifest, old_weights):
    old=FiniteMotorTranslator(old_manifest,old_weights)
    new=FiniteMotorTranslator(new_manifest)
    def identities(manifest, translator):
        by_local={}
        by_pool={}
        for row in manifest['rows']:
            local=row['local_indices']; roots=row.get('tier_root_ids',[])
            if len(local)!=len(roots):raise ValueError('root/local mapping lengths differ')
            for idx,root in zip(local,roots):
                if idx in by_local and by_local[idx]!=int(root):
                    raise ValueError('inconsistent local root identity')
                by_local[idx]=int(root)
            key=tuple(sorted(local))
            if key:
                anatomy=tuple(sorted(int(x) for x in row['pool_root_ids']))
                if key in by_pool and by_pool[key]!=anatomy:
                    raise ValueError('different anatomical pools collapsed onto one tier pool')
                by_pool[key]=anatomy
        result=[]
        for pool in translator.pools:
            if any(i not in by_local for i in pool):raise ValueError('missing local root identity')
            result.append((tuple(sorted(by_local[i] for i in pool)),by_pool[pool]))
        return result
    old_ids=identities(old_manifest,old);new_ids=identities(new_manifest,new)
    if len(set(old_ids))!=len(old_ids):raise ValueError('ambiguous old pool identities')
    weights={identity:old.weights[i] for i,identity in enumerate(old_ids)}
    if set(old_ids)!=set(new_ids):
        raise ValueError('pool membership/coverage changed: frozen weights cannot transfer silently')
    # Pin actuator to the same complete anatomical pool as well.
    def actuator_ids(m):
        return {r['actuator']:tuple(sorted(int(x) for x in r['pool_root_ids'])) for r in m['rows']}
    if actuator_ids(old_manifest)!=actuator_ids(new_manifest):
        raise ValueError('actuator anatomical identities changed')
    return np.stack([weights[x] for x in new_ids]) if new_ids else np.zeros((0,5))
