"""Read-only pool coverage attribution; never changes anatomy or IDs."""
from .neural_command import _ids


def pool_coverage(pool_root_ids, dataset_root_ids, old_tier_root_ids, new_tier_root_ids,
                  mapped_root_ids, *, unresolved=False):
    def ids(x):
        x=tuple(x)
        return set(_ids(x)) if x else set()
    pool=ids(pool_root_ids);dataset=ids(dataset_root_ids)
    old=ids(old_tier_root_ids);new=ids(new_tier_root_ids);mapped=ids(mapped_root_ids)
    if not old<=dataset or not new<=dataset:
        raise ValueError('tier identities must be contained in dataset')
    expected=pool&new
    unexpected=mapped-expected
    omitted=expected-mapped
    if unresolved and pool:raise ValueError('unresolved row unexpectedly has declared pool')
    return {'pool_count':len(pool),'dataset_missing':sorted(pool-dataset,key=int),
            'old_tier_missing':sorted((pool&dataset)-old,key=int),
            'new_tier_missing':sorted((pool&dataset)-new,key=int),
            'mapping_omitted':sorted(omitted,key=int),'mapping_unexpected':sorted(unexpected,key=int),
            'code_mapping_error':bool(omitted or unexpected),
            'cause':('root_mapping_error' if omitted or unexpected else
                     'source_annotation_unresolved' if unresolved or not pool else
                     'dataset_identity_absent' if pool-dataset else
                     'tier_selection' if pool-new else 'complete')}
