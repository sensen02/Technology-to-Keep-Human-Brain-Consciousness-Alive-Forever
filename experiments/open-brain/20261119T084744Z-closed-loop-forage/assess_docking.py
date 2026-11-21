"""Keep the paired feeding-stop repair separate from the fixed terrain suite."""
import argparse
import json
from pathlib import Path

from assess import inspect


def main(root):
    by_seed={}
    for seed,suffix in ((15103,'A'),(15107,'B')):
        old=inspect(root/f'FINAL_FLAT_{suffix}')
        new=inspect(root/f'DOCK_{seed}')
        for key in ('seed','geometry','policy','model_sha','neural_seed'):
            assert old[key]==new[key],key
        by_seed[str(seed)]=dict(old=old,after_completed_meal=new)
    res=dict(n=2,success=sum(v['after_completed_meal']['meal_pass'] for v in by_seed.values()),
             prior=sum(v['old']['meal_pass'] for v in by_seed.values()),cases=by_seed)
    (root/'DOCKING_PAIRED.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print({k:res[k] for k in ('n','success','prior')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    main(p.parse_args().root)
