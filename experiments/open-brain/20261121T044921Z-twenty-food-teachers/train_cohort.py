"""Check the twenty source runs before continuing training."""
from acquire_cohort import HERE, digest, read, verify_physical
from practice import demo, fit, save_json


def main():
    manifest=read(HERE/'cohort.json')
    accepted=manifest['accepted']
    original=manifest['original_teacher_seeds']
    seeds=original+[r['seed'] for r in accepted]
    assert len(original)==2 and len(accepted)==18 and len(set(seeds))==20
    for row in accepted:
        for file,digest_expected in row['hashes'].items():
            assert digest(demo(row['seed'])/file)==digest_expected
    for seed in seeds:
        row=verify_physical(demo(seed))
        assert row['intake']>=.18 and row['ended_standing'] and row['first_fall_s'] is None
    fit(seeds,3500,'twenty',initial='long-rollout',long=True)
    save_json(HERE/'twenty'/'cohort-identity.json',{'seeds':seeds,'cohort_sha256':digest(HERE/'cohort.json')})


if __name__=='__main__':
    main()
