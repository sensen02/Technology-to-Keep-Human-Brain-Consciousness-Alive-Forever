"""Independently verify physical provenance and the train/test split."""
from collections import Counter
import numpy as np
from acquire_cohort import HERE, ASSETS, digest, read, verify_physical, write

BRAIN = '71904e7a9765e5b0fb58afbe9c0ce27efa591f84997b51f0996f58a6eafbeaf3'


def main():
    manifest = read(HERE/'cohort.json')
    accepted=manifest['accepted']
    assert len(accepted)==18
    assert len({tuple(r['goal_mm'])+(r['initial_yaw_deg'],) for r in accepted})==18
    assert len({r['hashes']['BODY.npz'] for r in accepted})==18
    assert len({r['hashes']['external.npz'] for r in accepted})==18
    for r in accepted:
        dest=HERE/f"aligned-{r['seed']}"
        physical=verify_physical(dest)
        assert physical['seed']==r['seed'] and physical['model_sha']==BRAIN
        assert physical['meal_pass'] and physical['native_extra_force']==0
        assert physical['intake']==r['intake']
        with np.load(dest/'BODY.npz') as z:
            yaw=2*np.rad2deg(np.arctan2(z['initial'][6],z['initial'][3]))
            assert np.isclose(yaw,r['initial_yaw_deg'],atol=1e-8)
        with np.load(dest/'NECK.npz') as z:
            assert str(z['weights'])==BRAIN
            emitted,arrived,delays=z['emitted'],z['arrived'],z['delays']
            for j,delay in enumerate(delays):
                assert not arrived[:delay,j].any()
                assert np.array_equal(emitted[:-delay,j],arrived[delay:,j])
        assert all(digest(dest/file)==r['hashes'][file] for file in ASSETS)
    training=read(HERE/'twenty'/'training.json')
    assert training['train_seeds']==manifest['original_teacher_seeds']+[r['seed'] for r in accepted]
    assert not {88211,88224,88225}&set(training['train_seeds'])
    assert training['weights_sha256']==digest(HERE/'twenty'/'weights.pt')
    write(HERE/'verified.json',dict(accepted=18,training_count=20,updates=training['updates'],
            physical_provenance_verified=True,actual_over_neck_timing_verified=True,
            band_counts=dict(Counter(r['band'] for r in accepted)),sample_frames=sum(r['observed_frames'] for r in accepted)))
    print('PASS',read(HERE/'verified.json'))

if __name__=='__main__':main()
