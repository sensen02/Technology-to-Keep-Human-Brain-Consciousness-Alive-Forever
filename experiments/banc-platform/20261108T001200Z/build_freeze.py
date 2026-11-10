"""Build the pre-injury freeze manifest for the first-version platform.

Frozen here: the body and actuator/plant stack, the neural interface artefacts, the two
movement layers (connectome module and the labelled replacement ANN), the platform state
components, and every platform parameter and threshold. NOT frozen: anything that would
claim accepted walking, any injury/surgery model, and the absolute SI / physiological
calibration, which remains NOT_ESTABLISHED.
"""
from pathlib import Path
import hashlib, json
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
HERE = Path(__file__).resolve().parent
PQ = ROOT / 'experiments/banc-platform'


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


ARTIFACTS = [
    # role, label, path
    ('body_model', 'frozen_full_body_no_change', '/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml'),
    ('actuator', 'frozen_bounded_incremental_actuator', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py'),
    ('plant', 'frozen_plant_probe', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/plant_probe.py'),
    ('connectome_core', 'frozen_adaptation_conductance_core', ROOT / 'experiments/banc-local-learning/20261107T192931Z/dynamics/module.py'),
    ('connectome_foundation_runtime', 'frozen_neural_physical_runtime', ROOT / 'experiments/banc-physical-learning/20261107T195435Z/foundation/runtime.py'),
    ('connectome_canonical_state', 'frozen_canonical_neural_state', ROOT / 'experiments/banc-physical-learning/20261107T195435Z/foundation/NEURAL_START.npz'),
    ('connectome_readout_base', 'frozen_323x24_motor_basis', ROOT / 'experiments/banc-physical-learning/20261107T195435Z/foundation/BASE.npz'),
    ('encoder', 'frozen_uncalibrated_sensory_encoder', ROOT / 'experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py'),
    ('ann_movement_trace', 'labelled_replacement_ANN_NOT_accepted', HERE / 'ANN_BODY_u32.npz'),
    ('ann_movement_result', 'labelled_replacement_ANN_assessment', '/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32/RESULT.json'),
    ('platform_state', 'engineering_energy_food_feeding_selector', PQ / '20261107T234820Z/platform_state.py'),
    ('platform_unit_tests', 'engineering_component_tests', PQ / '20261107T234820Z/test_platform.py'),
    ('movement_status', 'movement_acceptance_state', HERE / 'movement_status.py'),
    ('leg_pools', 'structurally_derived_pools_not_anatomy', ROOT / 'experiments/banc-descending-learning/20261107T203049Z/leg-pools/POOLS.json'),
]

PARAMETERS = dict(
    energy=dict(initial=.6, drain_per_s=.02, intake_per_s=.25, bounds=[0., 1.]),
    food=dict(radius_native=.05, total_mass=1.0, intake_per_s=.25),
    selector=dict(feed_if='mouth_in_food and energy<.95', forage_if='energy<.5 and food_mass>0', else_='rest'),
    note='declared engineering constants; uncalibrated to fly metabolism, food detection distance or feeding rate',
)

NOT_FROZEN = dict(
    accepted_walking='NOT ACHIEVED: connectome module produces no displacement; the labelled ANN passes D1-D8 but fails the material foot slip criterion',
    brain_level_modulation='only a weak named-descending entry point was probed (round 11); no accepted brain-to-gait pathway',
    feeding_biomechanics='no food collision, no proboscis mechanics; feeding is a declared virtual gate',
    intake_to_locomotion_feedback='absent: the movement controller has no food sensing, so coupling is one-way',
    injury_models='none created; no surgery, no lesion experiment',
    absolute_units='NOT_ESTABLISHED for body SI and neural physiology',
    connectome_scope='frozen 20k strict subgraph and the whole-BANC strict reconstruction are models, not the fly',
)


def main():
    entries = []
    for role, label, path in ARTIFACTS:
        p = Path(path)
        entries.append(dict(role=role, label=label, path=str(p), exists=p.exists(), bytes=p.stat().st_size if p.exists() else None,
                            sha256=sha(p)))
    missing = [e['role'] for e in entries if not e['exists']]
    manifest = dict(schema='pre-injury-platform-freeze/1', frozen_at_round='round15',
                    status='FROZEN_WITH_DECLARED_LIMITS' if not missing else 'INCOMPLETE',
                    missing=missing, artefacts=entries, parameters=PARAMETERS, not_frozen=NOT_FROZEN,
                    reproduction=dict(
                        script=str(HERE / 'run_frozen_platform.py'),
                        result=str(HERE / 'RESULT_REPRODUCTION.json'),
                        expectation='arrival tick 224 at 1.125 s, feed 58 ticks, energy .5999 -> .5925, food 1.0 -> .9275'),
                    labels=dict(movement='replacement_ANN_not_connectome_not_accepted',
                                connectome='frozen_strict_reconstruction_not_the_fly',
                                feeding='declared_virtual_gate', selector='engineering_threshold_rule_not_neural',
                                energy='declared_state_uncalibrated'))
    (HERE / 'FREEZE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status=manifest['status'], artefacts=len(entries), missing=missing)))


if __name__ == '__main__':
    main()
