"""Unit tests for the platform skeleton (no simulator)."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pf', HERE / 'platform_state.py')
pf = importlib.util.module_from_spec(spec); sys.modules['pf'] = pf; spec.loader.exec_module(pf)


def check(records, name, ok, **details):
    records.append(dict(name=name, passed=bool(ok), **details))
    if not ok:
        raise AssertionError(name)


def main():
    r = []
    e = pf.EnergyState()
    check(r, 'initial_energy', np.isclose(e.energy, .6))
    e.update(1.)
    check(r, 'drain_applied', np.isclose(e.energy, .58, atol=1e-12))
    e.update(1., feeding=True)
    check(r, 'intake_net', np.isclose(e.energy, .58 + .23, atol=1e-12))
    for _ in range(50):
        e.update(10., feeding=True)
    check(r, 'energy_clamped_high', np.isclose(e.energy, 1.))
    for _ in range(50):
        e.update(10.)
    check(r, 'energy_clamped_low', np.isclose(e.energy, 0.))
    try:
        e.update(-1.)
    except ValueError:
        check(r, 'negative_dt_rejected', True)
    else:
        check(r, 'negative_dt_rejected', False)

    f = pf.FoodPatch()
    check(r, 'food_full', np.isclose(f.mass, 1.))
    check(r, 'food_contains_center', f.contains((0., 0., 0.)))
    check(r, 'food_excludes_far', not f.contains((0., 0., .01)))
    taken = 0.
    for _ in range(100):
        taken += f.consume(.1)
    check(r, 'food_never_negative', f.mass >= 0. and np.isclose(f.mass, 0.))
    check(r, 'food_consumed_capped', np.isclose(taken, 1.))
    check(r, 'food_depleted_flag', f.depleted)

    s = pf.BehaviourSelector()
    check(r, 'feed_when_mouth_and_hungry', s.decide(.4, True, 1.) == 'feed')
    check(r, 'no_feed_when_sated', s.decide(.99, True, 1.) == 'rest')
    check(r, 'forage_when_hungry_away', s.decide(.4, False, 1.) == 'forage')
    check(r, 'rest_when_absent_food', s.decide(.4, False, 0.) == 'rest')
    check(r, 'no_feed_on_empty_patch', s.decide(.4, True, 0.) == 'rest')
    table = s.table()
    check(r, 'selector_table_size', len(table) == 20, rows=len(table))

    plat = pf.PlatformStep()
    out = plat.step(.1, (0., 0., 0.))
    check(r, 'platform_feeds_at_food', out['decision'] == 'feed' and out['consumed'] > 0)
    check(r, 'platform_energy_rises', out['energy'] > .6)
    hungry = pf.PlatformStep(energy=pf.EnergyState(pf.EnergyParams(initial=.3)))
    away = hungry.step(.1, (0., 0., .02))
    check(r, 'platform_forages_away', away['decision'] == 'forage' and away['consumed'] == 0.)
    check(r, 'platform_rest_above_threshold', plat.step(.1, (0., 0., .02))['decision'] == 'rest')
    check(r, 'provenance_labels', out['provenance']['behaviour_decision'] == 'engineering_selector_not_neural'
          and out['provenance']['feeding'] == 'declared_gate_virtual_food')

    plat2 = pf.PlatformStep(energy=pf.EnergyState(pf.EnergyParams(initial=.99)))
    sated = plat2.step(.1, (0., 0., 0.))
    check(r, 'sated_does_not_feed', sated['decision'] == 'rest' and sated['consumed'] == 0.)

    out_json = dict(status='PLATFORM_UNIT_TESTS_PASS' if all(x['passed'] for x in r) else 'FAIL',
                    checks_passed=sum(x['passed'] for x in r), checks=r, physics_steps=0, neural_steps=0)
    (HERE / 'UNIT_TESTS.json').write_text(json.dumps(out_json, indent=2) + '\n')
    print(json.dumps({k: out_json[k] for k in ('status', 'checks_passed', 'physics_steps', 'neural_steps')}))


if __name__ == '__main__':
    main()
