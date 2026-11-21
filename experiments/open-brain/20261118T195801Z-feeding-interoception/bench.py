"""Stationary near-mouth execution tests for the one-command feeding body."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engine.embodied.feeding import FeedingBody, FeedingState, Food


DT = 0.01
STEPS = 1600
CASES = {
    'nutrition': (Food(1., 1.), Food(0., 0.), True, False, .2),
    'same_volume_no_nutrition': (Food(1., 0.), Food(0., 0.), True, False, .2),
    'blocked': (Food(1., 1.), Food(0., 0.), True, True, .2),
    'no_contact': (Food(1., 1.), Food(0., 0.), False, False, .2),
    'full_crop': (Food(1., 1.), Food(1., 0.), True, False, .2),
    'initially_replete': (Food(1., 1.), Food(0., 0.), True, False, .8),
    'last_drop': (Food(.0013, .0031), Food(0., 0.), True, False, .2),
}


def run_case(body, case):
    patch, crop, contact, blocked, reserve = case
    state = FeedingState(patch, crop=crop, reserve=reserve, circulating=reserve)
    start_volume = patch.volume + crop.volume
    start_nutrient = patch.nutrient + crop.nutrient + reserve
    volume_waste = nutrient_waste = metabolized = 0.
    rows = []
    for tick in range(STEPS):
        feeding = 25 <= tick < 125
        result = body.advance(state, DT, feed=feeding, contact=contact, blocked=blocked)
        state = result.state
        volume_waste += result.excreted.volume
        nutrient_waste += result.excreted.nutrient
        metabolized += result.metabolized
        volume_error = (state.patch.volume + state.crop.volume + state.midgut.volume +
                        volume_waste - start_volume)
        nutrient_error = (state.patch.nutrient + state.crop.nutrient + state.midgut.nutrient +
                          nutrient_waste + state.reserve + metabolized - start_nutrient)
        assert result.crop_stretch.route == 'body_ascending'
        assert result.circulating_nutrient.route == 'hemolymph'
        rows.append(dict(time_s=(tick + 1) * DT, feed=int(feeding), contact=int(contact),
                         can_ingest=int(result.can_ingest), food=state.patch.volume,
                         ingested=result.ingested.volume, absorbed=result.absorbed,
                         intake_feedback=result.intake_feedback.value, crop=result.crop_stretch.value,
                         circulating=result.circulating_nutrient.value, reserve=state.reserve,
                         volume_error=volume_error, nutrient_error=nutrient_error))
    return rows


def verify(traces):
    fed = traces['nutrition']
    empty = traces['same_volume_no_nutrition']
    for rows in traces.values():
        assert max(abs(r['volume_error']) for r in rows) < 1e-12
        assert max(abs(r['nutrient_error']) for r in rows) < 1e-12
        assert all(not r['ingested'] for r in rows if not r['feed'])
        assert all(bool(r['intake_feedback']) == bool(r['ingested']) for r in rows)
    assert all(a['crop'] == b['crop'] and a['ingested'] == b['ingested']
               for a, b in zip(fed, empty, strict=True))
    assert fed[25]['ingested'] > 0 and fed[25]['absorbed'] == 0
    assert fed[-1]['reserve'] > empty[-1]['reserve'] + .1
    assert fed[125]['feed'] == 0 and fed[125]['can_ingest'] == 1
    assert traces['full_crop'][25]['crop'] > .9 and fed[25]['crop'] < .1
    assert traces['last_drop'][25]['ingested'] == CASES['last_drop'][0].volume
    assert traces['last_drop'][26]['ingested'] == 0.
    for name in ('blocked', 'no_contact'):
        assert not any(r['ingested'] or r['absorbed'] for r in traces[name])
        assert all(r['food'] == 1. for r in traces[name])


def main(output):
    output.mkdir()
    body = FeedingBody()
    saturated = body.advance(FeedingState(Food(1., 1.), crop=Food(1., 0.)),
                             DT, feed=True, contact=True)
    assert saturated.ingested.volume == 0. and not saturated.can_ingest
    assert saturated.state.patch.volume == 1.
    resumed = body.advance(saturated.state, DT, feed=True, contact=True)
    assert resumed.can_ingest and 0. < resumed.ingested.volume < body.intake_rate * DT
    traces = {name: run_case(body, case) for name, case in CASES.items()}
    verify(traces)
    metrics = {}
    for name, rows in traces.items():
        with (output / (name + '.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        metrics[name] = dict(
            ingested=sum(r['ingested'] for r in rows),
            absorbed=sum(r['absorbed'] for r in rows),
            maximum_crop=max(r['crop'] for r in rows),
            maximum_circulating=max(r['circulating'] for r in rows),
            final_reserve=rows[-1]['reserve'],
            max_volume_error=max(abs(r['volume_error']) for r in rows),
            max_nutrient_error=max(abs(r['nutrient_error']) for r in rows),
        )
    (output / 'RESULT.json').write_text(json.dumps(
        dict(status='BODY_BENCH_PASS', timestep_s=DT, total_steps=STEPS * len(CASES),
             parameters=asdict(body), commands='scripted actuator test',
             native_physics_steps=0, training_updates=0, metrics=metrics),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
