"""Turn published measurements into the declared normalisers the sensory front end uses.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

This tool is a GATE, not a converter with defaults: every derived parameter must name the published
measurement it comes from, the arithmetic is shown, and a parameter without a source is REFUSED
rather than filled in.  That is the project's rule expressed in code -- no source, no parameter.

Input:  data/literature_sensory_parameters.json
        {channel: {quantity: str, published_range: {low, high, unit},
                   threshold: {...} or null, gain: {...} or null,
                   source: {citation, url}, note}}
Output: outputs/model_branches/flybody_loop/sourced_sensory_config.json
        {reference_angle_rad, reference_velocity_rad_s, reference_force,
         channel_quantity_map, derivation: {parameter: {from, arithmetic, value, source}}}

The derived normalisers keep the SAME MEANING the project already declared: a channel's input value is
(measured quantity) / (normaliser), so a normaliser is the magnitude at which that channel's input
reaches one "unit" of its own physical range.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'data/literature_sensory_parameters.json'
OUT = HERE / 'outputs/model_branches/flybody_loop/sourced_sensory_config.json'
REQUIRED_FIELDS = ('quantity', 'published_range', 'source')


def fail(message):
    raise SystemExit(f'REFUSED: {message}')


def require_source(entry, channel, field):
    if field not in entry:
        fail(f'{channel}: missing {field!r}; a parameter without a published source is not written')
    source = entry[field] if field == 'source' else entry.get('source')
    if not isinstance(source, dict) or not source.get('citation'):
        fail(f'{channel}: {field!r} carries no citation')
    if not str(source.get('url', '')).startswith('http'):
        fail(f'{channel}: the citation for {field!r} carries no URL')
    return source


def main():
    if not SOURCE.exists():
        fail(f'{SOURCE} does not exist yet; the survey of published tuning data has to be recorded '
             f'first, with a citation per number')
    doc = json.loads(SOURCE.read_text())
    entries = doc.get('channels', {})
    if not entries:
        fail('the literature file has no channels')
    derivation = {}

    def range_of(channel):
        entry = entries[channel]
        for field in REQUIRED_FIELDS:
            require_source(entry, channel, field)
        rng = entry['published_range']
        if not isinstance(rng, dict) or 'low' not in rng or 'high' not in rng or 'unit' not in rng:
            fail(f'{channel}: published_range needs low, high and unit')
        return entry, float(rng['low']), float(rng['high']), rng['unit']

    # ANGLES: the position-encoding organ supplies the angular normaliser.
    angle_channel = doc.get('angle_normaliser_channel')
    if angle_channel is None:
        fail('angle_normaliser_channel is not declared; the tool will not guess which organ sets it')
    entry, low, high, unit = range_of(angle_channel)
    if unit not in ('rad', 'deg'):
        fail(f'{angle_channel}: the published range must be an angle, got {unit!r}')
    span = high - low
    reference_angle_rad = span/2.0 if unit == 'rad' else (span/2.0)*(3.141592653589793/180.0)
    derivation['reference_angle_rad'] = dict(
        value=reference_angle_rad, from_channel=angle_channel,
        arithmetic=f'half of the published range {low}..{high} {unit}',
        source=entry['source'], quantity=entry['quantity'], note=entry.get('note'))

    # VELOCITIES: a velocity normaliser needs a published velocity range; deriving it from an angle
    # range and an assumed time constant would be an invention, so it is only written when published.
    reference_velocity_rad_s = None
    vel_channel = doc.get('velocity_normaliser_channel')
    if vel_channel:
        entry_v, low_v, high_v, unit_v = range_of(vel_channel)
        if unit_v not in ('rad/s', 'deg/s'):
            fail(f'{vel_channel}: the published range must be an angular velocity, got {unit_v!r}')
        span_v = high_v - low_v
        reference_velocity_rad_s = (span_v/2.0 if unit_v == 'rad/s'
                                    else (span_v/2.0)*(3.141592653589793/180.0))
        derivation['reference_velocity_rad_s'] = dict(
            value=reference_velocity_rad_s, from_channel=vel_channel,
            arithmetic=f'half of the published range {low_v}..{high_v} {unit_v}',
            source=entry_v['source'], quantity=entry_v['quantity'], note=entry_v.get('note'))

    # FORCES: the load-encoding organ supplies the force normaliser, expressed as a fraction of body
    # weight when that is what the source reports.
    force_channel = doc.get('force_normaliser_channel')
    reference_force = None
    if force_channel:
        entry_f, low_f, high_f, unit_f = range_of(force_channel)
        if unit_f in ('body_weight', 'body_weights', 'fraction_of_body_weight'):
            reference_force = high_f
            arithmetic = f'the published upper bound {high_f} body weights'
        elif unit_f in ('uN', 'N', 'mN'):
            scale = {'uN': 1e-3, 'mN': 1.0, 'N': 1e3}[unit_f]
            reference_force = high_f*scale      # mN
            arithmetic = f'the published upper bound {high_f} {unit_f} = {reference_force} mN'
        else:
            fail(f'{force_channel}: unsupported force unit {unit_f!r}')
        derivation['reference_force'] = dict(
            value=reference_force, from_channel=force_channel, arithmetic=arithmetic,
            source=entry_f['source'], quantity=entry_f['quantity'], note=entry_f.get('note'))

    club = doc.get('club_quantity')
    if club is None:
        fail('club_quantity is not declared; the club organ channel stays at zero until a source '
             'says what it encodes')
    if club != 'zero':
        if club not in entries:
            fail(f'club_quantity {club!r} must name a channel entry with a published source')

    out = dict(
        status='DERIVED_FROM_PUBLISHED_MEASUREMENTS',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; derived parameters, one citation per number',
        source_file=str(SOURCE), source_sha256=__import__('hashlib').sha256(
            SOURCE.read_bytes()).hexdigest(),
        reference_angle_rad=reference_angle_rad,
        reference_velocity_rad_s=reference_velocity_rad_s,
        reference_force=reference_force,
        club_quantity=club,
        channel_quantity_map=doc.get('channel_quantity_map', {}),
        derivation=derivation,
        refused_without_source=[k for k in ('angle_normaliser_channel', 'velocity_normaliser_channel',
                                           'force_normaliser_channel', 'club_quantity')
                                if doc.get(k) in (None, 'zero')],
        what_this_does_not_support=[
            'A published tuning range is not this project\'s calibration: it sets the magnitude at '
            'which a channel reaches one unit, and nothing about the cell-level tuning, the sign, or '
            'the synaptic weights in the tier.',
            'Every number here is only as good as the citation recorded beside it.',
            'A parameter that had no published source was refused, not filled in, and the refusal is '
            'listed in this artifact.',
        ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2)[:2000])
    print(OUT)
    return out


if __name__ == '__main__':
    sys.exit(main())
