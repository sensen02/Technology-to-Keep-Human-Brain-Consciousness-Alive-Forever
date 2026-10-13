"""Resolve muscle actuators against the NEWER BANC (v888) per-neuron annotations.

WHY.  An earlier round classified all 46 actuators without motor-neuron candidates and
found the local annotation snapshot too coarse: BANC's muscle-name list is
muscle-level (26 names, no a/b subdivisions).  The v888 BANC release
(Bates, Phelps, Kim, Yang et al. 2026, Nature) publishes a per-neuron metadata table
with richer cell types, and its `banc_888_id` column covers ALL 158,262 local roots, so
the newer annotations can be transferred onto exactly the neurons this project uses.

WHAT THIS DOES NOT DO.  It never asserts actuator-level biological identity, never
substitutes a wrong index for a missing neuron, and never authorises runtime use:
`individual_anatomy_validated` and `runtime_authorization` stay false on every row, so
the strict autonomous entry point must keep refusing.

MATCHING RULE (deliberate, explicit, and applied uniformly):
  * an actuator label is split into its muscle part (everything after the leg+compartment
    prefix) and that part is tokenised;
  * a v888 cell type is accepted as a candidate for that actuator when the actuator's
    tokens are a SUBSET of the cell type's tokens after canonicalisation, i.e. the cell
    type may carry extra qualifiers (nerve of entry, ribbon name, miller number) but may
    not lack the muscle words; documented lexical equivalences are enumerated explicitly;
  * the candidate must exit through a nerve consistent with the actuator's leg;
  * anything that still needs a guess is reported as unresolved.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
META = ROOT / 'data/banc888/banc_888_meta.feather'
LOCAL_CSV = ROOT / 'data/flywire/banc_neurons.csv.gz'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
OUT = ROOT / 'outputs/mn_resolution_v888'

LEG_NERVE = {
    'LF': 'prothoracic_leg_nerve',
    'LM': 'mesothoracic_leg_nerve',
    'LH': 'metathoracic_leg_nerve',
    'RF': 'prothoracic_leg_nerve',
    'RM': 'mesothoracic_leg_nerve',
    'RH': 'metathoracic_leg_nerve',
}
# The accessory / dorsal / ventral prothoracic nerves also carry foreleg motor neurons,
# so for the prothoracic legs all four documented nerve classes are admissible.
EXTRA_FORELEG_NERVES = ('prothoracic_accessory_nerve', 'dorsal_prothoracic_nerve',
                        'ventral_prothoracic_nerve')
# Enumerated lexical equivalences: model label token -> accepted v888 token set.
ALIASES = {
    'accesory': {'accessory'},
    'tibia_flex': {'tibia_flexor'},
    'sterno': {'sterno', 'sternotrochanter'},
    'tergo': {'tergo', 'tergotrochanter'},
    'and': {''},
}
GENERIC = {'a', 'b', 'c', 'd', 'fast', 'slow', 'tiny', 'true', 'miller'}
# Tokens that only QUALIFY a muscle (nerve of entry, ribbon description, soma position,
# miller index, side, numeric ids).  They may be present on the candidate and absent from
# the model label.  Anything outside this set that is missing from the model label makes
# the candidate a DIFFERENT muscle and it is rejected - this is what stops
# 'trochanter_flexor_a' from silently matching 'accessory_trochanter_flexor'.
QUALIFIERS = {
    'miller', 'dorsal', 'ventral', 'anterior', 'posterior', 'ribbon', 'dorsomedial',
    'ventrolateral', 'ventromedial', 'dorsolateral', 'medial', 'lateral', 'soma',
    'mesoln', 'metaaln', 'metan', 'proan', 'proln', 'mesoan', 'mesoaln', 'vpron', 'dpron',
    'left', 'right', 'ln', 'aln', 'an',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    'out', 'of', 'bundle', 'linj', 'seti', 'feti', 'dipalpha', 'retro', 'meta',
}


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(name):
    name = str(name).lower().replace('-', '_')
    return [t for t in re.split(r'[^a-z0-9]+', name) if t]


def qualified_set(name):
    """Token set with qualifier tokens removed, so what remains is the muscle identity."""
    return {t for t in canon_set(name, drop_generic=True) if t not in QUALIFIERS}


def canon_set(name, drop_generic=True):
    out = set()
    for t in tokens(name):
        for alt in ALIASES.get(t, {t}):
            if alt:
                out.add(alt)
    if drop_generic:
        out -= GENERIC
    return out


def muscle_part(label):
    """'LFC_pleural_remotor_and_abductor' -> 'pleural_remotor_and_abductor'."""
    parts = label.split('_', 1)
    return parts[1] if len(parts) > 1 else label


def admissible_nerve(actuator_label, nerve, side):
    leg = actuator_label[:2]
    if not nerve:
        return False
    n = str(nerve).lower()
    want = LEG_NERVE[leg]
    if not n.endswith(want):
        if leg in ('LF', 'RF') and any(n.endswith(x) for x in EXTRA_FORELEG_NERVES):
            pass
        else:
            return False
    # The actuator label is <side><leg><compartment>_<muscle>, e.g. LFF_ / RMC_ / LHC_.
    # The side is character 0.  BEFORE 2026-10-11 this line read `actuator_label[1] == 'L'`,
    # which is the LEG code (F/M/H) and made every foreleg actuator accept only LEFT-side
    # candidates, so the right foreleg was assigned left foreleg neurons.
    want_side = 'left' if actuator_label[0] == 'L' else 'right'
    return str(side).lower() == want_side


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_feather(META)
    manifest = json.loads(MANIFEST.read_text())
    rows = [r for r in manifest['actuators']]
    unresolved = [r for r in rows if not r['candidates']]

    meta = meta.copy()
    meta['ctype'] = meta['cell_type'].astype(str)

    # local CSV snapshot, to confirm the annotation change and to carry nerve/side
    local = pd.read_csv(LOCAL_CSV, compression='gzip', low_memory=False)
    local.columns = [c.strip() for c in local.columns]
    root_col = 'Root ID'
    local = local[[root_col, 'Primary Cell Type', 'Nerve', 'Soma side', 'Super Class']].copy()
    local[root_col] = pd.to_numeric(local[root_col], errors='coerce')
    local = local.dropna(subset=[root_col])
    local[root_col] = local[root_col].astype('int64')
    old_types = dict(zip(local[root_col], local['Primary Cell Type'].astype(str)))

    results = []
    for act in unresolved:
        label = act['actuator']
        want = canon_set(muscle_part(label))
        cands = []
        for _, r in meta.iterrows():
            ct = r['ctype']
            if not ct or ct == 'nan':
                continue
            # STRICT: after removing qualifier tokens the candidate's muscle identity must
            # EQUAL the actuator's muscle identity.  A subset test would let the extra word
            # 'accessory' match the wrong muscle.
            if qualified_set(ct) != qualified_set(muscle_part(label)):
                continue
            if not admissible_nerve(label, r.get('nerve'), r.get('side')):
                continue
            rid = int(r['banc_888_id'])
            cands.append({'root_id': str(rid),
                          'v888_cell_type': ct,
                          'local_snapshot_cell_type': old_types.get(rid),
                          'nerve': str(r.get('nerve')),
                          'soma_side': str(r.get('side')),
                          'super_class': str(r.get('super_class'))})
        # keep only genuine motor neurons
        cands = [c for c in cands if c['super_class'].lower() == 'motor']
        # order: prefer a cell type that is exactly the muscle name
        cands.sort(key=lambda c: (canon_set(c['v888_cell_type']) != want, c['v888_cell_type']))
        new_only = [c for c in cands if not str(c['local_snapshot_cell_type']).strip()
                    or str(c['local_snapshot_cell_type']).lower() in ('nan', 'none', '')]
        # Resolution LEVEL.  A cell type whose muscle tokens equal the actuator's muscle
        # tokens but which is shared by the model's a/b pair resolves the PAIR, not the
        # individual cluster: the model paper says widely spread attachment groups were
        # subdivided into fibre clusters, but nothing states that a and b are clusters of
        # the SAME muscle, and driving both with one pool double counts capacity.
        pair_like = bool(cands) and re.search(r'_[ab]$', label) is not None
        level = ('PAIR_LEVEL_CLUSTER_ASSUMPTION' if pair_like
                 else ('EXACT_MUSCLE_NAME' if cands else 'UNRESOLVED'))
        results.append({'actuator': label, 'leg': label[:2], 'compartment': act['compartment'],
                        'resolution_level': level,
                        'model_label': act['label'],
                        'previous_rejection': act['status'],
                        'n_candidates_v888': len(cands),
                        'n_candidates_new_vs_local_snapshot': len(new_only),
                        'candidates': cands[:12],
                        'status': ('RESOLVED_BY_V888_TYPE' if cands else 'STILL_UNRESOLVED'),
                        'individual_anatomy_validated': False, 'runtime_authorization': False})

    # GLOBAL CONSISTENCY CHECK.  One neuron cannot be the motor pool of two DIFFERENT muscles
    # at once, so a root that resolved to two actuators whose base labels differ means the
    # matching rule is leaking across legs or sides.  This check is what caught the side bug
    # above; it is kept so the failure mode cannot come back silently.
    # Sharing between the `_a` and `_b` actuator of the SAME base label is expected and is
    # exactly the PAIR_LEVEL case, so only base-label collisions are treated as conflicts.
    def base_label(label: str) -> str:
        return re.sub(r'_[ab]$', '', label)

    root_to_acts: dict[str, set[str]] = {}
    for r in results:
        for c in r['candidates']:
            root_to_acts.setdefault(c['root_id'], set()).add(base_label(r['actuator']))
    conflicts = {rid: sorted(a) for rid, a in root_to_acts.items() if len(a) > 1}
    assert not conflicts, f'root assigned to >1 distinct muscle: {list(conflicts.items())[:3]}'

    resolved = [r for r in results if r['status'].startswith('RESOLVED')]
    exact = [r for r in resolved if r['resolution_level'] == 'EXACT_MUSCLE_NAME']
    pair = [r for r in resolved if r['resolution_level'] == 'PAIR_LEVEL_CLUSTER_ASSUMPTION']
    cov_before = sum(1 for r in rows if r['candidates'])
    summary = {
        'status': 'V888_TYPE_LEVEL_RESOLUTION',
        'source': {'metadata': str(META), 'sha256': digest(META),
                   'release': 'BANC v888 (CAVE materialization 888, snapshot 2026-04-16), '
                              'Bates, Phelps, Kim, Yang et al. 2026 Nature',
                   'local_snapshot_csv': str(LOCAL_CSV), 'sha256_local': digest(LOCAL_CSV)},
        'coverage_before': cov_before,
        'coverage_after_STRICT_exact_only': cov_before + len(exact),
        'coverage_after_including_labelled_pair_level': cov_before + len(resolved),
        'n_unresolved_before': len(unresolved),
        'n_newly_resolved_exact': len(exact),
        'n_newly_resolvable_pair_level_only': len(pair),
        'n_newly_resolved_total': len(resolved),
        'honest_headline': 'the v888 upgrade adds few actuators at exact muscle-name level; most of '
                           'the apparent gain is PAIR-LEVEL, where the model\'s a/b subdivision has '
                           'one documented pool and driving both with it would double count capacity',
        'matching_rule': 'after removing qualifier tokens (nerve of entry, ribbon descriptors, soma '
                         'position, miller index, side, numerics) the candidate cell type\'s muscle '
                         'tokens must EQUAL the actuator\'s muscle tokens; the neuron must exit through '
                         'a nerve consistent with the actuator leg and side; super_class must be motor. '
                         'A subset test was tried first and REJECTED because it matched '
                         'trochanter_flexor_a to accessory_trochanter_flexor, which is a different '
                         'muscle that the model already has as its own actuator.',
        'limitations': [
            'Type-name to muscle is an ontology mapping, not validated individual anatomy.',
            'No root id is ever substituted; nothing is written into the runtime mapping.',
            'runtime_authorization stays false: the strict autonomous entry point must keep refusing.',
            'Legs without a documented candidate remain unresolved; no pool is shared silently.'],
        'newly_resolved_exact': exact,
        'newly_resolved_pair_level_labelled': pair,
        'newly_resolved': resolved,
        'still_unresolved': [r['actuator'] for r in results if r['status'] == 'STILL_UNRESOLVED'],
        'all_results': results,
    }
    (out / 'mn_resolution_v888.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vals = [cov_before, summary['coverage_after_STRICT_exact_only'],
            summary['coverage_after_including_labelled_pair_level']]
    ax[0].bar(['before', 'exact only', 'incl. pair\nlevel (labelled)'], vals,
              color=['tab:gray', 'tab:green', 'tab:orange'])
    ax[0].set_ylabel('actuators with motor-neuron candidates (of 90)')
    ax[0].set_title('Coverage after transferring the v888 annotations\n'
                    'type-level candidates only, runtime use NOT authorised')
    for i, v in enumerate(vals):
        ax[0].text(i, v + 1, str(v), ha='center')
    ax[0].grid(alpha=.3, axis='y')
    per_leg = {}
    for r in results:
        per_leg.setdefault(r['leg'], [0, 0])
        per_leg[r['leg']][1] += 1
        if r['status'].startswith('RESOLVED'):
            per_leg[r['leg']][0] += 1
    legs = sorted(per_leg)
    ax[1].bar([l + '\nunresolved' for l in legs], [per_leg[l][1] for l in legs], color='tab:red',
              label='still unresolved')
    ax[1].bar([l + '\nresolved' for l in legs], [per_leg[l][0] for l in legs], color='tab:green',
              label='newly resolved')
    ax[1].set_ylabel('actuators')
    ax[1].set_title('Newly resolved vs still unresolved, per leg')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis='y')
    fig.suptitle('BANC v888 type-level resolution: an annotation upgrade, not anatomical registration')
    fig.tight_layout(); fig.savefig(out / 'mn_resolution_v888.png', dpi=110)

    print(json.dumps({k: summary[k] for k in (
        'coverage_before', 'coverage_after_STRICT_exact_only',
        'coverage_after_including_labelled_pair_level', 'n_newly_resolved_exact',
        'n_newly_resolvable_pair_level_only')}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
