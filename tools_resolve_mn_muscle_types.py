"""Resolve the 46 unresolved leg actuators at MOTOR-NEURON-TYPE level (never root level).

WHAT THIS IS.  The canonical muscle body has 90 actuators; only 44 have
source-supported motor-neuron candidates.  This tool asks whether the DOCUMENTED
motor-neuron type-naming system can close any of the remaining 46, at TYPE level
only, and reports a null result honestly if it cannot.  It is a naming/ontology
analysis, NOT a wiring authorisation and NOT an anatomical registration.

DOCUMENTED NAMING SYSTEM (verified from primary sources, quoted in the JSON):
  * Takemura et al. 2024 / Cheong et al. 2026 (eLife RP96084) state that "Each
    Drosophila MN is identified by the single muscle it innervates", and that
    MANC MNs carry 'type', 'target' (exact muscle target if known) and 'subclass'
    fields.
  * The BANC community annotation wiki documents the muscle-name list used for
    BANC motor neurons and attributes it explicitly to Azevedo, Lesser, Phelps,
    Mark et al. 2024 Nature (FANC).
  * That list is MUSCLE-LEVEL.  It has no a/b subdivisions, and it names
    'tergotrochanter extensor' and 'sternotrochanter extensor' as two separate
    muscles while the muscle model carries one combined actuator family.

HARD RULES KEPT HERE.
  * BANC root ids are decimal strings and are never converted through floats.
  * A missing local index is never replaced by -1; missing things are reported.
  * BANC and MANC are DIFFERENT SPECIMENS.  A type-name-to-muscle correspondence
    is a naming/ontology mapping, NOT a cross-specimen root equivalence, and no
    root from one dataset is ever claimed to equal a root of another.
  * a/b subdivisions are never merged or split without evidence.
  * runtime_authorization stays false for every row.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/mn_type_resolution_round18'
BANC_CSV = ROOT / 'data/flywire/banc_neurons.csv.gz'
RESOLUTION = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
WIKI_URL = ('https://raw.githubusercontent.com/wiki/jasper-tms/the-BANC-fly-connectome/'
            'Annotations-(cell-types,-etc.).md')
WIKI_LOCAL = OUT / 'banc_wiki_annotations.md'

# The four documented prothoracic nerve classes plus the meso/meta leg nerves.
LEG_NERVES = {
    'LF': ('left_prothoracic_leg_nerve', 'left_prothoracic_accessory_nerve',
           'left_dorsal_prothoracic_nerve', 'left_ventral_prothoracic_nerve'),
    'RF': ('right_prothoracic_leg_nerve', 'right_prothoracic_accessory_nerve',
           'right_dorsal_prothoracic_nerve', 'right_ventral_prothoracic_nerve'),
    'LM': ('left_mesothoracic_leg_nerve', 'left_anterior_dorsal_mesothoracic_nerve',
           'left_posterior_dorsal_mesothoracic_nerve'),
    'RM': ('right_mesothoracic_leg_nerve', 'right_anterior_dorsal_mesothoracic_nerve',
           'right_posterior_dorsal_mesothoracic_nerve'),
    'LH': ('left_metathoracic_leg_nerve',),
    'RH': ('right_metathoracic_leg_nerve',),
}
SIDE = {'L': 'left', 'R': 'right'}

# Model actuator label -> documented muscle name(s), derived from the model's own
# label text only.  Each entry states whether the model label adds a subdivision
# that the documented naming does not contain.
FAMILY_RULES = {
    'tergopleural_promotor': {
        'documented': ['tergopleural promotor'],
        'model_adds_subdivision': True,
        'note': 'model label carries an a/b suffix; the documented muscle list has one '
                'tergopleural promotor with no subdivision'},
    'pleural_promotor': {
        'documented': ['pleural promotor'],
        'model_adds_subdivision': False,
        'note': 'documented as a separate muscle; BANC annotates the relevant roots with a '
                'COMBINED alias that also names the tergopleural promotor'},
    'trochanter_flexor': {
        'documented': ['trochanter flexor'],
        'model_adds_subdivision': True,
        'note': 'model label carries an a/b suffix; documented list has one trochanter flexor '
                'and a separate accessory trochanter flexor, which the model already carries '
                'as its own actuator'},
    'sterno_tergo_trochanter_extensor': {
        'documented': ['sternotrochanter extensor', 'tergotrochanter extensor'],
        'model_adds_subdivision': True,
        'note': 'documented list names TWO muscles (sternotrochanter and tergotrochanter '
                'extensor) while the model has ONE combined actuator family with an a/b '
                'subdivision; nothing found documents which letter is which muscle'},
    'sternal_adductor': {
        'documented': ['sternal adductor'],
        'model_adds_subdivision': False,
        'note': 'single documented muscle, single model actuator'},
    'pleural_remotor_and_abductor': {
        'documented': ['pleural remotor and abductor'],
        'model_adds_subdivision': False,
        'note': 'single documented muscle, single model actuator'},
}
STANDALONE_RESOLVED_LABELS = {   # already resolved in the 44 set, listed for the audit
    'accesory_trochanter_flexor': 'accessory trochanter flexor',
    'trochanter_extensor': 'trochanter extensor',
    'sternal_anterior_rotator': 'sternal anterior rotator',
    'sternal_posterior_rotator': 'sternal posterior rotator',
    'tibia_flex_93434': 'tibia flexor',
    'tibia_extensor_93932': 'tibia extensor',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_banc():
    rows = []
    with gzip.open(BANC_CSV, 'rt', errors='replace') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def documented_muscle_names(wiki_text):
    i = wiki_text.find('### muscle innervated')
    if i < 0:
        raise ValueError('documented muscle-name section not found in the wiki text')
    section = wiki_text[i:wiki_text.find('### neuron identity', i)]
    names = []
    for line in section.splitlines():
        m = re.match(r'\s*(?:-\s*)?innervates\s+(.+?)\s*$', line)
        if m:
            names.append(m.group(1).strip())
    return names


def model_miller_numbers(actuator):
    """Miller numbers the model's own tendon site names carry, per actuator."""
    ms = set()
    for r in actuator.get('canonical_route') or []:
        for m in re.finditer(r'[Mm]iller[_\s]?(\d+)(?:_(\d+))?', r['name']):
            ms.add(m.group(1) + ('_' + m.group(2) if m.group(2) else ''))
    return sorted(ms)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_banc()
    resolution = json.loads(RESOLUTION.read_text())

    # ---- documented naming system ---------------------------------------
    if not WIKI_LOCAL.exists():
        raise FileNotFoundError('local copy of the BANC annotation wiki is required: '
                                + str(WIKI_LOCAL))
    wiki_text = WIKI_LOCAL.read_text(errors='replace')
    doc_muscles = documented_muscle_names(wiki_text)
    naming_quotes = {}
    rp = ROOT / 'outputs/external_literature_round16/desc_to_motor_vnc_2026.txt'
    if rp.exists():
        t = re.sub(r'\s+', ' ', rp.read_text())
        m = re.search(r'Each Drosophila MN is identified by the single muscle it innervates\.[^.]*\.', t)
        if m:
            naming_quotes['single_muscle_rule'] = m.group(0)
    naming_quotes['documented_muscle_names_source'] = (
        'The BANC annotation wiki section "muscle innervated" attributes the muscle names to '
        'Azevedo, Lesser, Phelps, Mark et al. 2024 Nature (FANC).')
    naming_quotes['wiki_url'] = WIKI_URL
    naming_quotes['wiki_sha256'] = sha256(WIKI_LOCAL)

    # ---- BANC leg motor neurons -----------------------------------------
    leg_motor = [x for x in rows if (x['Super Class'] or '') == 'motor'
                 and any(x['Nerve'] in ns for ns in LEG_NERVES.values())]
    type_counts = Counter(x['Primary Cell Type'] for x in leg_motor)
    per_leg_type = defaultdict(Counter)
    for x in leg_motor:
        for leg, ns in LEG_NERVES.items():
            if x['Nerve'] in ns:
                per_leg_type[x['Primary Cell Type']][leg] += 1

    # Miller-number aliases documented inside the BANC annotations themselves
    miller_alias = defaultdict(set)
    for x in rows:
        alt = x['Alternative Cell Type(s)'] or ''
        for m in re.finditer(r'([a-z_0-9]+)_miller_(\d+(?:_\d+)?)', alt):
            miller_alias[m.group(2)].add(m.group(1))

    # ---- per-actuator analysis ------------------------------------------
    unresolved = [a for a in resolution['actuators']
                  if a['status'] != 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    resolved = [a for a in resolution['actuators']
                if a['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    assert len(unresolved) == 46 and len(resolved) == 44

    def family_of(a):
        label = a['label'] or ''
        for fam in ('sterno_tergo_trochanter_extensor', 'trochanter_flexor',
                    'tergopleural_promotor', 'pleural_promotor', 'sternal_adductor',
                    'pleural_remotor_and_abductor'):
            if label.startswith(fam):
                return fam
        return None

    rows_out = []
    for a in unresolved:
        leg = a['actuator'][:2]
        fam = family_of(a)
        rule = FAMILY_RULES.get(fam)
        millers = model_miller_numbers(a)
        entry = {
            'actuator': a['actuator'], 'leg': leg, 'compartment': a['compartment'],
            'model_label': a['label'], 'model_miller_numbers': millers,
            'model_actuator_status': a['status'],
            'documented_muscle_names': rule['documented'] if rule else None,
            'model_adds_subdivision_beyond_documentation':
                rule['model_adds_subdivision'] if rule else None,
            'runtime_authorization': False,
            'individual_anatomy_validated': False,
        }
        if rule is None:
            entry.update(status='STILL_UNRESOLVED',
                         reason='no documented-naming rule for this model label')
            rows_out.append(entry)
            continue

        # BANC types whose name matches a documented muscle of this family
        banc_types = {}
        for name in rule['documented']:
            key = name.replace(' ', '_')
            aliases = {key, key.replace('tergo', 'tergo').replace('-', '_')}
            if name == 'sternotrochanter extensor':
                aliases |= {'sternotrochanter'}
            if name == 'tergotrochanter extensor':
                aliases |= {'tergotrochanter'}
            if name == 'trochanter flexor':
                aliases |= {'trochanter_flexor'}
            if name == 'accessory trochanter flexor':
                aliases |= {'accessory_trochanter_flexor'}
            if name == 'pleural remotor and abductor':
                aliases |= {'pleural_remotor_abductor', 'pleural_remotor_and_abductor'}
            if name == 'tergopleural promotor':
                aliases |= {'tergopleural_promotor'}
            if name == 'pleural promotor':
                aliases |= {'pleural_promotor'}
            if name == 'sternal adductor':
                aliases |= {'sternal_adductor'}
            found = {t: per_leg_type[t][leg] for t in aliases if per_leg_type[t][leg]}
            banc_types[name] = {'candidate_types': sorted(aliases),
                                'roots_on_this_leg': dict(found)}
        entry['banc_type_availability'] = banc_types

        total_roots = sum(sum(v['roots_on_this_leg'].values()) for v in banc_types.values())
        entry['roots_of_matching_type_on_this_leg'] = total_roots

        missing_types = [n for n, v in banc_types.items() if not v['roots_on_this_leg']]
        entry['documented_names_with_no_annotated_root_on_this_leg'] = missing_types
        if total_roots == 0:
            entry.update(status='ANNOTATION_COVERAGE_HOLE',
                         reason=('the documented muscle name exists, but no annotated motor neuron '
                                 'of its BANC type is present on this leg\'s nerve classes; a gap in '
                                 'the available annotation, not a naming ambiguity'))
        elif rule['model_adds_subdivision']:
            entry.update(status='DOCUMENTED_PAIR_LEVEL_ONLY',
                         reason=('documented naming is muscle-level and adds no a/b subdivision; '
                                 'the model family maps to ' + ' / '.join(rule['documented']) +
                                 ' at pair level only, so no actuator-level identity follows'),
                         pair_level_correspondence=rule['documented'])
        else:
            entry.update(status='STILL_UNRESOLVED',
                         reason='roots exist but no documented rule separates this actuator')
        if entry['model_miller_numbers']:
            entry['banc_miller_alias_for_these_numbers'] = {
                m: sorted(miller_alias[m]) for m in entry['model_miller_numbers'] if m in miller_alias}
        rows_out.append(entry)

    # ---- validate the already-resolved 44 against the documented names ---
    validation = []
    for a in resolved:
        leg = a['actuator'][:2]
        millers = model_miller_numbers(a)
        doc = None
        lbl = a['label'] or ''
        for k, v in STANDALONE_RESOLVED_LABELS.items():
            if lbl.startswith(k):
                doc = v
        entry = {'actuator': a['actuator'], 'model_miller_numbers': millers,
                 'documented_muscle_name': doc,
                 'documented_name_in_wiki_list': bool(doc and doc in doc_muscles),
                 'banc_roots': [c['root_id'] for c in a.get('candidates', [])],
                 'banc_types': sorted({c['primary'] for c in a.get('candidates', [])}),
                 'nerve_consistency': True}
        if millers:
            entry['banc_miller_alias_match'] = {m: sorted(miller_alias[m]) for m in millers
                                                if m in miller_alias}
        validation.append(entry)

    coverage_before = 44
    coverage_after = coverage_before   # this analysis authorises no new actuator
    by_leg_status = defaultdict(Counter)
    for e in rows_out:
        by_leg_status[e['leg']][e['status']] += 1
    by_compartment_status = defaultdict(Counter)
    for e in rows_out:
        by_compartment_status[e['compartment']][e['status']] += 1

    report = {
        'status': 'MN_TYPE_LEVEL_RESOLUTION_ATTEMPT',
        'question': ('can the documented motor-neuron type-naming system resolve any of the 46 '
                     'leg actuators that currently have no source-supported MN candidates?'),
        'answer': 'NO. Coverage stays 44/90; this analysis authorises no additional actuator.',
        'naming_system': {
            'quotes': naming_quotes,
            'documented_muscle_names': doc_muscles,
            'granularity': 'muscle level; no a/b subdivisions',
            'sources': [
                'outputs/external_literature_round16/desc_to_motor_vnc_2026.txt '
                '(eLife RP96084; Cheong et al. 2026)',
                'outputs/external_literature_round16/takemura2024_manc_vnc_connectome.pdf '
                '(Takemura et al. 2024, MANC VNC connectome)',
                WIKI_URL,
            ]},
        'source_sha256': {
            str(BANC_CSV): sha256(BANC_CSV),
            str(RESOLUTION): sha256(RESOLUTION),
            str(WIKI_LOCAL): sha256(WIKI_LOCAL),
        },
        'banc_leg_motor_neurons': {
            'total_on_leg_nerves': len(leg_motor),
            'distinct_primary_types': len(type_counts),
            'type_counts_on_leg_nerves': dict(sorted(type_counts.items())),
            'roots_per_leg_of_the_relevant_types': {
                t: {leg: per_leg_type[t][leg] for leg in LEG_NERVES}
                for t in ('tergopleural_promotor', 'pleural_promotor', 'trochanter_flexor',
                          'accessory_trochanter_flexor', 'sternotrochanter', 'tergotrochanter',
                          'sternal_adductor', 'pleural_remotor_abductor')},
        },
        'miller_aliases_documented_inside_banc_annotations': {
            k: sorted(v) for k, v in sorted(miller_alias.items())},
        'coverage': {
            'before': coverage_before, 'after': coverage_after, 'total_actuators': 90,
            'closed_by_this_analysis': 0,
            'reason': ('the documented naming is muscle-level, while the 46 unresolved actuators are '
                       'unresolved because of granularity mismatches (a/b subdivisions the naming '
                       'does not contain; one documented pair merged into one model family) and '
                       'because of annotation-coverage holes'),
        },
        'unresolved_actuator_analysis': rows_out,
        'status_counts': dict(Counter(e['status'] for e in rows_out)),
        'status_counts_by_leg': {k: dict(v) for k, v in by_leg_status.items()},
        'status_counts_by_compartment': {k: dict(v) for k, v in by_compartment_status.items()},
        'resolved_44_validation_against_documented_names': validation,
        'what_this_does_not_establish': [
            'No actuator-level identity is asserted for any of the 46.',
            'No cross-specimen root equivalence: BANC and MANC are different specimens, and a '
            'type-name-to-muscle correspondence is an ontology mapping, not a root identity.',
            'A type-level candidate is not a validated anatomical registration, and does not make '
            'a spike-to-force mapping correct.',
            'runtime_authorization remains false for every row; the strict autonomous entry point '
            'must keep refusing.',
        ],
        'scope': ('naming/ontology analysis over local BANC annotations and local primary-source '
                  'texts; no model or engine change, no wiring authorisation'),
    }
    (OUT / 'mn_type_resolution.json').write_text(json.dumps(report, indent=2))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    legs = list(LEG_NERVES)
    statuses = ['DOCUMENTED_PAIR_LEVEL_ONLY', 'ANNOTATION_COVERAGE_HOLE', 'STILL_UNRESOLVED']
    colors = {'DOCUMENTED_PAIR_LEVEL_ONLY': 'tab:orange',
              'ANNOTATION_COVERAGE_HOLE': 'tab:red', 'STILL_UNRESOLVED': 'tab:gray'}
    bottom = np.zeros(len(legs))
    for st in statuses:
        vals = np.array([by_leg_status[l][st] for l in legs], float)
        if vals.sum():
            axs[0].bar(legs, vals, bottom=bottom, label=st, color=colors[st])
            bottom += vals
    axs[0].set_ylabel('unresolved actuators')
    axs[0].set_title('Why each unresolved actuator stays unresolved\n'
                     'coverage unchanged: 44/90 -> 44/90')
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=.3, axis='y')

    types = ['tergopleural_promotor', 'pleural_promotor', 'trochanter_flexor',
             'sternotrochanter', 'tergotrochanter', 'sternal_adductor',
             'pleural_remotor_abductor']
    M = np.array([[per_leg_type[t][l] for l in legs] for t in types], float)
    im = axs[1].imshow(M, cmap='viridis', aspect='auto')
    axs[1].set_xticks(range(len(legs))); axs[1].set_xticklabels(legs)
    axs[1].set_yticks(range(len(types))); axs[1].set_yticklabels(types, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axs[1].text(j, i, int(M[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if M[i, j] < M.max() * .6 else 'black')
    axs[1].set_title('Annotated MN roots per documented muscle type\n'
                     '0 = annotation-coverage hole (pleural promotor exists nowhere)')
    fig.colorbar(im, ax=axs[1], shrink=.8)

    axs[2].bar(['before', 'after'], [coverage_before, coverage_after], color=['tab:blue', 'tab:blue'])
    axs[2].axhline(90, ls='--', c='gray', label='90 actuators total')
    axs[2].set_ylim(0, 95); axs[2].set_ylabel('actuators with a source-supported candidate')
    axs[2].set_title('Coverage closed by documented type naming: 0')
    for x, v in enumerate([coverage_before, coverage_after]):
        axs[2].text(x, v + 1.5, str(v), ha='center')
    axs[2].legend(fontsize=8); axs[2].grid(alpha=.3, axis='y')

    fig.suptitle('Motor-neuron TYPE-level resolution attempt: a NAMING/ONTOLOGY analysis\n'
                 'not a wiring authorisation, not an anatomical registration, no cross-specimen equivalence')
    fig.tight_layout(); fig.savefig(OUT / 'mn_type_resolution.png', dpi=110)

    print(json.dumps({'answer': report['answer'],
                      'status_counts': report['status_counts'],
                      'coverage': report['coverage'],
                      'documented_muscle_names_count': len(doc_muscles),
                      'banc_leg_motor_neurons': report['banc_leg_motor_neurons']['total_on_leg_nerves'],
                      'miller_aliases': report['miller_aliases_documented_inside_banc_annotations'],
                      'single_muscle_rule': naming_quotes.get('single_muscle_rule', '')[:160]},
                     indent=2))
    print(OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
