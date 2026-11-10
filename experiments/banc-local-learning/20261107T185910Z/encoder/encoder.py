"""Identity-qualified stateless physical sensory transduction; no simulator imports.

All gains and common within-class tuning are uncalibrated engineering choices.
Source cell annotations establish identity/modality candidates, not tuning curves.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import numpy as np

PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE = PROJECT / 'experiments/flybody-retarget/20261107T182940Z/neural-interface'
MAPPING = PROJECT / 'experiments/flybody-teacher/20261107T180844Z/mapping'
FOOT_ORDER = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
BODY_WEIGHT_NATIVE = 24.446354755878005
NERVES = dict(zip(FOOT_ORDER, (
    'left_prothoracic_leg_nerve', 'left_mesothoracic_leg_nerve',
    'left_metathoracic_leg_nerve', 'right_prothoracic_leg_nerve',
    'right_mesothoracic_leg_nerve', 'right_metathoracic_leg_nerve')))
CLASSES = {
    'claw_position': 'chordotonal_organ_neuron',
    'hook_direction': 'chordotonal_organ_neuron',
    'hairplate_angle': 'hair_plate_neuron',
    'campaniform_strain': 'campaniform_sensillum_neuron',
    'club_vibration': 'chordotonal_organ_neuron',
    'bristle_touch': 'bristle_neuron',
    'multidendritic_noxious': 'multidendritic_neuron',
}
SUFFIXES = {
    'claw_position': ('claw_chordotonal_organ_neuron',),
    'hook_direction': ('hook_chordotonal_organ_neuron',),
    'hairplate_angle': ('hair_plate_neuron',),
    'campaniform_strain': ('campaniform_sensillum_neuron', 'bilateral_campaniform_sensillum_neuron'),
    'club_vibration': ('club_chordotonal_organ_neuron',),
    'bristle_touch': ('bristle_neuron',),
    'multidendritic_noxious': ('multidendritic_neuron',),
}
DISABLED = {
    'multidendritic_noxious': 'no_calibrated_damage_input',
    'bristle_touch': 'no_individual_hair_collision_input',
    'club_vibration': 'no_acceleration_input',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _readonly(a):
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


def qualify_rows(rows, cache, groups, local_roots, local_cache, ambiguous_roots):
    """Fail closed per source record; return admitted records and all exclusions.

    Every repeated source identity is rejected, even if the duplicate agrees.
    No fallback to legacy IDs, alternate leg, inferred modality or nearest index.
    """
    root_counts = Counter(r['root_888'] for r in rows)
    index_counts = Counter(int(r['cache_index_0based']) for r in rows if int(r['cache_index_0based']) >= 0)
    local_lookup = {int(x): i for i, x in enumerate(local_cache)}
    union = set(map(int, groups['leg_sensory_nerve_union']))
    group_sets = {k: set(map(int, v)) for k, v in groups.items()}
    qualified, exclusions = [], []
    for row in rows:
        root = row['root_888']
        ci = int(row['cache_index_0based'])
        leg = row['leg'].upper()
        channel = row['channel']
        li = local_lookup.get(ci)
        reasons = []
        if root in ambiguous_roots:
            reasons.append('review4_modality_ambiguity')
        if root_counts[root] > 1 or (ci >= 0 and index_counts[ci] > 1):
            reasons.append('reused_source_identity_or_cache_index')
        if not root.isdecimal():
            reasons.append('invalid_root888_decimal_identity')
        if ci < 0 or ci >= len(cache['root_ids']):
            reasons.append('missing_cache_identity')
        else:
            if str(int(cache['root_ids'][ci])) != root:
                reasons.append('root888_cache_identity_mismatch')
            if li is None:
                reasons.append('outside_local_candidate')
            elif str(int(local_roots[li])) != root:
                reasons.append('root888_local_identity_mismatch')
            if ci not in union:
                reasons.append('outside_leg_sensory_nerve_union')
            nerve = str(cache['ann_nerve'][ci])
            matched = [x for x, token in NERVES.items() if token in nerve.split(',')]
            if matched != [leg]:
                reasons.append('cache_nerve_source_leg_mismatch_or_ambiguity')
            if ci not in group_sets.get('leg_sensory_' + leg.lower(), set()):
                reasons.append('outside_matching_leg_group')
            if ci not in group_sets.get('sensory_channel__' + leg.lower() + '__' + channel, set()):
                reasons.append('outside_matching_source_channel_group')
            if str(cache['ann_class'][ci]) != row['cache_class']:
                reasons.append('csv_cache_class_contradiction')
            if row['cache_class'] != CLASSES.get(channel):
                reasons.append('channel_source_class_contradiction')
            if str(cache['ann_super_class'][ci]) != row['cache_super_class'] or row['cache_super_class'] not in ('sensory', 'sensory_ascending', 'sensory_descending'):
                reasons.append('sensory_superclass_contradiction')
        region = {'F': 'front', 'M': 'middle', 'H': 'hind'}.get(leg[-1:], '')
        expected_subclasses = [region + '_leg_' + suffix for suffix in SUFFIXES.get(channel, ())]
        if leg not in FOOT_ORDER or row['cell_sub_class'] not in expected_subclasses:
            reasons.append('source_subclass_leg_or_modality_contradiction')
        if channel not in CLASSES:
            reasons.append('unknown_channel')
        if channel in DISABLED:
            reasons.append(DISABLED[channel])
        record = dict(row, root_888=root, cache_index_0based=ci, local_index_0based=li,
                      leg=leg, reasons=reasons)
        if 0 <= ci < len(cache['root_ids']):
            record['cache_nerve'] = str(cache['ann_nerve'][ci])
        if reasons:
            exclusions.append(record)
        else:
            record['reasons'] = ['exact_root888_cache_and_local_identity', 'unique_matching_leg_nerve',
                                 'leg_sensory_union_and_channel_group_member',
                                 'matching_source_subclass_and_cache_class',
                                 'admitted_uncalibrated_engineering_modality_proxy']
            qualified.append(record)
    return qualified, exclusions


class SensoryEncoder:
    """SensoryEncoder() -> encode(q42_rad, qvel42_rad_s, foot_load6_native).

    Output is a new float64 (20000,) vector in LOCAL_MODULE_CANDIDATE order.
    Inputs must have exact shapes (42,), (42,), (6,) and finite real values.
    Negative loads are clamped to zero; angles clamp to recorded joint bounds.
    Construction verifies frozen source hashes; no simulator or neural dynamics.
    """
    foot_order = FOOT_ORDER
    body_weight_native = BODY_WEIGHT_NATIVE

    def __init__(self):
        manifest_path = SOURCE / 'NEURAL_INTERFACE_MANIFEST.json'
        manifest = json.loads(manifest_path.read_text())
        cache_path = Path(manifest['sources']['cache']['path'])
        source_paths = {name: SOURCE / name for name in (
            'LOCAL_MODULE_CANDIDATE.npz', 'GROUP_CACHE_INDICES.npz',
            'SENSORY_CHANNEL_CANDIDATES.csv', 'SENSORY_CHANNEL_REVIEW.json')}
        source_paths.update(manifest=manifest_path, cache=cache_path,
                            student_interface=MAPPING / 'STUDENT_INTERFACE.json',
                            joint_ranges=MAPPING / 'current42_candidates.csv',
                            illustrative_physiology=PROJECT / 'engine/neural_cond.py')
        self.source_contract = {k: {'path': str(p), 'sha256': sha256(p)} for k, p in source_paths.items()}
        for name in ('LOCAL_MODULE_CANDIDATE.npz', 'GROUP_CACHE_INDICES.npz', 'SENSORY_CHANNEL_CANDIDATES.csv'):
            if self.source_contract[name]['sha256'] != manifest['artifacts'][name]['sha256']:
                raise ValueError('Source hash differs from manifest: ' + name)
        if self.source_contract['cache']['sha256'] != manifest['sources']['cache']['sha256']:
            raise ValueError('Cache hash differs from manifest')
        with np.load(source_paths['LOCAL_MODULE_CANDIDATE.npz'], allow_pickle=False) as a:
            roots, indices = a['root_ids'].copy(), a['cache_indices'].copy()
        with np.load(source_paths['GROUP_CACHE_INDICES.npz'], allow_pickle=False) as a:
            groups = {k: a[k] for k in a.files}
        with np.load(cache_path, allow_pickle=False) as a:
            cache = {k: a[k] for k in ('root_ids', 'ann_class', 'ann_super_class', 'ann_nerve')}
        if roots.shape != (20000,) or indices.shape != (20000,) or len(np.unique(roots)) != 20000 or len(np.unique(indices)) != 20000:
            raise ValueError('Candidate must contain 20000 unique identities and indices')
        if np.any(indices < 0) or np.any(indices >= len(cache['root_ids'])) or not np.array_equal(cache['root_ids'][indices], roots):
            raise ValueError('Candidate root/cache mapping differs')
        self.root_ids = _readonly(roots)
        self.cache_indices = _readonly(indices)
        rows = _read_csv(source_paths['SENSORY_CHANNEL_CANDIDATES.csv'])
        if set(r['channel'] for r in rows) != set(CLASSES):
            raise ValueError('Expected exact seven source channel names')
        review = json.loads(source_paths['SENSORY_CHANNEL_REVIEW.json'].read_text())
        ambiguous = {r['root_id_string'] for r in review['ambiguous_records']}
        if len(ambiguous) != 4:
            raise ValueError('Expected explicit review4 identities')
        self.qualified_records, self.exclusions = qualify_rows(rows, cache, groups, roots, indices, ambiguous)
        self.local_indices = _readonly(sorted(r['local_index_0based'] for r in self.qualified_records))
        self.qualified_count = len(self.local_indices)
        self.allowed_mask = _readonly(np.isin(np.arange(20000), self.local_indices))
        joints = sorted(_read_csv(source_paths['joint_ranges']), key=lambda r: int(r['current_action_index']))
        interface = json.loads(source_paths['student_interface'].read_text())
        names = tuple(interface['output']['ordered_joint_names'])
        if len(names) != 42 or len(set(names)) != 42 or [int(r['current_action_index']) for r in joints] != list(range(42)) or tuple(r['current_joint'] for r in joints) != names:
            raise ValueError('Actual action order/range identity disagreement')
        self.joint_names = names
        self.joint_ranges_rad = _readonly(np.array([json.loads(r['range_rad']) for r in joints], dtype=float))
        if self.joint_ranges_rad.shape != (42, 2) or not np.isfinite(self.joint_ranges_rad).all() or np.any(np.diff(self.joint_ranges_rad, axis=1) <= 0):
            raise ValueError('Joint ranges must be finite and strictly ordered')
        self.knee_indices = _readonly([names.index('joint_' + leg + 'Tibia_pitch') for leg in FOOT_ORDER])
        self.coxa_indices = _readonly([[names.index('joint_' + leg + 'Coxa_' + axis) for axis in ('yaw', 'pitch', 'roll')] for leg in FOOT_ORDER])
        ports = defaultdict(list)
        for r in self.qualified_records:
            ports[(r['leg'], r['channel'])].append(r['local_index_0based'])
        self.channel_local_indices = {leg: {ch: _readonly(ports[(leg, ch)]).astype(np.int64) for ch in CLASSES} for leg in FOOT_ORDER}
        self._ports = tuple((FOOT_ORDER.index(leg), ch, _readonly(np.array(port, dtype=np.int64))) for (leg, ch), port in ports.items() if port)
        self.feature_contract = {
            'status': 'UNCALIBRATED_ENGINEERING_MODALITY_PROXIES',
            'foot_order': list(FOOT_ORDER), 'q_order': list(names),
            'joint_ranges_rad': self.joint_ranges_rad.tolist(),
            'knee_action_indices_in_foot_order': self.knee_indices.tolist(),
            'coxa_yaw_pitch_roll_action_indices_in_foot_order': self.coxa_indices.tolist(),
            'claw_position': '0.01+0.04*normalized_clamped_tibia_pitch',
            'hairplate_angle': '0.01+0.04*mean(normalized_clamped_coxa_yaw_pitch_roll)',
            'hook_direction': '0.05*tanh(abs(tibia_pitch_velocity_rad_s)/10); no direction tuning inferred',
            'campaniform_strain': '0.05*tanh(max(foot_load_native,0)/(body_weight_native/6))',
            'body_weight_native': BODY_WEIGHT_NATIVE, 'body_weight_source': 'explicit task supplied original body weight; absolute SI force NOT_ESTABLISHED',
            'parameters': {'angle_baseline_nA': 0.01, 'angle_span_nA': 0.04, 'speed_load_span_nA': 0.05, 'speed_scale_rad_s': 10.0, 'hard_current_bounds_nA': [0.0, 0.06]},
            'parameter_status': 'engineering illustrative; not measured fly physiology; no fit or tuning search',
            'disabled_channels': DISABLED,
            'within_class_tuning': 'identical per leg/channel; increasing coordinate is engineering convention, not biological direction knowledge',
            'tonic': 'only qualified claw/hairplate angle cells receive declared 0.01nA baseline',
            'unused_joint_features': 'trochanter angles and speeds, all coxa speeds',
            'state': 'none; no time/root/phase/teacher/targets/history input',
        }

    @staticmethod
    def _vector(value, shape, name):
        raw = np.asarray(value)
        if raw.shape != shape or raw.dtype.kind not in 'fiu':
            raise ValueError(name + ' must be a real numeric array of shape ' + str(shape))
        a = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(name + ' must be finite')
        return a

    def normalize_angles(self, q42_rad):
        q = self._vector(q42_rad, (42,), 'q42_rad')
        lo, hi = self.joint_ranges_rad.T
        # Clip first so extreme but finite inputs cannot overflow subtraction.
        return (np.clip(q, lo, hi) - lo) / (hi - lo)

    def encode(self, q42_rad, qvel42_rad_s, foot_load6_native):
        qn = self.normalize_angles(q42_rad)
        v = self._vector(qvel42_rad_s, (42,), 'qvel42_rad_s')
        load = self._vector(foot_load6_native, (6,), 'foot_load6_native')
        features = {
            'claw_position': 0.01 + 0.04 * qn[self.knee_indices],
            'hairplate_angle': 0.01 + 0.04 * qn[self.coxa_indices].mean(axis=1),
            'hook_direction': 0.05 * np.tanh(np.abs(v[self.knee_indices]) / 10.0),
            'campaniform_strain': 0.05 * np.tanh(np.maximum(load, 0.0) / (BODY_WEIGHT_NATIVE / 6.0)),
        }
        currents = np.zeros(20000, dtype=np.float64)
        for leg_index, channel, port in self._ports:
            currents[port] = features[channel][leg_index]
        np.clip(currents, 0.0, 0.06, out=currents)
        return currents

    def contract(self):
        return {
            'schema': 'banc-local-sensory-contract/1', 'api': 'SensoryEncoder().encode(q42_rad,qvel42_rad_s,foot_load6_native)->float64(20000,) currents_nA',
            'feature_contract': self.feature_contract, 'source_contract': self.source_contract,
            'qualified_count': self.qualified_count, 'local_indices': self.local_indices.tolist(),
            'qualified_root888_ids': [str(int(self.root_ids[i])) for i in self.local_indices],
            'qualified_records': self.qualified_records, 'exclusions': self.exclusions,
            'exclusion_reason_counts': dict(Counter(reason for r in self.exclusions for reason in r['reasons'])),
            'channel_leg_counts': {leg: {ch: len(a) for ch, a in channels.items()} for leg, channels in self.channel_local_indices.items()},
            'local_candidate_root_ids': [str(int(x)) for x in self.root_ids],
            'runtime_provenance': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'platform': platform.platform(), 'encoder_path': str(Path(__file__).resolve()), 'encoder_sha256': sha256(__file__), 'simulator_required': False, 'neural_steps': 0, 'physical_steps': 0, 'training_updates': 0},
            'limitations': ['Annotation-qualified engineering proxies, not calibrated cell tuning or electrophysiology.', 'No acceleration/hair-collision/damage channels.', 'Parent circuit/readout/physical evaluation is separate; failed slow4 is not a success label.'],
        }


if __name__ == '__main__':
    encoder = SensoryEncoder()
    out = Path(__file__).resolve().parent
    (out / 'CONTRACT.json').write_text(json.dumps(encoder.contract(), indent=2, ensure_ascii=False) + '\n')
    (out / 'EXCLUSIONS.json').write_text(json.dumps(encoder.exclusions, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'qualified_count': encoder.qualified_count, 'exclusions': len(encoder.exclusions), 'channel_leg_counts': encoder.contract()['channel_leg_counts']}))
