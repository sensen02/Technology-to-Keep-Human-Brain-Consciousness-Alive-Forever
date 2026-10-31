"""Strict saved-evidence consumer. Never certifies locking or regenerates phase."""
import hashlib
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), 'nonfinite numeric evidence')
    elif isinstance(value, dict):
        for v in value.values():
            _finite_tree(v)
    elif isinstance(value, list):
        for v in value:
            _finite_tree(v)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def consume_record(legacy, evidence=None, *, history_only=False, provenance=None):
    """Preserve legacy verbatim; history-only must be explicitly requested.

    provenance is a validated loader receipt, not a claim of physical truth.
    In-memory callers must supply their own source association receipt.
    """
    _require(isinstance(legacy, dict), 'legacy must be a dict')
    _finite_tree(legacy)
    if evidence is None:
        _require(history_only, 'new evidence missing; explicit history_only required')
        return dict(legacy=legacy, evidence_integrity='NOT_AVAILABLE',
                    metrology=None, locking_conclusion='NOT_ASSESSED',
                    mode='HISTORY_ONLY_NO_NEW_METROLOGY')
    _require(not history_only, 'history_only cannot silently ignore supplied evidence')
    _require(isinstance(evidence, dict), 'evidence must be an object')
    _require(isinstance(provenance, dict) and bool(provenance.get('source_id')),
             'source association receipt required')
    _finite_tree(evidence)
    try:
        _require(evidence['schema'] == 'phase-metrology-evidence-v1', 'unknown schema')
        _require(evidence['scope'] == 'ADDITIONAL_EVIDENCE_NOT_CERTIFICATION', 'invalid scope')
        m = evidence['measurement']
        t = m['time_s']; n = len(t)
        _require(n >= 2 and all(_number(v) for v in t), 'invalid times')
        _require(all(y > x for x, y in zip(t, t[1:])), 'non-increasing times')
        flags = m['phase_defined']
        _require(len(flags) == n and all(type(v) is bool for v in flags), 'invalid defined flags')
        _require(evidence['defined_at_samples'] == flags, 'defined flags mismatch')
        for channel in ('analytic_A', 'analytic_B'):
            for component in ('real', 'imag'):
                v = evidence[channel][component]
                _require(len(v) == n and all(_number(x) for x in v), 'invalid analytic array')
        zero = []
        for j in range(n):
            nonzero = all(evidence[c]['real'][j] != 0 or evidence[c]['imag'][j] != 0
                          for c in ('analytic_A', 'analytic_B'))
            _require(nonzero == flags[j], 'analytic zero/defined mismatch')
            if not nonzero: zero.append(j)
        _require(m['sampled_zero_indices'] == zero, 'zero indices mismatch')
        for field in ('principal_rad', 'product_unwrapped_rad', 'separate_unwrapped_rad',
                      'separate_minus_product_turns'):
            v = m[field]; _require(len(v) == n, 'path length mismatch')
            for j, x in enumerate(v):
                _require(_number(x) if flags[j] else x is None, 'invalid null/defined path')
                if field == 'separate_minus_product_turns' and flags[j]:
                    _require(x == int(x), 'noninteger turn difference')
        for field in ('legacy_product_rad', 'legacy_separate_reunwrapped_rad'):
            _require(len(m[field]) == n and all(_number(x) for x in m[field]), 'invalid legacy path')
        low = m['existing_low_amplitude']
        _require(low is None or (len(low) == n and all(type(x) is bool for x in low)), 'invalid legacy amplitude flags')
        _require(m['legacy_statistics_defined_at_samples'] is all(flags), 'statistics domain mismatch')
        for field in ('product_statistics', 'separate_statistics', 'legacy_product_statistics',
                      'legacy_separate_reunwrapped_statistics'):
            v = m[field]
            if field in ('product_statistics', 'separate_statistics') and zero:
                _require(v is None, 'cross-zero statistics must be null')
            else:
                _require(isinstance(v, dict) and _number(v['advance_deg'])
                         and type(v['halfturn_count']) is int and v['halfturn_count'] >= 0,
                         'invalid endpoint statistic')
                _require(v['halfturn_count'] == math.floor(abs(v['advance_deg'])/180),
                         'endpoint count/formula mismatch')
        turns = m['separate_minus_product_turns']; expected = []
        for j in range(n-1):
            if flags[j] and flags[j+1] and turns[j] != turns[j+1]:
                expected.append(dict(start_index=j, end_index=j+1, start_s=t[j], end_s=t[j+1],
                                     change_turns=int(turns[j+1]-turns[j])))
        _require(m['branch_change_intervals'] == expected, 'branch interval ledger mismatch')
        disagrees = any(x != 0 for x in turns if x is not None)
        status = ('UNDEFINED_AT_SAMPLED_ZERO' if zero else
                  'PATH_DISAGREEMENT_CONTINUOUS_BRANCH_UNIDENTIFIED' if disagrees else
                  'PATHS_AGREE_CONTINUOUS_BRANCH_UNIDENTIFIED')
        _require(m['status'] == status, 'unknown or inconsistent state')
        _require(m['continuous_branch_identified'] is False, 'unsupported identified branch claim')
        _require(m['locking_status'] == 'NOT_ASSESSED', 'unknown or unsupported locking state')
    except (KeyError, TypeError, IndexError, OverflowError) as e:
        raise EvidenceError('missing or malformed evidence: '+str(e)) from e
    return dict(legacy=legacy, mode='SAVED_METROLOGY_EVIDENCE',
                provenance=provenance, evidence_integrity='VALIDATED_STRUCTURE_AND_ASSOCIATION',
                metrology=m, defined_at_samples=flags,
                locking_conclusion='NOT_ASSESSED')


def load_saved_case(root, implementation, case):
    """Verify immutable manifest and bind named artifact to saved integration row."""
    root = Path(root)
    try:
        manifest = json.loads((root/'MANIFEST_SHA256.json').read_text())
        def read(name):
            _require(name in manifest, 'artifact missing from manifest')
            data = (root/name).read_bytes()
            _require(hashlib.sha256(data).hexdigest() == manifest[name], 'artifact hash mismatch')
            return json.loads(data)
        results = read('RESULTS.json')
        rows = [r for r in results['cases'] if r['implementation'] == implementation and r['case'] == case]
        _require(len(rows) == 1, 'case association missing or ambiguous')
        row = rows[0]; filename = implementation+'_'+case+'_EVIDENCE.json'
        ev = read(filename)
        _require(ev['measurement']['status'] == row['status'], 'case state association mismatch')
        _require(ev['measurement']['branch_change_intervals'] == row['branch_intervals'], 'case interval association mismatch')
        old_stat = ('legacy_separate_reunwrapped_statistics' if implementation == 'main'
                    else 'legacy_product_statistics')
        _require(ev['measurement'][old_stat]['halfturn_count'] == row['legacy_slips'], 'legacy count association mismatch')
        legacy = dict(saved_integration_row=row, endpoint_statistics=ev['measurement'][old_stat],
                      original_certification_result=None, historical_locking_label=None,
                      unavailable_fields_reason='not stored in source integration artifact; not inferred from zero slip count')
        return consume_record(legacy, ev, provenance=dict(source_id=str(root/filename),
                              evidence_sha256=manifest[filename], results_sha256=manifest['RESULTS.json']))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise EvidenceError('saved artifact load failed: '+str(e)) from e
