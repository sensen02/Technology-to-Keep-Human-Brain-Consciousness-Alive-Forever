#!/usr/bin/env python3
"""Bounded real-FAFB passive cable demonstration; never repair source geometry.

Only selected candidate ZIP members are streamed, never the whole archive.
Candidate screening is in memory; only a passing sample is written verbatim.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from engine.cable import Morphology, CableNeuron, DN_PASSIVE, swc_to_morphology

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/flywire'
OUT = ROOT / 'outputs/brain_isolation'
PREFIX = OUT / 'real_morphology'
MAX_NODES = 2500
MAX_BYTES = 160000
CAPS = [('descending', 64), ('visual_projection', 128), ('optic', 384)]
CAVEATS = [
    'Real FAFB skeleton; the computational edge cut is NOT exact neck anatomy.',
    'FAFB is brain-only/truncated; no whole-CNS or neck continuation is established.',
    'Passive parameters borrowed from DNp01/DNp03 (PMC11071487), not whole-fly calibration or a fit to this cell.',
    'Injection and recording sites are computational probes; no real synapse positions are claimed.',
    'Upstream/downstream mean root-side/subtree-side topology, not biological signal direction.',
    'No cross-BANC identity or geometry match; only same-FAFB annotation/connectome ID membership.',
    'SWC radius estimates are accepted as supplied, not independently calibrated; no repair, resampling or radius clamping.',
    'Sealed cut removes one axial conductance without changing membrane/geometry; no tissue injury biochemistry.',
]


def annotation_table(path):
    with gzip.open(path, 'rt', newline='') as f:
        return {r['root_id']: r for r in csv.DictReader(f)}


def screen(raw, source_id):
    """Strict in-memory screening, followed by official importer on saved sample."""
    text = raw.decode('utf-8', errors='strict')
    headers = [s for s in text.splitlines() if s.lstrip().startswith('#')]
    meta = [json.loads(s.split('Meta:', 1)[1]) for s in headers if 'Meta:' in s]
    if len(meta) != 1 or str(meta[0].get('id')) != source_id:
        raise ValueError('missing/ambiguous/mismatched source ID in Meta header')
    if meta[0].get('units') != '1 nanometer':
        raise ValueError('expected explicit source units: 1 nanometer')
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f'line {lineno}: expected seven columns')
        try:
            nid, kind, par = int(fields[0]), int(fields[1]), int(fields[6])
            x, y, z, radius = map(float, fields[2:6])
        except ValueError as exc:
            raise ValueError(f'line {lineno}: invalid numeric field') from exc
        if nid < 0:
            raise ValueError(f'line {lineno}: negative node ID')
        rows.append((nid, kind, x, y, z, radius, par))
    if not 100 <= len(rows) <= MAX_NODES:
        raise ValueError(f'node count {len(rows)} outside [100, {MAX_NODES}]')
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate node IDs')
    idx = {nid: i for i, nid in enumerate(ids)}
    missing = sorted({r[6] for r in rows} - set(ids) - {-1})
    if missing:
        raise ValueError(f'missing parents: {missing[:20]}')
    radii = np.array([r[5] for r in rows])
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError(f'invalid radii: zero={int((radii == 0).sum())}, negative={int((radii < 0).sum())}, nonfinite={int((~np.isfinite(radii)).sum())}')
    xyz = np.array([r[2:5] for r in rows]) * .001
    parent = np.array([idx[r[6]] if r[6] != -1 else -1 for r in rows])
    m = Morphology(*xyz.T, 2 * radii * .001, parent)
    return m, ids, headers


def select(result):
    cls = annotation_table(DATA / 'fafb_classification.csv.gz')
    neurons = annotation_table(DATA / 'fafb_neurons.csv.gz')
    with np.load(DATA / 'fafb_connectome.npz', allow_pickle=False) as f:
        conn_ids = {str(int(x)) for x in f['root_ids']}
    result['selection'] = {'order': CAPS, 'sort': 'uncompressed size, filename; evenly spaced size ranks per class if over cap',
                           'min_member_bytes': 6000, 'max_member_bytes': MAX_BYTES,
                           'max_nodes': MAX_NODES, 'max_candidates': sum(c for _, c in CAPS),
                           'attempts': [], 'eligible_by_class': {}, 'streamed_bytes': 0}
    s = result['selection']
    with zipfile.ZipFile(DATA / 'fafb_skeletons_swc.zip') as z:
        infos = z.infolist()
        s['central_directory_members'] = len(infos)
        for category, cap in CAPS:
            candidates = sorted((i for i in infos if i.filename.endswith('.swc')
                                 and 6000 <= i.file_size <= MAX_BYTES
                                 and i.filename[:-4] in conn_ids and i.filename[:-4] in neurons
                                 and cls.get(i.filename[:-4], {}).get('super_class') == category),
                                key=lambda i: (i.file_size, i.filename))
            s['eligible_by_class'][category] = len(candidates)
            ranks = np.unique(np.linspace(0, len(candidates)-1, min(cap, len(candidates)), dtype=int)) if candidates else []
            for rank in ranks:
                info = candidates[int(rank)]
                sid = info.filename[:-4]
                evidence = {'member': info.filename, 'source_id': sid, 'super_class': category,
                            'size_rank': int(rank), 'uncompressed_bytes': info.file_size,
                            'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}',
                            'classification': cls[sid], 'neuron_annotation': neurons[sid],
                            'same_fafb_classification_neurons_connectome_id_match': True}
                s['attempts'].append(evidence)
                try:
                    with z.open(info) as stream:
                        raw = stream.read(MAX_BYTES + 1)
                    s['streamed_bytes'] += len(raw)
                    if len(raw) != info.file_size:
                        raise ValueError('incomplete or oversized stream')
                    evidence['sha256'] = hashlib.sha256(raw).hexdigest()
                    m, ids, headers = screen(raw, sid)
                except (ValueError, UnicodeError, zipfile.BadZipFile) as exc:
                    evidence.update(status='rejected', reason=str(exc))
                    continue
                sample = OUT / 'real_morphology_sample.swc'
                sample.write_bytes(raw)
                imported, imported_ids = swc_to_morphology(sample, units='nm')
                assert imported_ids == ids
                for field in ('x', 'y', 'z', 'd', 'parent'):
                    assert np.array_equal(getattr(m, field), getattr(imported, field))
                evidence['status'] = 'accepted'
                result['source'] = {**evidence, 'archive': str(DATA / 'fafb_skeletons_swc.zip'),
                    'sample': str(sample), 'headers': headers, 'source_units': 'nm', 'engine_units': 'um',
                    'coordinate_and_radius_scale': .001, 'classification': cls[sid],
                    'neuron_annotation': neurons[sid], 'connectome_id_match': True,
                    'classification_id_match': True, 'node_count': m.n, 'repair_applied': False,
                    'coordinate_range_um': [np.min(np.c_[m.x,m.y,m.z], axis=0).tolist(), np.max(np.c_[m.x,m.y,m.z], axis=0).tolist()],
                    'diameter_range_um': [float(m.d.min()), float(m.d.max())]}
                return imported, ids
    return None


def simulate(m, ids, result):
    children = m.children()
    root = int(np.flatnonzero(m.parent == -1)[0])
    order, depth = [root], np.zeros(m.n, dtype=int)
    for i in order:
        for c in children[i]:
            depth[c] = depth[i] + 1
            order.append(c)
    sizes = np.ones(m.n, dtype=int)
    for i in reversed(order[1:]):
        sizes[m.parent[i]] += sizes[i]
    eligible = [i for i in range(m.n) if m.parent[i] >= 0 and m.parent[m.parent[i]] >= 0 and children[i] and 5 <= sizes[i] <= m.n-5]
    if not eligible:
        raise ValueError('valid morphology has no suitable interior cut')
    cut = min(eligible, key=lambda i: (abs(int(sizes[i])-m.n/2), ids[i]))
    up = int(m.parent[cut]); down = cut
    subtree = [cut]
    for i in subtree:
        subtree.extend(children[i])
    dt, duration, amplitude = .025, 20., .001
    t = np.arange(1, round(duration/dt)+1)*dt
    pulse = (np.arange(len(t))*dt >= 2.) & (np.arange(len(t))*dt < 10.)
    traces, nnz = {}, {}
    for variant in ('intact', 'sealed_cut'):
        for stimulus in ('probe', 'noinput'):
            neuron = CableNeuron(m, dt_ms=dt)
            if variant == 'sealed_cut':
                neuron.cut_edges([cut])
                assert neuron.g_ax[cut] == 0 and np.all(neuron.g_end == 0)
            nnz[variant] = neuron.G.nnz
            v = np.empty((len(t), 2))
            current = np.zeros(m.n)
            for k in range(len(t)):
                current[up] = amplitude if stimulus == 'probe' and pulse[k] else 0.
                v[k] = neuron.step(i_inject=current)[[up, down]]
            assert np.isfinite(v).all()
            traces[f'{variant}_{stimulus}'] = v
    delta = {v: traces[f'{v}_probe'] - traces[f'{v}_noinput'] for v in ('intact', 'sealed_cut')}
    metrics = {}
    for v, dv in delta.items():
        metrics[v] = {f'{site}_peak_delta_mV': float(dv[:,j].max()) for j, site in enumerate(('upstream','downstream'))}
        metrics[v]['max_noinput_drift_mV'] = float(np.max(abs(traces[f'{v}_noinput']-DN_PASSIVE['E_leak_mV'][0])))
    metrics['downstream_peak_suppression_fraction'] = 1 - metrics['sealed_cut']['downstream_peak_delta_mV']/metrics['intact']['downstream_peak_delta_mV']
    metrics['upstream_peak_cut_minus_intact_mV'] = metrics['sealed_cut']['upstream_peak_delta_mV']-metrics['intact']['upstream_peak_delta_mV']
    assert metrics['intact']['downstream_peak_delta_mV'] > 1e-8
    assert abs(metrics['sealed_cut']['downstream_peak_delta_mV']) < 1e-9
    assert all(v['max_noinput_drift_mV'] < 1e-6 for v in [metrics['intact'], metrics['sealed_cut']])
    assert metrics['upstream_peak_cut_minus_intact_mV'] > 0
    result['simulation'] = {'dt_ms':dt, 'duration_ms':duration, 'inward_probe_nA':amplitude,
        'pulse_interval_ms':'[2,10)', 'passive_parameters': DN_PASSIVE,
        'cut_child_row':cut, 'cut_parent_row':up, 'cut_child_swc_id':ids[cut], 'cut_parent_swc_id':ids[up],
        'injection_row':up, 'recording_rows':[up, down], 'downstream_nodes':len(subtree),
        'upstream_nodes':m.n-len(subtree), 'cut_timing':'before simulation in separate instance',
        'sealed_end_added_conductance_uS':0., 'sparse_G_nnz':nnz,
        'storage':'O(nodes) sparse matrices; only two recording sites stored per run',
        'metrics':metrics, 'tests':{'strict_import_verified':True, 'finite_traces':True,
        'noinput_drift_below_1e-6_mV':True, 'intact_transmission_positive':True,
        'sealed_downstream_below_1e-9_mV':True, 'upstream_response_increases_after_sealing':True}}
    np.savez_compressed(str(PREFIX)+'_traces.npz', time_ms=t, pulse=pulse, **traces,
                        intact_delta_mV=delta['intact'], sealed_cut_delta_mV=delta['sealed_cut'])
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(121, projection='3d')
    xyz = np.c_[m.x,m.y,m.z]
    downstream = set(subtree)
    for is_down, color, label in [(False,'#3066be','root-side'),(True,'#e68a23','subtree-side')]:
        edges = [i for i in range(m.n) if m.parent[i]>=0 and (i in downstream)==is_down and i!=cut]
        segments = np.array([[xyz[m.parent[i]],xyz[i]] for i in edges])
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=.8, label=label))
    ax.plot(*xyz[[up,cut]].T, color='crimson', lw=3, ls='--', label='removed axial edge')
    ax.scatter(*xyz[up],color='black',s=40,label='inward probe / upstream')
    ax.scatter(*xyz[cut],color='crimson',s=45,marker='x',label='downstream')
    if 'truncation' in result:
        index_by_id = {nid:i for i,nid in enumerate(ids)}
        boundary_rows = sorted({index_by_id[b['retained_id']] for b in result['truncation']['artificial_sealed_boundary_edges']})
        ax.scatter(*xyz[boundary_rows].T, color='purple', s=20, marker='s', alpha=.65, label='artificial subtree boundaries')
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='FAFB X (um)',ylabel='FAFB Y (um)',zlabel='FAFB Z (um)')
    ax.set_box_aspect(np.maximum(hi-lo,1))
    ax.set_title(f"{result.get('geometry_label', 'Real FAFB morphology')}: {m.n} nodes\ncomputational cut, not exact neck anatomy")
    ax.legend(loc='upper left', fontsize=8)
    for j, site in enumerate(('Upstream (root-side)', 'Downstream (subtree-side)')):
        a = fig.add_subplot(2,2,2+2*j)
        for v, color in [('intact','#3066be'),('sealed_cut','#d44331')]:
            a.plot(t, delta[v][:,j], label=v.replace('_',' '), color=color)
        a.axvspan(2,10,color='grey',alpha=.12,label='+1 pA inward probe')
        a.set(xlabel='Time (ms)',ylabel='Probe - matched noinput (mV)',title=site)
        a.legend(fontsize=8); a.grid(alpha=.2)
    src = result['source']
    fig.suptitle(f"FAFB root ID {src['source_id']} | {src['classification']['super_class']} / {src['classification']['sub_class']}",fontsize=13)
    fig.text(.03,.035,result.get('figure_caption', 'Real FAFB skeleton, not exact neck anatomy. Passive parameters borrowed from DNp01/DNp03, not whole-fly calibration.\nNo real synapse positions claimed. nm to um; original topology/radii unchanged. Separate sealed-cut and matched noinput runs.'),fontsize=10)
    fig.tight_layout(rect=(0,.095,1,.94))
    fig.savefig(str(PREFIX)+'_demo.png',dpi=170)
    plt.close(fig)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'status':'searching', 'caveats':CAVEATS, 'implementation':'manually implemented and tested prototype, not a calibrated biological product'}
    selected = select(result)
    if selected is None:
        result['status'] = 'no_valid_sample_in_bounded_search'
        result['exact_evidence'] = 'All streamed candidate rejection reasons are in selection.attempts; no silent geometry repair and no simulation claimed. This is not an archive-wide impossibility claim.'
        fig, ax = plt.subplots(figsize=(12,5)); ax.axis('off')
        from collections import Counter
        reasons = Counter(a['reason'] for a in result['selection']['attempts'])
        categories = Counter('node_limit' if a['reason'].startswith('node count') else a['reason'].split(':')[0] for a in result['selection']['attempts'])
        result['selection']['rejection_categories'] = dict(categories)
        result['tests'] = {'bounded_search_completed': True, 'no_repairs_applied': True,
                           'real_simulation_executed': False, 'all_candidates_rejected_with_reasons': all(a['status']=='rejected' and a['reason'] for a in result['selection']['attempts'])}
        ax.text(.02,.95,'No strictly valid real FAFB sample in bounded search\n\n'+f"Attempted {len(result['selection']['attempts'])} candidates; streamed {result['selection']['streamed_bytes']} bytes.\n"+f"Aggregate: {dict(categories)}\n\nMost common exact rejection reasons:\n"+'\n'.join(f'{n} x {r}' for r,n in reasons.most_common(5))+'\n\nNo repaired skeleton, no simulated traces, no exact neck anatomy claim.',va='top',fontsize=12)
        fig.savefig(str(PREFIX)+'_demo.png',dpi=170); plt.close(fig)
    else:
        simulate(*selected, result)
        result['status'] = 'passed'
    result['elapsed_seconds'] = time.monotonic()-start
    metrics = Path(str(PREFIX)+'_metrics.json')
    result['outputs'] = sorted({str(p) for p in OUT.glob('real_morphology*')} | {str(metrics)})
    metrics.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'], 'attempts':len(result['selection']['attempts']), 'elapsed_seconds':result['elapsed_seconds'], 'metrics':str(metrics), 'source':result.get('source',{}).get('source_id'), 'simulation':result.get('simulation',{}).get('metrics')}, indent=2))

if __name__ == '__main__':
    main()
