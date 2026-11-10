"""What is the whole-population readout actually reading?

Joins the 12000 tier cells to data/banc888/banc_888_meta.feather by banc_888_id (the
declared key; root_id leaves 439 of the 12000 unmatched and is only reported as a
control) and decomposes the readout weight mass over the annotation fields.

Cheap: reads READOUT_FIT.npz and the feather only.  No simulation.
"""
import numpy as np
import pyarrow.feather as f

import common as C

FIELDS = ['super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'nerve', 'region',
          'body_part_sensory', 'body_part_effector', 'cell_function', 'hemilineage',
          'neurotransmitter_predicted']


def log(msg):
    print(msg, flush=True)


def col(t, name):
    v = t.column(name).to_numpy(zero_copy_only=False)
    return np.array([('NA' if x is None else str(x)) for x in v], dtype=object)


def shares(values, energy, total, top=12):
    agg = {}
    for v, e in zip(values, energy):
        a = agg.setdefault(v, [0.0, 0])
        a[0] += float(e)
        a[1] += 1
    rows = [dict(value=k, energy_share=v[0] / total, count=v[1],
                 count_share=v[1] / len(values)) for k, v in agg.items()]
    rows.sort(key=lambda r: -r['energy_share'])
    out = rows[:top]
    rest = rows[top:]
    if rest:
        out.append(dict(value='__other_%d_values__' % len(rest),
                        energy_share=float(sum(r['energy_share'] for r in rest)),
                        count=int(sum(r['count'] for r in rest)),
                        count_share=float(sum(r['count_share'] for r in rest))))
    return out


def main():
    ids = np.load(C.IDENTITIES_NPZ)['root_ids']
    assert ids.shape == (12000,)
    t = f.read_table(C.META, memory_map=True)
    b888 = col(t, 'banc_888_id').astype(np.int64)
    root = col(t, 'root_id').astype(np.int64)
    assert len(b888) == len(root) == t.num_rows
    row_of = {}
    for k, v in enumerate(b888):
        row_of[int(v)] = k
    n_as_b888 = sum(1 for i in ids if int(i) in row_of)
    root_rows = {}
    for k, v in enumerate(root):
        root_rows.setdefault(int(v), []).append(k)
    n_as_root = sum(1 for i in ids if int(i) in root_rows)
    rows = [row_of.get(int(i), -1) for i in ids]
    missing = int(sum(1 for r in rows if r < 0))
    log('join by banc_888_id: %d/12000 matched, unmatched %d; by root_id only %d/12000'
        % (n_as_b888, missing, n_as_root))

    ann = {name: col(t, name) for name in FIELDS}

    ro = C.arrays(C.O / 'READOUT_FIT.npz')
    cells204 = np.asarray(ro['cells204'], int)
    is204 = np.zeros(12000, bool)
    is204[cells204] = True
    W204sum = (float(((ro['W_P2'][:, cells204]) ** 2).sum())
               / float((ro['W_P2'] ** 2).sum()))

    out = dict(
        source_meta=str(C.META), source_meta_sha256=C.sha(C.META),
        source_tier_identities=str(C.IDENTITIES_NPZ),
        join=dict(key='banc_888_id', matched=n_as_b888, unmatched=missing,
                  matched_if_key_were_root_id=n_as_root,
                  meta_rows=int(t.num_rows),
                  note='the declared key is banc_888_id; root_id alone cannot resolve 439 of the '
                       '12000 tier cells, so a root_id join would silently lose them'),
        fields=FIELDS,
        motor_cell_weight_share=dict(
            cells=204,
            P1_energy_share_on_the_204_motor_cells=float(
                ((ro['W_P1'][:, cells204]) ** 2).sum() / (ro['W_P1'] ** 2).sum()),
            P2_energy_share_on_the_204_motor_cells=W204sum,
            note='share of the squared readout weight mass that sits on the cells the current '
                 'readout already uses; the rest is on cells it never sees'),
        per_readout={})

    for name in ('P1', 'P2'):
        Wm = ro['W_' + name]
        energy = (Wm ** 2).sum(0)
        total = float(energy.sum())
        order = np.argsort(-energy)
        top50 = order[:50]
        profile = {fld: shares([ann[fld][rows[j]] if rows[j] >= 0 else 'NA' for j in range(12000)],
                               energy, total) for fld in FIELDS}
        top = []
        for rank, j in enumerate(top50, 1):
            r = rows[j]
            e = float(energy[j])
            top.append(dict(
                rank=rank, tier_index=int(j), banc_888_id=int(ids[j]),
                annotation=(dict({fld: ann[fld][r] for fld in FIELDS}, meta_row=int(r))
                            if r >= 0 else None),
                is_204_motor_cell=bool(is204[j]),
                energy=float((Wm[:, j] ** 2).sum()),
                energy_share=float(e / total),
                row_norm=float(np.sqrt((Wm[:, j] ** 2).sum())),
                cumulative_energy_share=float(np.cumsum(np.sort(energy)[::-1])[rank - 1] / total)))
        cum = np.cumsum(np.sort(energy)[::-1]) / total
        out['per_readout'][name] = dict(
            weight_energy_total=total,
            top50_shared_energy_share=float(cum[49]),
            top200_shared_energy_share=float(cum[199]),
            cells_carrying_50pct=int(np.searchsorted(cum, 0.5) + 1),
            cells_carrying_90pct=int(np.searchsorted(cum, 0.9) + 1),
            top50=top,
            profile_top50_count_share=shares(
                [ann['super_class'][rows[j]] if rows[j] >= 0 else 'NA' for j in top50],
                [1.0] * 50, 50.0),
            profile=profile,
            top50_cells_that_are_motor_class=int(sum(
                1 for j in top50 if rows[j] >= 0 and ann['super_class'][rows[j]] == 'motor')),
            top50_cells_that_are_sensory_super_classes=int(sum(
                1 for j in top50
                if rows[j] >= 0 and str(ann['super_class'][rows[j]]).startswith('sensory'))))

    # explicit verdict on the question the parent asked
    for name in ('P1', 'P2'):
        prof = {d['value']: d['energy_share'] for d in
                out['per_readout'][name]['profile']['super_class']}
        motor = prof.get('motor', 0.0)
        sensory = sum(v for k, v in prof.items() if str(k).startswith('sensory'))
        intrinsic = prof.get('central_brain_intrinsic', 0.0) + prof.get('ventral_nerve_cord_intrinsic', 0.0)
        ascend = prof.get('ascending', 0.0) + prof.get('descending', 0.0)
        out['per_readout'][name]['super_class_energy'] = dict(
            motor=motor, sensory=sensory, intrinsic=intrinsic, ascending_descending=ascend,
            classified_as=('MOTOR_LIKE' if motor >= 0.5 else
                           ('SENSORY_LIKE' if sensory >= 0.5 else 'MIXED')))
    C.dump(C.O / 'CELL_ANNOTATION.json', out)
    for name in ('P1', 'P2'):
        d = out['per_readout'][name]
        log('%s: top50 energy %.3f, 50%% of mass in %d cells, motor-class top50 cells %d, '
            'super_class energy %s'
            % (name, d['top50_shared_energy_share'], d['cells_carrying_50pct'],
               d['top50_cells_that_are_motor_class'], d['super_class_energy']))


if __name__ == '__main__':
    main()
