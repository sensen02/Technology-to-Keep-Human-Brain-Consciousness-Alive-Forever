"""Finish the gain validation honestly: record the crash, apply the frozen rule, and
quantify why reduced gain collapses the body.

No new physics is run here; everything is computed from runs already saved by run.py.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib
from pathlib import Path
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-readout-gain/20261106T024500Z')
ARCH = Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
SRC = Path(__file__).resolve().parent


def js(p):
    return json.loads(Path(p).read_text())


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    GAINS = [1.0, 0.25, 0.1, 0.05]
    DEV = [1.225, 1.275, 1.325, 1.375]
    runs = {}
    failed_episodes = []
    for folder in sorted((OUT / 'RUNS').iterdir()):
        reg = js(folder / 'PREREGISTRATION.json')
        if not (folder / 'RESULT.json').exists():
            failed_episodes.append(dict(folder=str(folder), label=reg['label'],
                                        files=sorted(x.name for x in folder.iterdir())))
            continue
        runs[reg['label']] = dict(reg=reg, result=js(folder / 'RESULT.json'), folder=str(folder))

    def get(g, kind, phase):
        return runs[f'gain_{g}_{kind}_{phase}']['result']

    ranking = []
    base_guard = get(1.0, 'guard', 1.3)
    for g in GAINS:
        guard = get(g, 'guard', 1.3)
        dev = [get(g, 'dev', p) for p in DEV]
        b = base_guard['parts']
        r = guard['parts']

        def ge(x, y):
            if y is None:
                return True
            if x is None:
                return True
            return x >= y - 1e-9
        guard_pass = (guard['J'] <= base_guard['J'] + 1e-9
                      and ge(r['foot_support_mean'], b['foot_support_mean'])
                      and ge(r['com_z_min'], b['com_z_min'])
                      and ge(r['first_nonfoot_s'], b['first_nonfoot_s']))
        ranking.append(dict(
            gain=g, guard_pass=bool(guard_pass), guard_J=guard['J'],
            guard_foot_support=guard['parts']['foot_support_mean'],
            guard_com_z_min=guard['parts']['com_z_min'],
            guard_first_nonfoot_s=guard['parts']['first_nonfoot_s'],
            guard_saturation=guard['saturation_fraction'],
            dev_passes=sum(x['validation']['passed'] for x in dev),
            worst_dev_J=max(x['J'] for x in dev),
            mean_dev_J=float(np.mean([x['J'] for x in dev])),
            score=(sum(x['validation']['passed'] for x in dev), -max(x['J'] for x in dev),
                   -float(np.mean([x['J'] for x in dev])))))
    safe = [r for r in ranking if r['guard_pass']]
    selected = max(safe, key=lambda r: r['score']) if safe else ranking[0]

    # Mechanism: is the surviving command a stiff saturated pattern rather than a gait?
    folder = Path([v for k, v in runs.items() if k == 'gain_1.0_guard_1.3'][0]['folder'])
    body = arrays(folder / 'BODY.npz')
    native = arrays(folder / 'NATIVE.npz')['rows']
    ctrl = body['ctrl'][1::2]
    limits = np.array(js(ARCH / 'TRANSLATOR_IDENTITIES.json')['limits'])
    warm = arrays(ARCH / 'DATA/train_offset_0/BODY.npz')
    m = warm['time'] >= 1.0
    assist = warm['ctrl'][m][:, :42]
    norm = ctrl / limits[None, :]
    assist_norm = assist / limits[None, :]
    mech = dict(
        release_saturation=float(np.mean(np.abs(ctrl) >= (limits[None, :] - 1e-9))),
        per_joint_std_over_mean=float(np.mean(np.std(norm, axis=0) / (np.abs(np.mean(norm, axis=0)) + 1e-9))),
        mean_normalized_torque=float(np.mean(norm)),
        assist_mean_normalized_torque=float(np.mean(assist_norm)),
        assist_saturation=float(np.mean(np.abs(assist) >= (limits[None, :] - 1e-9))),
        assist_per_joint_std_over_mean=float(np.mean(np.std(assist_norm, axis=0) / (np.abs(np.mean(assist_norm, axis=0)) + 1e-9))),
        release_mean_abs_torque=float(np.mean(np.abs(ctrl))),
        assist_mean_abs_torque=float(np.mean(np.abs(assist))),
        fraction_of_release_samples_at_clip=float(np.mean(np.all(np.abs(ctrl) >= (limits[None, :] - 1e-9), axis=1))),
    )

    verdict = dict(
        stage='S5 bounded decoder-gain validation', importance='核心必需',
        outcome='REFUTED. Reducing the decoder gain removes saturation but collapses support; the surviving '
                'baseline command works only as a near-saturated stiff pattern.',
        ranking=ranking, selected=selected,
        selected_is_baseline=bool(selected['gain'] == 1.0),
        final_phases_not_run=dict(
            reason='run.py crashed with KeyError 1.209 because the new final phases are absent from the archived '
                   'handover cache; the readout gain does not affect the warmup, so new phases would need a new 1.4 s '
                   'warmup. After the frozen rule resolved to the 1.0 baseline, running untouched final phases for the '
                   'baseline would only restate a known result, so they were not run and no unseen-state claim is made.',
            exit_code=1, physics_completed_before_crash=20, warmup_needed='one 1.4 s warmup with the four new capture times',
            leftover_incomplete_run_directories=failed_episodes),
        baseline_reproduction=dict(this_run_J=base_guard['J'], archived_J=1.5903704331878943,
                                   matches=bool(abs(base_guard['J'] - 1.5903704331878943) < 1e-9)),
        mechanism=mech,
        interpretation='With the gain reduced to 0.25 or below the saturation falls to 0.0 but the loss rises to about '
                       '4.08-4.30 and no gate passes: graded torque of comparable average magnitude to the assist does '
                       'not hold the body, because the command is not a gait pattern. At gain 1.0 the command is '
                       'largely pinned at the actuator limits, which behaves as a stiff stance and produces the best '
                       'loss so far only for roughly 0.1 s. This is an engineering observation; it does not establish '
                       'missing information, a required action, or walking intent.',
        boundaries=['decoder scalar only; neurons, connectome, sensory encoder, plant and limits unchanged',
                    'no body-state feedback, no online PD or CPG, zero MLP residual branch',
                    'absolute forces and SI remain NOT_ESTABLISHED',
                    'no behaviour gate passed at any gain'])
    dump(OUT / 'VERDICT.json', verdict)
    dump(OUT / 'GAIN_SELECTION.json', dict(ranking=ranking, selected=selected, final_untouched=True,
                                           note='Selection finished from saved dev and guard runs; see VERDICT.json.'))
    dump(OUT / 'INPUT_HASHES_AFTER.json', {p: sha(p) for p in js(OUT / 'INPUT_HASHES_BEFORE.json')})
    print(json.dumps(verdict['ranking'], ensure_ascii=False, indent=1))
    print('MECH', json.dumps(mech, indent=1))


if __name__ == '__main__':
    main()
