"""Score the recorded best vector on each ablation arm, with nothing trained.

This is the cheap decisive step of Route B. The question is not "can a controller be trained on a
rewired graph" but "does the SPECIFIC wiring of this sampled BANC graph carry the result". The
cleanest way to ask that is to take the exact vector the project recorded as its best, drop it
unchanged onto three networks that differ only in their recurrent matrices, and score it.

  A  real        the archived rebuilt BANC weights
  B  rewired     same neurons, same edges, same per node in and out degree, same weight multiset,
                 random pairing - the honest control
  C  uniform      same neurons, same edges, same weight multiset, no degree control

Nothing is optimised, so no arm gets an advantage from a search that suits its own dynamics. Every
arm uses the same timestep, which matters here because the previous package measured that J moves by
0.028 when the timestep moves - that number is the scale a real difference has to clear.

The run also records the network side of each arm from the release file, so the per leg actuator
drive can be compared and not just the score.

Usage: env AB_ARM_OUT (package to write into). Build the arms first with build_ablation_networks.py.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PCR_SYNTH'] = '1'
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'tools'))
O = Path(os.environ.get('AB_ARM_OUT', '/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z'))
BEST = Path('/home/sensen/Desktop/engine修复验收/s5-synthetic-readout-cont/20261107T120000Z')
REF_J = 3.402395132105888
TIMESTEP_BAND = 0.027848515225160853     # measured in s5-neural-dt-sensitivity, the 2x timestep shift
ARMS = ('A', 'B', 'C')


def js(p):
    return json.loads(Path(p).read_text())


def _san(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, dict):
        return {k: _san(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_san(x) for x in v]
    return v


def dump(p, v):
    Path(p).write_text(json.dumps(_san(v), ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


_PATCH = {}


def install_arm_matrices(we, wi):
    """Install the arm's matrices at the point where the network actually receives them.

    Two defects were found here, both worth recording because both silently produced a wrong graph:

    1. Pointing the connectivity set at an arm only changes the file the trainer COMPARES against.
       The build is deterministic from the BANC data, so the network still got the real matrices and
       the trainer's weight hash assertion fired. The first attempt failed on arms B and C for this.
    2. Replacing built['edges'] looked like the fix but was not. Where the tier is consumed,
       engine/embodied/continuous_handover.py line 31 does

           net.set_connectivity(built['edges']['We'] * .2, built['edges']['Wi'] * .2)

       because the archived matrices are the built weights times 0.2. Handing it an archived matrix
       therefore fed the network a second factor of 0.2 and made every arm, including A, fail - which
       is exactly what the second attempt did.

    So the substitution happens at ConductanceNetwork.set_connectivity, after all scaling, which is
    the only place where the matrix the network ends up with is unambiguous.
    """
    import engine.neural_cond as nc
    orig = nc.ConductanceNetwork.set_connectivity

    def substituted(self, exc_uS, inh_uS):
        del exc_uS, inh_uS      # the scaling upstream is deliberately overridden; see the note above
        return orig(self, we, wi)

    nc.ConductanceNetwork.set_connectivity = substituted
    _PATCH['nc'] = nc
    _PATCH['orig_sc'] = orig


def restore_builder():
    if 'orig_sc' in _PATCH:
        _PATCH['nc'].ConductanceNetwork.set_connectivity = _PATCH['orig_sc']
        _PATCH.clear()


def build_set_for_arm(pcr, arm):
    """A connectivity set identical to the project's, except for the recurrent matrices."""
    base = O / 'SET_BASE'
    arm_dir = O / ('SET_' + arm)
    for d in (base, arm_dir):
        if d.exists():
            shutil.rmtree(d)
    pcr.O = O
    pcr.SET = base
    pcr.make_set()
    arm_dir.mkdir(parents=True)
    for f in sorted(base.iterdir()):
        if f.name in ('ACTUAL_We.npz', 'ACTUAL_Wi.npz'):
            continue
        (arm_dir / f.name).symlink_to(f)
    for name in ('We', 'Wi'):
        shutil.copyfile(O / 'ARMS' / arm / ('ACTUAL_' + name + '.npz'), arm_dir / ('ACTUAL_' + name + '.npz'))
    return arm_dir


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import train_per_cell_readout as pcr
    import scipy.sparse as sparse
    out = {}
    vec = np.asarray(js(BEST / 'EVALS' / 'eval_556' / 'SCORE.json')['vec'], float)
    for arm in ARMS:
        t0 = time.time()
        arm_dir = build_set_for_arm(pcr, arm)
        pcr.SET = arm_dir
        vec_sha = dict(We=sha(arm_dir / 'ACTUAL_We.npz'), Wi=sha(arm_dir / 'ACTUAL_Wi.npz'))
        folder = O / 'EVALS' / ('arm_' + arm)
        if folder.exists():
            shutil.rmtree(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(arm=arm, matrix_sha256=vec_sha, vec_dim=int(vec.size))
        we = sparse.load_npz(arm_dir / 'ACTUAL_We.npz')
        wi = sparse.load_npz(arm_dir / 'ACTUAL_Wi.npz')
        install_arm_matrices(we, wi)
        try:
            r = pcr.eval_one((-1, vec, folder))
            rec.update(J=r.get('J'), parts=r.get('parts'), failed=bool(r.get('failed', False)))
        except BaseException:
            rec.update(failed=True, traceback=traceback.format_exc()[-1500:])
        finally:
            restore_builder()
        rec['wall_s'] = time.time() - t0
        for name in ('We', 'Wi'):
            m = sparse.load_npz(arm_dir / ('ACTUAL_' + name + '.npz'))
            rec[name + '_nnz'] = int(m.nnz)
        try:
            nz = np.load(folder / 'RUN' / 'NEURAL.npz')
            counts = nz['spike_counts']
            cmds = nz['commands']
            rec['cells_spiking'] = int(np.count_nonzero(counts))
            rec['cells_total'] = int(counts.size)
            rec['spikes_total'] = int(counts.sum())
            rec['command_mean'] = float(cmds.mean())
            rec['command_zero_rows'] = int((cmds.max(axis=0) <= 0).sum()) if cmds.ndim == 2 else None
            rec['drive_by_leg'] = {}
            layout = js(BEST / 'LAYOUT.json') if (BEST / 'LAYOUT.json').exists() else {}
            man = js(folder / 'RUN' / 'MAPPING.json')['rows']
            last = cmds[-1] if cmds.ndim == 2 else cmds
            for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
                idx = [i for i, r2 in enumerate(man) if r2.get('leg') == leg]
                rec['drive_by_leg'][leg] = dict(actuators=len(idx),
                                                driven_last=int(sum(1 for i in idx if last[i] > 0)),
                                                mean_command_last=float(np.mean([last[i] for i in idx])) if idx else None)
        except BaseException:
            rec['drive_error'] = traceback.format_exc()[-500:]
        out[arm] = rec
        dump(O / 'FROZEN_SCORE.json', dict(reference_J=REF_J, timestep_band=TIMESTEP_BAND,
                                          instruction='the recorded best vector, unchanged, scored on each arm',
                                          arms=out,
                                          note='nothing was trained; a difference must exceed the timestep band to count'))
        print('arm %s  J=%s  spiking=%s/%s  %.0fs' % (arm, rec.get('J'), rec.get('cells_spiking'),
                                                     rec.get('cells_total'), rec['wall_s']), flush=True)
    # verdict
    a = out.get('A', {}); b = out.get('B', {}); c = out.get('C', {})
    v = dict(timestep_band=TIMESTEP_BAND, reference_J=REF_J, arms={k: out[k].get('J') for k in out})
    if all(out.get(k, {}).get('J') is not None for k in ARMS):
        dAB = out['B']['J'] - out['A']['J']; dAC = out['C']['J'] - out['A']['J']
        v.update(d_B_minus_A=float(dAB), d_C_minus_A=float(dAC),
                 rewired_differs_beyond_timestep_band=bool(abs(dAB) > TIMESTEP_BAND),
                 uniform_differs_beyond_timestep_band=bool(abs(dAC) > TIMESTEP_BAND),
                 reading=('the specific wiring does not matter to this task: a degree and weight matched '
                          'rewiring scores within the timestep band'
                          if abs(dAB) <= TIMESTEP_BAND else
                          'the specific wiring matters: the rewiring moves the score beyond the timestep band'))
    dump(O / 'FROZEN_VERDICT.json', v)
    print(json.dumps(_san(v), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
