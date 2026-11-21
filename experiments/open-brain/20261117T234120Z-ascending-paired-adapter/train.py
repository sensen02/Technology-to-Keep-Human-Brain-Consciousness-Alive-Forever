"""Supervised fitter; fails closed before creating weights without native targets.

The guarded optimizer branch is implemented but has not been scientifically
qualified until a reviewed teacher and receiving boundary have been supplied.
"""
import argparse
import time
import numpy as np
import torch

from common import *
from adapter import AscendingRateAdapter


def qualify(audit, teacher_path):
    verify(audit['sources'])
    if not audit['eligible_common_channels']:
        return ['NO_REVIEWED_COMMON_SENSORY_CHANNEL']
    if teacher_path is None:
        return ['NO_NATIVE_TEACHER_MANIFEST']
    teacher = read(teacher_path)
    reasons = []
    if teacher.get('kind') != 'archived_original_sensory_encoder':
        reasons.append('TEACHER_IS_NOT_ORIGINAL_SENSORY_ENCODER')
    if teacher.get('receiving_boundary_review') != 'PASS':
        reasons.append('RECEIVING_BOUNDARY_NOT_REVIEWED')
    if teacher.get('channels') != audit['eligible_common_channels']:
        reasons.append('CHANNELS_NOT_ADMITTED')
    if teacher.get('causal_alignment_review') != 'PASS':
        reasons.append('DELAY_OR_CAUSAL_ALIGNMENT_NOT_REVIEWED')
    verify(teacher.get('source_sha256', {}))
    if not teacher.get('source_sha256') or not teacher.get('episodes'):
        reasons.append('MISSING_PINNED_TEACHER_DATA')
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--teacher', type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    audit = read(HERE/'SOURCE_AUDIT.json')
    reasons = qualify(audit, args.teacher)
    if reasons:
        write(args.output/'RESULT.json', dict(status='TRAINING_NOT_RUN', reasons=reasons,
              optimizer_updates=0, weights_written=False, brain_steps=0, physical_steps=0,
              audit_sha256=sha(HERE/'SOURCE_AUDIT.json')))
        print('training gate:', reasons, flush=True)
        return
    teacher = read(args.teacher)
    pins = pin(imported_sources() | {HERE/'SOURCE_AUDIT.json', args.teacher})
    pins.update(teacher['source_sha256'])
    train_x, train_y, held_x, held_y = [], [], [], []
    splits = {}
    for episode in teacher['episodes']:
        data_path = Path(episode['path']).resolve()
        assert str(data_path) in teacher['source_sha256']
        if episode['physical_episode_id'] in splits and splits[episode['physical_episode_id']] != episode['split']:
            raise ValueError('same physical episode leaked across splits')
        splits[episode['physical_episode_id']] = episode['split']
        with np.load(data_path, allow_pickle=False) as a:
            x, y = a['features_hz'].copy(), a['native_teacher_rate_hz'].copy()
            assert x.shape == (*y.shape, 3) and np.isfinite(x).all() and np.isfinite(y).all()
            assert (y >= 0).all() and (y <= 300).all()
            if episode['split'] == 'train':
                train_x.append(x); train_y.append(y)
            elif episode['split'] == 'heldout':
                held_x.append(x); held_y.append(y)
            else:
                raise ValueError('unrecognized split')
    if not train_x or not held_x:
        raise ValueError('need complete separate training and heldout episodes')
    torch.set_num_threads(1)
    x, y, hx, hy = [torch.from_numpy(np.concatenate(a)).double() for a in (train_x, train_y, held_x, held_y)]
    model = AscendingRateAdapter(len(teacher['channels']))
    mask = torch.zeros(len(teacher['channels']), dtype=torch.bool)
    def loss():
        return ((model(x, mask)-y)/300.).square().mean()
    # Check the actual objective, not a stand-in network loss.
    loss().backward()
    gradient = model.log_gain.grad[0].item()
    with torch.no_grad():
        old = model.log_gain[0].item()
        model.log_gain[0] = old+1e-5; plus=loss().item()
        model.log_gain[0] = old-1e-5; minus=loss().item()
        model.log_gain[0] = old
    fd = (plus-minus)/2e-5
    if abs(gradient-fd) > 1e-7:
        raise ValueError('actual supervised gradient failed')
    model.zero_grad()
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    began = time.monotonic()
    log = []
    for update in range(101):
        if time.monotonic()-began > 120:
            raise TimeoutError('fixed 120 second fit budget exceeded; no candidate published')
        if update:
            opt.zero_grad(); value=loss(); value.backward(); opt.step()
        with torch.no_grad():
            metrics=dict(update=update, train_mse=float(loss()), heldout_mse=float(((model(hx,mask)-hy)/300.).square().mean()))
        path=args.output/f'update{update:03d}.pt'
        torch.save(dict(state=model.state_dict(), optimizer=opt.state_dict(), update=update,
                        torch_rng=torch.get_rng_state(), source_sha256=pins, metrics=metrics),path)
        metrics['checkpoint_sha256']=sha(path); log.append(metrics)
    verify(pins)
    write(args.output/'RESULT.json',dict(status='FITTED_NOT_BRAIN_VALIDATED', optimizer_updates=100,
          gradient_analytic=gradient, gradient_fd=fd, log=log, source_sha256=pins,
          independent_replay=False, brain_steps=0, physical_steps=0,
          candidate=str(args.output/'update100.pt')))


if __name__ == '__main__':
    main()
