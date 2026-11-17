"""Fit the local phase-to-joint map and a bounded local sensory correction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import expansion as study
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof

HARMONICS = np.arange(1,33)
RESIDUAL_LIMIT = .015


class Oscillator:
    """Frozen intrinsic tripod dynamics, modulated only by actual VNC drive."""
    def __init__(self, body):
        self.cpg = body.motor.cpg_network
        self.frequencies = self.cpg.intrinsic_freqs.copy()

    def step(self, command):
        self.cpg.intrinsic_amps = np.repeat(np.abs(command), 3)
        signs = np.where(command>=0,1.,-1.)
        self.cpg.intrinsic_freqs = self.frequencies*np.repeat(signs,3)
        self.cpg.step()

    def features(self):
        return np.concatenate((self.cpg.curr_phases%(2*np.pi),self.cpg.curr_magnitudes)).astype(np.float32)


def basis(phases):
    theta = np.asarray(phases)[..., None]*HARMONICS
    return np.concatenate((np.ones((*theta.shape[:-1],1)),np.cos(theta),np.sin(theta)),axis=-1)


def fit_gait(body):
    phases = np.arange(512)*2*np.pi/512
    design = basis(phases)
    coefficients = []
    template_error = []
    positions = {}
    for index, leg in enumerate(body.motor.legs):
        template = body.motor.preprogrammed_steps.get_joint_angles(leg,phases).T
        neutral = body.motor.preprogrammed_steps.get_joint_angles(leg,0.,0.)
        offsets = template-neutral
        solution, *_ = np.linalg.lstsq(design, offsets,rcond=None)
        coefficients.append(solution)
        template_error.append(float(np.max(np.abs(design@solution-offsets))))
        for joint_index,spec in enumerate(body.motor.preprogrammed_steps.dofs_per_leg):
            positions[dof_spec_to_jointdof(leg,spec)] = 7*index+joint_index
    return np.asarray(coefficients), np.array([positions[dof] for dof in body.dofs]), template_error


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        # Remove the previous actuator setpoint columns physically from this model's input.
        self.reflex = nn.Sequential(nn.Linear(131,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,42))
        self.register_buffer('center',torch.zeros(131))
        self.register_buffer('scale',torch.ones(131))
        self.register_buffer('fourier',torch.zeros(6,65,7,dtype=torch.float64))
        self.register_buffer('order',torch.zeros(42,dtype=torch.int64))

    def features(self, local, phase):
        phases = phase[..., :6]
        return torch.cat((local[..., :86],local[..., 128:],
                          torch.sin(phases),torch.cos(phases)),dim=-1)

    def residual(self, local, phase):
        x = ((self.features(local,phase)-self.center)/self.scale).clamp(-10,10)
        gain = (local[...,:2].abs().amax(-1,keepdim=True)/.30).clamp(0,1)
        return RESIDUAL_LIMIT*torch.tanh(self.reflex(x))*gain

    def rhythmic_batch(self, phase):
        values = np.einsum('nle,lej->nlj',basis(phase[:, :6]),self.fourier.numpy())
        return (values*phase[:, 6:, None]).reshape(-1,42)[:, self.order.numpy()]

    @torch.inference_mode()
    def act(self, local, phase, with_residual):
        gait = self.rhythmic_batch(phase[None,:])[0]
        residual = self.residual(torch.from_numpy(local),torch.from_numpy(phase)).numpy() if with_residual else 0.
        return gait+residual


def phase_rows(body, command, seed):
    body.new_teacher(seed)
    oscillator = Oscillator(body)
    rows = []
    for tick, drive in enumerate(command):
        oscillator.step(drive)
        if tick%study.HOLD == 0:
            rows.append(oscillator.features())
    return np.asarray(rows)


def load_data(body, model, cases, source=None):
    local_rows, phases, residuals, sources = [], [], [], []
    for case in cases:
        episode = (source or HERE.parent/'TEACHER')/case.name
        metadata = json.loads((episode/'RESULT.json').read_text())
        if source is None and not metadata['pass_']:
            continue
        trace = episode/'TRACE.npz'
        digest = study.sha(trace)
        assert metadata['trace_sha256'] == digest
        with np.load(trace,allow_pickle=False) as arrays:
            obs,y,command = arrays['x'],arrays['y'],arrays['trace'][:,6:8]
        phase = phase_rows(body,command,case.seed)
        residual = (y-model.rhythmic_batch(phase)).astype(np.float32)
        local_rows.append(obs)
        phases.append(phase)
        residuals.append(residual)
        sources.append(str(trace)+':'+digest)
    packed = [torch.from_numpy(np.concatenate(arrays)) for arrays in (local_rows,phases,residuals)]
    return (*packed, sources)


def fit(correction=None, initial=None):
    torch.set_num_threads(1)
    torch.manual_seed(4708)
    rng = torch.Generator().manual_seed(6795)
    model = Policy()
    body = study.Body()
    fourier,order,template_error = fit_gait(body)
    assert max(template_error)<.002
    model.fourier.copy_(torch.from_numpy(fourier))
    model.order.copy_(torch.from_numpy(order))
    for parameter in model.reflex.parameters():
        if parameter.shape[0]==42:
            parameter.data.zero_()
    x, phase, residual, sources = load_data(body, model, study.TRAIN)
    dev_x, dev_phase, dev_r, dev_sources = load_data(body, model, study.DEVELOP)
    if correction is not None:
        assert initial is not None
        start = torch.load(initial,weights_only=True,map_location='cpu')
        model.load_state_dict(start['state'])
        extra = load_data(body,model,study.CORRECT,correction)
        x, phase, residual = [torch.cat((a,b)) for a,b in zip((x,phase,residual),extra[:3],strict=True)]
        sources += extra[3]
    sens = model.features(x, phase)
    model.center.copy_(sens.mean(0))
    model.scale.copy_(sens.std(0).clamp_min(.025))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    out = HERE/('FIT_CORRECT' if correction else 'FIT')
    out.mkdir()
    study.write(out/'INPUTS.json',dict(train=sources,develop=dev_sources,n_train=len(x), n_dev=len(dev_x),
                  template_max_error=template_error, no_last_action=True, harmonics=len(HARMONICS),
                  residual_limit=RESIDUAL_LIMIT))
    def save(epoch):
        path=out/f'epoch{epoch:03d}.pt'
        tmp=path.with_suffix('.temp')
        torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),epoch=epoch),tmp)
        tmp.replace(path)
        return study.sha(path)
    save(0)
    with (out/'JOURNAL.jsonl').open('x',buffering=1) as log:
        for epoch in range(1,16):
            for ids in torch.randperm(len(x),generator=rng).split(1024):
                error = model.residual(x[ids],phase[ids])-residual[ids]
                loss = torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=.01)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                error = model.residual(dev_x,dev_phase)-dev_r
                record = dict(epoch=epoch,sha256=save(epoch), loss=float(loss.detach()),
                              dev_rms=float(error.square().mean().sqrt()))
            log.write(json.dumps(record)+'\n')
            print(record,flush=True)


def read_policy(checkpoint):
    model = Policy()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['state'])
    model.eval()
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correct',type=Path)
    parser.add_argument('--initial',type=Path)
    options=parser.parse_args()
    fit(options.correct,options.initial)
