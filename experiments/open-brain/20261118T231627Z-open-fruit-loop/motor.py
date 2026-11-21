"""Calibrate named upstream Eon contexts against the BANC ground receiver."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from fastbrain import ARCHIVE, BASE, FastBrain, context_groups, eon, load_connectome, verify_upstream
from neural_runtime import BIDIR, FIT, read, sources, verify, write

PRIOR = Path(__file__).resolve().parent.parent/'20261118T223412Z-fruit-cues'
sys.path.insert(0, str(PRIOR))
from transport_probe import FIXED, GROUND, CalibratedReceiver, Ground
sys.path.insert(0, str(BASE/'20261118T031132Z-flat-neck-assay'))
from assay_timing import CanonicalEvents
from assay_transport import FixedCircuit
from assay_timing import FrontTouchNeck


SIDES = ('left', 'right')
DT = .0001
BODY_WEIGHT = BASE/'20261117T211320Z-body-command-calibration/FIT/epoch040.pt'
POLICY_RATES = ((0,0),(25,25),(55,55),(90,90),(120,120),(130,50),(50,130),
                (105,25),(25,105),(115,0),(0,115),(65,25),(25,65))
ODOUR_GAIN = 40.


def odor_indices(index):
    directory = ARCHIVE/'data/flywire'
    with gzip.open(directory/'fafb_classification.csv.gz', 'rt') as stream:
        side = {int(row['root_id']): row['side'] for row in csv.DictReader(stream)}
    sets = {key: [] for key in SIDES}
    with gzip.open(directory/'fafb_consolidated_cell_types.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            raw = int(row['root_id'])
            if row['primary_type'] in ('ORN_DM1','ORN_VA2') and raw in index:
                sets[side[raw]].append(index[raw])
    assert sum(map(len,sets.values())) == 135
    return [np.array(sets[s], dtype=int) for s in SIDES]


class NeuralInputs:
    def __init__(self,index):
        groups = context_groups(index)
        self.pfl = [np.array(groups[f'pfl_{side}']) for side in SIDES]
        assert not set(self.pfl[0]).intersection(self.pfl[1])
        self.odor = odor_indices(index)
        self.sugar = np.array([index[root] for root in eon.get_experiment('sugar')['neu_exc']])
        self.an = read(FIT/'BINDING.json')['records']
        self.an_ix = [row['flywire_index'] for row in self.an]
        self.stn = np.array([index[int(row['flywire_root'])] for row in read(
            PRIOR.parent/'20261118T203315Z-active-meal-stop/DOSE127/BINDING.json')['set_4']])
        self.mn9 = [index[720575940660219265], index[720575940618238523]]
        self.assembled = np.concatenate((*self.pfl,*self.odor,self.sugar,self.stn))
        assert len(set(self.assembled)) == len(self.assembled)

    def masks(self, ticks, seed):
        rng = np.random.default_rng(seed)
        return rng.random((ticks, len(self.assembled)),dtype=np.float32)

    def rates(self, pfl, smell, sugar, stn):
        return np.repeat([*pfl,*smell,sugar,stn],
                         [*(len(g) for g in self.pfl),*(len(g) for g in self.odor),
                          len(self.sugar), len(self.stn)])


class Neck:
    def __init__(self,index, inputs, tactile=False):
        catalog_path = BIDIR/'CATALOG.json'
        catalog = read(catalog_path)['ports']
        admitted = read(BIDIR/'EON/RESULT.json')['admitted']
        ground = {root: int(match) for root, match in admitted.items()
                  if catalog[root]['group'] not in ('takeoff','wing_amplitude','landing','flight_steer')}
        assert len(ground) == 17
        self.root_names = list(ground)
        self.dn_ix = np.array([index[root] for root in ground.values()])
        an = [row['banc_root'] for row in inputs.an]
        groups = {catalog[root]['group'] for root in ground} | {'front_touch','front_load'}
        ports = {root: datum for root, datum in catalog.items() if datum['group'] in groups}
        if tactile:
            self.model = FrontTouchNeck({'ports':ports},an,matched_lanes=(0,),feedback_lanes=(0,),batch=1)
        else:
            self.model = FixedCircuit({'ports':ports})
        self.an_port = [self.model.port_index[root] for root in an]
        self.receptors = [self.model.validate_inputs([r for r,_ in catalog[root]['incoming_receptors']],
                                                     'body') for root in an]
        self.an_boundary = CanonicalEvents((2,))
        self.an_inputs = np.zeros(2, dtype=bool)
        self.columns = [self.model.port_index[root] for root in ground]
        self.reader = CalibratedReceiver(self.model)
        anchorfile = BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json'
        anchors = read(anchorfile)['probes']
        for key, (_, _, sign) in self.reader.probes.items():
            roots = anchors['/'.join(key)]
            self.reader.probes[key] = roots,[self.model.index[r] for r in roots],sign
        self.calibration_file = GROUND/'REFINE/CALIBRATION.json'
        self.decode = SimpleNamespace(receiver=self.reader, calibration=read(self.calibration_file))
        self.events = np.zeros((1, len(ports)),dtype=bool)
        self.cut = np.zeros(len(ports),dtype=bool)
        self.drive = np.zeros((1,len(self.model.roots)),dtype=np.float32)
        self.command = np.zeros(2)
        self.last_dn = np.zeros(len(ground),dtype=bool)
        self.last_an = np.zeros(2,dtype=bool)

    def update(self,tick,dn,front,blocked=False):
        self.events[0,self.columns] = dn
        self.cut[self.columns] = blocked
        self.drive.fill(0.)
        if tick % 23 == 0:
            for idx in np.flatnonzero(front > .04):
                self.drive[0,self.receptors[idx]] = 68.75
        _, arrived, _, voltage = self.model.step(self.drive,self.cut,eon_output=self.events)
        self.last_dn = arrived[0,self.columns]
        self.last_an = arrived[0,self.an_port]
        self.an_inputs = self.an_boundary.step(self.last_an)
        if tick % 5 == 0:
            self.reader.update(voltage, DT*5)
            self.command = Ground.decode(self.decode)
        return self.command


def fit_response(obs):
    # Fit a small forward model and invert it online over the calibrated rate lattice.
    a = np.array([[row['odor_hz'], *row['rates_hz']] for row in obs])
    features = response_features(a[:,0],a[:,1],a[:,2])
    outputs = np.array([r['speed_turn'] for r in obs])
    coeff = np.linalg.solve(features.T @ features + np.eye(features.shape[1])*.02,
                            features.T @ outputs)
    return coeff


def response_features(odor,left,right):
    advance = (left+right)/200.
    turn = (left-right)/120.
    c = np.broadcast_to(odor/30., advance.shape)
    return np.stack((np.ones_like(advance),advance,turn,c,advance**2,turn**2,advance*turn,
                     advance*c,turn*c),axis=-1)


@torch.inference_mode()
def calibrate(out, levels):
    torch.set_num_threads(1)
    verify_upstream()
    index,_ = eon.get_hash_tables(eon.path_comp)
    outgoing = load_connectome(index).T.tocsr()
    binding = NeuralInputs(index)
    out.mkdir()
    rows = []
    for baseline in levels:
        for j,rates in enumerate(POLICY_RATES):
            brain = FastBrain(outgoing,binding.an_ix)
            neck = Neck(index,binding)
            draws = binding.masks(2300,1701+j)
            active_p = binding.rates(rates,(baseline,baseline),0.,0.) * DT
            telemetry = np.zeros((2300,2))
            for tick in range(len(draws)):
                neural = binding.assembled[draws[tick] < active_p]
                brain.step(neural)
                telemetry[tick] = neck.update(tick,brain.spikes[neck.dn_ix],np.zeros(2))
            average = telemetry[850:].mean(axis=0)
            value = [float(np.mean(average)),float((average[1]-average[0])/2)]
            rows.append(dict(rates_hz=rates,odor_hz=baseline,speed_turn=value))
            print(baseline,rates,np.round(value,3).tolist(),flush=True)
    model = fit_response(rows)
    data = np.array([[r['odor_hz'],*r['rates_hz']] for r in rows])
    predicted = response_features(*data.T) @ model
    error = np.abs(predicted-np.array([r['speed_turn'] for r in rows])).mean(axis=0)
    write(out/'FIT.json',dict(fitting='ridge system identification in recipient command space',
        sampling_rows=rows, coefficients=model.tolist(), command_fit_mae=error.tolist(),
        neural_steps=2300*len(rows), real_cervical_transport=True))


class MotorAdapter:
    def __init__(self,trained):
        self.model = np.array(trained['coefficients'])
        pair = np.arange(0.,151.,5.)
        self.left,self.right = np.meshgrid(pair,pair)

    def rates(self,speed,turn,mean_odor):
        if speed == turn == 0.:
            return (0.,0.)
        result = response_features(mean_odor,self.left,self.right)@self.model
        cost = 2.5*(result[...,0]-speed)**2 + 2.4*(result[...,1]-turn)**2
        cost += .0000003*(self.left**2 + self.right**2)
        selected = np.unravel_index(np.argmin(cost),cost.shape)
        return self.left[selected],self.right[selected]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--levels',nargs='+',type=float,default=[0.,17.,34.])
    a=p.parse_args()
    calibrate(a.output,a.levels)
