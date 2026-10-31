"""Independent S129 scalar arithmetic verifier. No simulator import or calls."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from verify_muscle_reconstruction import evaluate

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收')
ARMS = [BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms',
        BASE/'s127-joint-candidate-closed-loop/20261024T082500Z/init80ms']
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def main(out):
    out = out.resolve()
    target = out/'PARENT_INDEPENDENT_CHECK.json'
    assert not target.exists(), 'Never overwrite prior independent evidence'
    inputs = json.loads((out/'INPUT_HASHES.json').read_text())
    for p,h in inputs.items():
        assert sha(p) == h, 'Input commitment mismatch: '+p
    output_hashes = json.loads((out/'OUTPUT_HASHES.json').read_text())
    for p,h in output_hashes.items():
        assert sha(p) == h, 'Output commitment mismatch: '+p
    committed = {str(p):sha(p) for p in [XML, ROOT/'verify_muscle_reconstruction.py', Path(__file__).resolve(),
                  *[a/'TRAJECTORY.npz' for a in ARMS], out/'OFFLINE_2X2.npz',out/'ALL48_SHAPLEY.csv']}
    arms = [load(a/'TRAJECTORY.npz') for a in ARMS]
    nodes = list(ET.parse(XML).getroot().find('actuator'))
    assert len(nodes) == 90
    errors = {}
    def equal(label, actual, expected, atol=1e-10, rtol=1e-12):
        actual,expected = np.asarray(actual),np.asarray(expected)
        assert actual.shape == expected.shape, (label,actual.shape,expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
        residual = np.abs(actual-expected)
        errors[label] = float(np.max(residual, initial=0))
        assert np.all(residual <= atol+rtol*np.abs(expected)), (label,errors[label])
    for field,n in [('lengthrange',2),('gainprm',10),('biasprm',10)]:
        v = np.zeros((90,n),dtype=np.float64)
        for j,node in enumerate(nodes):
            values = list(map(float,node.get(field).split()))
            v[j,:len(values)] = values
        h = hashlib.sha256(v.tobytes()).hexdigest()
        for arm in ARMS:
            for phase in ('before','after'):
                assert json.loads((arm/'MODEL_PROVENANCE.json').read_text())[phase]['model.actuator_'+field]['sha256'] == h
    force = np.empty((2,2,121,90)); gains = np.empty((2,121,90)); biases = gains.copy()
    for s in range(2):
        assert arms[s]['act'].shape == (121,90)
        assert not arms[s]['qfrc_applied'].any() and not arms[s]['xfrc_applied'].any()
        for i in range(121):
            for u,node in enumerate(nodes):
                for a in range(2):
                    f,g,b = evaluate(arms[s]['actuator_length'][i,u],arms[s]['actuator_velocity'][i,u],arms[a]['act'][i,u],node)
                    force[s,a,i,u] = f
                    gains[s,i,u],biases[s,i,u] = g,b
        equal('historical_force_'+str(s), force[s,s],arms[s]['actuator_force'])
    signed = np.stack([force[s,:,:,:,None]*arms[s]['actuator_moment'][None,:,:,:] for s in range(2)])
    torque = np.sum(signed,axis=3)
    activation = .5*((torque[0,1]-torque[0,0])+(torque[1,1]-torque[1,0]))
    state = .5*((torque[1,0]-torque[0,0])+(torque[1,1]-torque[0,1]))
    total = torque[1,1]-torque[0,0]
    equal('shapley_efficiency',activation+state,total)
    for s in range(2):
        equal('historical_generalized_force_'+str(s),torque[s,s],arms[s]['qfrc_actuator'])
    evidence = load(out/'OFFLINE_2X2.npz')
    for k,v in dict(force=force,torque=torque,signed_per_actuator_torque=signed,gain=gains,bias=biases,
                    activation_direct=activation,state_FLV_moment=state,total=total).items():
        equal('archive_'+k,evidence[k],v)
    for k,s,a in [('torque_AA',0,0),('torque_AB',0,1),('torque_BA',1,0),('torque_BB',1,1)]:
        if k in evidence:
            equal('archive_'+k,evidence[k],signed[s,a])
    csvrows = list(csv.DictReader((out/'ALL48_SHAPLEY.csv').open()))
    assert len(csvrows) == 121*48
    seen = set()
    for row in csvrows:
        i,d = int(row['sample']),int(row['dof'])
        assert 0 <= i < 121 and 0 <= d < 48 and (i,d) not in seen
        seen.add((i,d))
        for k,v in [('tau_AA',torque[0,0,i,d]),('tau_AB',torque[0,1,i,d]),('tau_BA',torque[1,0,i,d]),('tau_BB',torque[1,1,i,d]),
                    ('activation_direct',activation[i,d]),('state_FLV_moment',state[i,d]),('total',total[i,d])]:
            equal('csv_'+str(i)+'_'+str(d)+'_'+k,float(row[k]),v)
    static = load(out/'STATIC_COUNTERFACTUAL.npz')
    gates = json.loads((out/'FORWARD_BASELINE_GATES.json').read_text())
    constraint_rows = [load(a/'CONSTRAINT_ROWS.npz') for a in ARMS]
    static_checks = []
    for gate in gates:
        s,i = int(gate['state_arm']),int(gate['sample'])
        prefix = f'arm{s}_sample{i:03d}_'
        has_swap = prefix+'swap_qacc' in static
        assert has_swap == bool(gate['passed']) == bool(gate['swap_executed'])
        if not has_swap:
            continue
        M = static[prefix+'reconstructed_mass']
        equal(prefix+'mass_symmetry', M,M.T,1e-12,1e-12)
        assert np.linalg.eigvalsh((M+M.T)*.5).min() > 0, prefix+'mass positive definite'
        dtau = static[prefix+'swap_qfrc_actuator']-static[prefix+'baseline_qfrc_actuator']
        dconstraint = static[prefix+'swap_qfrc_constraint']-static[prefix+'baseline_qfrc_constraint']
        dacc = static[prefix+'swap_qacc']-static[prefix+'baseline_qacc']
        equal(prefix+'delta_dynamics',M@dacc,dtau+dconstraint,1e-9,1e-10)
        equal(prefix+'fixed_constraint_driving',M@static[prefix+'fixed_constraint_driving_qacc'],dtau,1e-10,1e-10)
        equal(prefix+'swap_torque_offline',static[prefix+'swap_qfrc_actuator'],torque[s,1-s,i])
        equal(prefix+'baseline_torque_offline',static[prefix+'baseline_qfrc_actuator'],torque[s,s,i])
        hist = constraint_rows[s]
        J = hist[f'sample{i:03d}_efc_J_dense']
        equal(prefix+'baseline_JTforce',J.T@static[prefix+'baseline_efc_force'],static[prefix+'baseline_qfrc_constraint'])
        equal(prefix+'swap_JTforce',J.T@static[prefix+'swap_efc_force'],static[prefix+'swap_qfrc_constraint'])
        for label in ('baseline','swap'):
            assert np.array_equal(static[prefix+label+'_efc_type'],hist[f'sample{i:03d}_efc_type'])
            assert np.array_equal(static[prefix+label+'_efc_id'],hist[f'sample{i:03d}_efc_id'])
            if prefix+label+'_qacc_warmstart_before' in static:
                equal(prefix+label+'_cold_warmstart',static[prefix+label+'_qacc_warmstart_before'],np.zeros(48),0,0)
        static_checks.append(dict(state_arm=s,sample=i,mass_positive_definite=True,delta_dynamics=True,JTforce=True))
    response_path = out/'STATIC_RESPONSE_ALL48.csv'
    if static_checks:
        response = list(csv.DictReader(response_path.open()))
        assert len(response) == 48*len(static_checks)
        seen_response = set()
        for row in response:
            s,i,d = int(row['state_arm']),int(row['sample']),int(row['dof'])
            assert (s,i,d) not in seen_response
            seen_response.add((s,i,d))
            p = f'arm{s}_sample{i:03d}_'
            for name,v in [('delta_tau',static[p+'swap_qfrc_actuator'][d]-static[p+'baseline_qfrc_actuator'][d]),
                           ('fixed_constraint_driving_qacc',static[p+'fixed_constraint_driving_qacc'][d]),
                           ('actual_solver_delta_qacc',static[p+'swap_qacc'][d]-static[p+'baseline_qacc'][d]),
                           ('delta_qfrc_constraint',static[p+'swap_qfrc_constraint'][d]-static[p+'baseline_qfrc_constraint'][d])]:
                equal('response_csv_'+str(s)+'_'+str(i)+'_'+str(d)+'_'+name,float(row[name]),v)
    assert committed == {p:sha(p) for p in committed}
    for p,h in inputs.items():
        assert sha(p) == h
    result = dict(passed=True,scope='Independent all121/all90 exact force and full48 torque 2x2 arithmetic; NOT independent contact solver verification',
                  scalar_force_evaluations=2*2*121*90, csv_rows=len(csvrows),static_algebra_checks=static_checks,
                  maximum_absolute_errors={k:v for k,v in errors.items() if not k.startswith('csv_')},
                  csv_max_abs_error=max(v for k,v in errors.items() if k.startswith('csv_')),
                  input_sha256=committed, frozen_inputs_unchanged=True,model_loads=0,forward=0,steps=0,neural_ticks=0,
                  contact_causality_certified=False,physical_preheat_certified=False,
                  caveat='Historical warmstart absent; final endpoint COM/force identities do not prove solver restart reproducibility.')
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
