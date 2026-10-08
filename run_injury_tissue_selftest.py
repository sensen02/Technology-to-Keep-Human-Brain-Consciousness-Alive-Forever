"""Executable illustrative injury numerical/accounting tests, no viability thresholds."""
from dataclasses import replace
import json
import math
import numpy as np
from engine.injury_tissue import (InjuryPotassium, PotassiumConfig, InjuryLigand,
                                 nernst_k_mV, R_J_MOL_K, F_C_MOL)
from engine.local_tissue import FinitePool, PrescribedBath


def run_tests():
    results = {}
    c = PotassiumConfig(permeability_um_s=.1)
    closed = InjuryPotassium(c)
    for _ in range(300):
        closed.step(.1)
        assert min(closed.inside_mM, closed.outside_mM, closed.buffer_amount) >= 0
        assert closed.buffer_amount <= c.buffer_capacity_amount
    assert abs(closed.mass_balance_error()) < 1e-7
    results['closed_mass_error_mM_um3'] = closed.mass_balance_error()

    eq = InjuryPotassium(replace(c, initial_inside_mM=3., initial_outside_mM=3.,
                                buffer_capacity_amount=0., reservoir_conductance_um3_s=10.))
    eq.step(200.)
    assert np.allclose([eq.inside_mM, eq.outside_mM], 3., atol=1e-13)
    assert abs(eq.last_membrane_out_amount) < 1e-10
    results['zero_flux_equilibrium'] = True

    saturated = InjuryPotassium(replace(c, buffer_capacity_amount=10., buffer_off_s=0.,
                                        buffer_on_mM_inv_s=100.))
    for _ in range(20):
        saturated.step(1.)
    assert 9.999 < saturated.buffer_amount <= 10.
    assert abs(saturated.mass_balance_error()) < 1e-7
    results['buffer_saturation_amount'] = saturated.buffer_amount

    exchanged = []
    for bath in (0., 200.):
        m = InjuryPotassium(replace(c, reservoir_mM=bath, reservoir_conductance_um3_s=30.))
        for _ in range(100):
            m.step(.2)
        assert abs(m.mass_balance_error()) < 1e-7
        assert m.ledger['reservoir_out' if bath == 0 else 'reservoir_in'] > 0
        exchanged.append(dict(bath_mM=bath, ledger=m.ledger, error=m.mass_balance_error()))
    results['reservoir_ledgers'] = exchanged

    expected = 1000*R_J_MOL_K*298.15/F_C_MOL*math.log(.1)
    assert abs(nernst_k_mV(100, 10)-expected) < 1e-12
    assert abs(expected+59.159349686) < 1e-8
    assert nernst_k_mV(10, 10) == 0
    assert abs(nernst_k_mV(100, 10)-nernst_k_mV(.1, .01)) < 1e-12
    for values in ((0, 3), (100, 0), (-1, 1), (float('nan'), 1)):
        try:
            nernst_k_mV(*values)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid Nernst input accepted')
    results['nernst_decade_mV_at_298K'] = expected

    # No-buffer closed exchange has an exact two-pool exponential solution.
    c0 = replace(c, buffer_capacity_amount=0.)
    vi, ve = c0.intracellular_volume_um3, c0.extracellular_volume_um3
    g = c0.membrane_area_um2*c0.permeability_um_s
    mean = (vi*c0.initial_inside_mM+ve*c0.initial_outside_mM)/(vi+ve)
    exact = mean-(vi/(vi+ve))*(c0.initial_inside_mM-c0.initial_outside_mM)*math.exp(-g*(1/vi+1/ve)*10)
    errors = []
    for dt in (.2, .1, .05):
        m = InjuryPotassium(c0)
        for _ in range(round(10/dt)):
            m.step(dt)
        errors.append(abs(m.outside_mM-exact))
    assert errors[2] < .55*errors[1] < .55**2*errors[0]
    results['first_order_exchange_errors_mM'] = errors
    # Combined split kinetics convergence against a finer reference.
    end_states = []
    for dt in (.2, .1, .05, .0125):
        m = InjuryPotassium(c)
        for _ in range(round(5/dt)):
            m.step(dt)
        end_states.append(np.array([m.inside_mM, m.outside_mM, m.buffer_amount/200.]))
    errors = [float(np.linalg.norm(x-end_states[-1])) for x in end_states[:-1]]
    assert errors[2] < errors[1] < errors[0]
    results['split_kinetics_convergence_errors'] = errors

    before = closed.total_amount()
    concentrations = (closed.inside_mM, closed.outside_mM, closed.buffer_amount)
    closed.apply_injury(.4, 20.)
    assert closed.total_amount() == before and concentrations == (closed.inside_mM, closed.outside_mM, closed.buffer_amount)
    assert closed.events[-1]['mass_jump'] == 0
    ligand = InjuryLigand(shape=(2, 1, 1), initial_nM=np.array([2., 0.]).reshape(2, 1, 1),
        receptor_capacity_nM=2., source_amount_s=.1, clearance_s=.02,
        finite_pools=(FinitePool(5., 3., .2),), baths=(PrescribedBath(1., .1),),
        barrier_edges={(0, 1): 0.})
    for _ in range(5):
        ligand.step(.1)
    for edges in ({(0, 1): 2.}, {(0, 1): 0.}, {}):
        before = ligand.total_amount()
        ledger, response = ligand.ledger, ligand.model.response.copy()
        free, bound, pools = (ligand.model.free_nM.copy(), ligand.model.bound_nM.copy(), ligand.model.pool_nM.copy())
        ligand.switch_barrier(edges)
        assert ligand.total_amount() == before and ligand.ledger == ledger
        for old, new in ((response, ligand.model.response), (free, ligand.model.free_nM),
                         (bound, ligand.model.bound_nM), (pools, ligand.model.pool_nM)):
            assert np.array_equal(old, new)
        ligand.step(.1)
        assert abs(ligand.mass_balance_error()) < 1e-11
    before, ledger = ligand.total_amount(), ligand.ledger
    try:
        ligand.switch_barrier({(0, 4): 1.})
    except ValueError:
        pass
    else:
        raise AssertionError('invalid barrier accepted')
    assert before == ligand.total_amount() and ledger == ligand.ledger
    results['injury_events_no_instant_mass_creation'] = True
    results['ligand_multisegment_error_nM_um3'] = ligand.mass_balance_error()
    for dt in (0., 1e-6, 1., 1000.):
        m = InjuryPotassium(c)
        m.step(dt)
        assert min(m.inside_mM, m.outside_mM, m.buffer_amount) >= 0
        assert abs(m.mass_balance_error()) < 1e-6
    for bad in (replace(c, intracellular_volume_um3=0), replace(c, buffer_capacity_amount=-1),
                replace(c, initial_buffer_amount=c.buffer_capacity_amount+1)):
        try:
            InjuryPotassium(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid config accepted')
    results['positivity_extreme_dt_and_input_validation'] = True
    return results


if __name__ == '__main__':
    print(json.dumps({'status': 'PASS', 'tests': run_tests()}, indent=2))
