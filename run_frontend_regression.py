"""Tiny manually implemented numerical regressions; no BANC loading/hardware claims."""
import json
from types import SimpleNamespace as NS
import numpy as np
from engine.electrode import Contact, transfer_mV_per_nA
from engine.electrode_frontend import ContactGeometry, InterfaceImpedance, apply_frontend, contact_points_um, ContactQuadrature


def fixture(captures=((0,), (1,), ())):
    return NS(channel_centres_um=np.array([[0.,0.,0.],[20.,0.,0.],[40.,0.,0.]]),
              extent_um=np.array([40.,0.,0.]), channel_neurons=captures,
              soma_root_ids=np.array([101,102]), soma_positions_vox=np.array([[0.,0.,10.],[20.,0.,12.]]),
              resolution=NS(voxels_to_um=lambda v: np.asarray(v, dtype=float)), spec=NS(pitch_um=20.))


def main():
    c=ContactGeometry(7.)
    ii=InterfaceImpedance(c, amplifier_input_ohm=1e7)
    def run(am=None, **kw):
        return apply_frontend(am or fixture(), contact=c, interface=ii,
                              frequencies_Hz=[.001,100000.], convergence_check_channels=1, **kw).as_dict()
    p=run(); a=p['area_averaging']; src=[Contact(tuple(v),1.) for v in p['global_sources']['positions_um']]
    src.append(Contact(tuple(p['return_contact']['position_um']),1.))
    currents=np.array(p['global_sources']['currents_nA'])
    np.testing.assert_allclose(a['per_channel_point_mV'], transfer_mV_per_nA(fixture().channel_centres_um,src,.3)@currents)
    assert a['per_channel_point_other_mV'][2] > 0 and a['per_channel_area_mV'][2] != 0
    for prefix in ('point','area'):
        np.testing.assert_allclose(a[f'per_channel_{prefix}_neural_mV'],np.array(a[f'per_channel_{prefix}_own_mV'])+a[f'per_channel_{prefix}_other_mV'])
        np.testing.assert_allclose(a[f'per_channel_{prefix}_mV'],np.array(a[f'per_channel_{prefix}_neural_mV'])+a['return_contact_term'][f'per_channel_{prefix}_mV'])
    # Explicit reference subtraction identical for point, area and patch center.
    ref=[-200.,0.,-500.]; r=run(measurement_reference_um=ref)
    vref=float(transfer_mV_per_nA([ref],src,.3)[0]@currents)
    for key in ('per_channel_point_mV','per_channel_area_mV'):
        np.testing.assert_allclose(r['area_averaging'][key],np.array(a[key])-vref,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(r['patch_centre_sampling']['per_channel_patch_centre_mV'],np.array(p['patch_centre_sampling']['per_channel_patch_centre_mV'])-vref)
    assert run(measurement_reference_um=p['return_contact']['position_um'])['measurement_reference']['kind']=='infinity'
    # Full direct quadrature of the same globally balanced field.
    xyz,w=contact_points_um(c,source_um=fixture().channel_centres_um[2],quadrature=ContactQuadrature(4,16))
    np.testing.assert_allclose(a['per_channel_area_mV'][2],w@(transfer_mV_per_nA(xyz,src,.3)@currents)/w.sum(),rtol=1e-12)
    h=ii.transfer([1000.])[0]
    ph=p['recorded_probe_phasor_mV_area']; z=np.array(ph['real'])+1j*np.array(ph['imag'])
    np.testing.assert_allclose(z,np.array(a['per_channel_area_mV'])*h)
    assert np.any(abs(z.imag)>0)
    np.testing.assert_allclose(p['recorded_dc_mV_point'],np.array(a['per_channel_point_mV'])*ii.dc_transfer)
    # Overlap and duplicate rows/roots count only once and leave physical field unchanged.
    d=run(fixture(((0,0,1),(0,1),(0,1))))
    assert d['global_sources']['n_unique_roots']==2
    np.testing.assert_allclose(d['area_averaging']['per_channel_area_mV'],a['per_channel_area_mV'])
    dup=fixture(((0,2),(1,),()))
    dup.soma_root_ids=np.array([101,102,101]); dup.soma_positions_vox=np.vstack([dup.soma_positions_vox,dup.soma_positions_vox[0]])
    np.testing.assert_allclose(run(dup)['area_averaging']['per_channel_point_mV'],a['per_channel_point_mV'])
    for zero in (run(fixture(((),(),()))),run(neuron_current_nA=0)):
        assert not any(zero['area_averaging']['per_channel_area_mV'])
        assert zero['headline_amplitude_ratios']['asked_question_V_area_over_V_point_at_contact_centre_neural_only'] is None
        json.dumps(zero,allow_nan=False)
    json.dumps(p,allow_nan=False)
    ph=p['recorded_probe_phasor_mV_with_crosstalk']
    np.testing.assert_allclose(np.array(ph['real'])+1j*np.array(ph['imag']),np.array(p['crosstalk']['added']['matrix'])@z)
    # Defensive geometry/conductivity and frequency validation.
    for kwargs in ({'conductivity_S_m':.5},{'f_Hz':0}):
        try: run(**kwargs)
        except ValueError: pass
        else: raise AssertionError('invalid physical parameter accepted')
    from engine.embodied.access_map import AccessMap, access_map_to_recording
    f=fixture()
    am=AccessMap(f.spec,f.resolution,f.channel_centres_um,f.channel_centres_um,
                 channel_neurons=f.channel_neurons,channel_neurons_um=((10.,),(12.,),()),
                 soma_root_ids=f.soma_root_ids,soma_positions_vox=f.soma_positions_vox)
    legacy=access_map_to_recording(am,source_mode='soma_position',n_steps=2)
    assert legacy['reference_um'] != legacy['balancing_return_um']
    print('PASS: global other/distant field; point/area/reference; phase/probe/DC; overlap/root dedup; zero-source/current; JSON finite; legacy return/reference separation')

if __name__=='__main__':
    main()
