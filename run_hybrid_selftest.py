"""Hand-implemented bounded prototype verification, not calibrated validation.

Run: PYTHONDONTWRITEBYTECODE=1 venv/bin/python run_hybrid_selftest.py
Each test owns and closes its exclusive NEURON runtime. No output files written.
"""
import unittest
import numpy as np
from scipy import sparse
from engine.neural_active import ActiveRuntime, IdealCableSpec
from engine.neural_hybrid import HybridNetwork, ConductanceEdge
from engine.neural_cond import ConductanceNetwork, ConductanceParams


def trace(dt=0.025, cut=False, drive=True):
    with ActiveRuntime(dt) as rt:
        cell = rt.add_cable('ideal')
        if cut:
            cell.cut_axial(1)
            cell.cut_axial(1)  # idempotent
            assert not cell.sections[1].parentseg()
        rt.reset()
        v = []
        for k in range(round(12 / dt)):
            current = np.zeros(len(cell.segments))
            current[0] = 20.0 if drive and 1 <= k * dt < 2 else 0.0
            cell.set_inward_current_nA(current)
            rt.step()
            v.append(cell.voltage_mV())
        return np.arange(1, len(v)+1) * dt, np.asarray(v), cell.spike_count


class ActiveTests(unittest.TestCase):
    def test_conduction_and_axial_cut(self):
        t, intact, spikes = trace()
        _, cut, cut_spikes = trace(cut=True)
        proximal = t[np.flatnonzero(intact[:, 0] >= 0)[0]]
        distal = t[np.flatnonzero(intact[:, -1] >= 0)[0]]
        self.assertGreater(distal, proximal + 0.25)
        self.assertGreater(intact[:, -1].max(), 30)
        self.assertGreater(cut[:, 0].max(), 30)
        self.assertLess(cut[:, -1].max(), -60)
        self.assertEqual((spikes, cut_spikes), (1, 0))
        print(f'  physical HH propagation proximal={proximal:.3f}, distal={distal:.3f} ms; cut distal max={cut[:, -1].max():.3f} mV')

    def test_zero_input(self):
        _, v, spikes = trace(drive=False)
        self.assertEqual(spikes, 0)
        self.assertLess(np.max(np.abs(v + 65)), 0.1)

    def test_dt_convergence(self):
        _, a, _ = trace(0.025)
        _, b, _ = trace(0.0125)
        _, c, _ = trace(0.00625)
        err_ab = np.sqrt(np.mean((a[:, -1] - b[1::2, -1])**2))
        err_bc = np.sqrt(np.mean((b[:, -1] - c[1::2, -1])**2))
        self.assertLess(err_bc, 0.7 * err_ab)
        self.assertLess(err_bc, 1.5)
        print(f'  distal dt RMS errors 0.025->0.0125={err_ab:.4f}, 0.0125->0.00625={err_bc:.4f} mV')

    def test_exclusive_reset_cleanup(self):
        from neuron import h
        before = (h.dt, h.celsius, h.t, int(h.CVode().active()))
        with ActiveRuntime(0.025) as rt:
            a = rt.add_cable('a')
            b = rt.add_cable('b')
            with self.assertRaises(RuntimeError):
                ActiveRuntime()
            rt.reset()
            rt.step()
            self.assertAlmostEqual(h.t, 0.025)
            rt.reset()
            self.assertEqual(rt.t_ms, 0)
            np.testing.assert_allclose(a.voltage_mV(), -65)
            np.testing.assert_allclose(b.voltage_mV(), -65)
            h.dt = 0.01
            with self.assertRaises(RuntimeError):
                rt.step()
        self.assertEqual(list(h.allsec()), [])
        self.assertEqual(before, (h.dt, h.celsius, h.t, int(h.CVode().active())))
        foreign = h.Section(name='foreign_test')
        try:
            with self.assertRaises(RuntimeError):
                ActiveRuntime()
            self.assertEqual(len(list(h.allsec())), 1)
        finally:
            h.delete_section(sec=foreign)

    def test_extracellular_gauge_invariance(self):
        traces = []
        for offset in (0.0, 17.0):
            with ActiveRuntime() as rt:
                cell = rt.add_cable('gauge')
                rt.reset()
                cell.set_extracellular_voltage_mV(np.linspace(-2, 2, 63) + offset)
                v = []
                for _ in range(80):
                    rt.step()
                    v.append(cell.voltage_mV())
                traces.append(np.array(v))
        np.testing.assert_allclose(traces[0], traces[1], atol=1e-7, rtol=0)

    def test_current_field_and_probes(self):
        def response(field=False, inward=0.0):
            with ActiveRuntime() as rt:
                c = rt.add_cable(1)
                rt.reset()
                xyz = c.segment_xyz_um()
                self.assertEqual(xyz.shape, (63, 3))
                self.assertTrue(np.all(np.diff(xyz[:, 0]) > 0))
                cmd = np.linspace(-2, 2, 63) if field else np.zeros(63)
                c.set_extracellular_voltage_mV(cmd)
                I = np.zeros(63); I[0] = inward
                c.set_inward_current_nA(I)
                for _ in range(8):
                    rt.step()
                v = c.voltage_mV()
                np.testing.assert_allclose(c.extracellular_mV(), cmd, atol=1e-6)
                return v
        base, positive, negative, field = response(), response(inward=1), response(inward=-1), response(field=True)
        self.assertGreater(positive[0], base[0])
        self.assertLess(negative[0], base[0])
        self.assertGreater(np.max(np.abs(field - base)), 0.01)


class HybridTests(unittest.TestCase):
    def test_all_coarse_without_neuron_dependency(self):
        import subprocess
        import sys
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'neuron' or name.startswith('neuron.'):
        raise ImportError('NEURON intentionally unavailable in all-coarse test')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from engine.neural_hybrid import HybridNetwork
with HybridNetwork([1], [1], {}) as net:
    assert net.step() == set()
'''
        from pathlib import Path
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ownership_and_no_local_recurrence(self):
        for ids, point, fine in [([1,1], [1], {}), ([1,2], [1], {}),
                                 ([1], [1], {1: None}), ([1], [1,1], {})]:
            with self.assertRaises(ValueError):
                HybridNetwork(ids, point, fine)
        with HybridNetwork([1,2], [1], {2: None}) as net:
            self.assertEqual(dict(net.ownership), {1: 'point', 2: 'detailed'})
            with self.assertRaises(TypeError):
                net.ownership[1] = 'detailed'
            net.point.set_connectivity([[0.01]], [[0]])
            with self.assertRaises(RuntimeError):
                net.step()

    def test_all_coarse_equivalence(self):
        dt, p = 0.1, ConductanceParams(delay_ms=0.3)
        we = np.array([[0,0,0.003],[0.004,0,0],[0,0.005,0]])
        wi = np.array([[0,0.001,0],[0,0,0.002],[0.001,0,0]])
        ref = ConductanceNetwork(3, p, dt).set_connectivity(we, wi)
        edges = [ConductanceEdge(pre, post, r, w[post,pre], p.delay_ms)
                 for r,w in [('exc',we),('inh',wi)] for post,pre in zip(*np.nonzero(w))]
        with HybridNetwork(range(3), range(3), {}, edges, p, dt) as net:
            for k in range(300):
                I = np.array([0.5,0.2,0]) if k < 150 else np.zeros(3)
                expected = set(np.flatnonzero(ref.step(I)))
                self.assertEqual(net.step(I), expected)
                for name in ('v', 'ge', 'gi', 'adaptation', 'spike_count', 'refractory_until_ms'):
                    np.testing.assert_allclose(getattr(net.point,name), getattr(ref,name), rtol=1e-13, atol=1e-13)
            self.assertGreater(len(net.deliveries), 10)
            self.assertEqual(len({e.event_id for e in net.deliveries}), len(net.deliveries))
            for e in net.deliveries:
                self.assertAlmostEqual(e.arrival_ms-e.emitted_ms, 0.3)

    def test_single_event_end_to_start(self):
        p = ConductanceParams(delay_ms=0.3)
        edge = ConductanceEdge('a','b','exc',0.001,0.3)
        with HybridNetwork(['a','b'], ['a','b'], {}, [edge], p, 0.1) as net:
            self.assertEqual(net.step([4,0]), {'a'})  # emitted at 0.1
            for _ in range(3):  # interval starts 0.1,0.2,0.3
                net.step()
                self.assertEqual(len(net.deliveries),0)
            net.step()  # interval starts 0.4
            self.assertEqual(len(net.deliveries),1)
            e = net.deliveries[0]
            self.assertAlmostEqual(e.emitted_ms,0.1)
            self.assertAlmostEqual(e.arrival_ms,0.4)
            self.assertAlmostEqual(net.point.ge[1],0.001*np.exp(-0.1/3))
            net.step()
            self.assertEqual(len(net.deliveries),1)
            self.assertAlmostEqual(net.point.ge[1],0.001*np.exp(-0.2/3))

    def test_bidirectional_cross_tier_events(self):
        dt = 0.025
        edges = [ConductanceEdge('point','fine','exc',0.0001,0.1),
                 ConductanceEdge('point','fine','inh',0.0002,0.1),
                 ConductanceEdge('fine','point','inh',0.001,0.2)]
        with HybridNetwork(['point','fine'], ['point'], {'fine':None}, edges, dt_ms=dt) as net:
            fine = net.active.cables['fine']
            fine_g_before = None
            for k in range(480):
                I = np.zeros(len(fine.segments)); I[0] = 20 if 1<=k*dt<2 else 0
                fine.set_inward_current_nA(I)
                before = len(net.deliveries)
                net.step(20 if k==0 else 0)
                fresh = net.deliveries[before:]
                for e in fresh:
                    self.assertAlmostEqual(e.arrival_ms, k*dt)
                    if e.post=='fine':
                        actual = fine.synapses[e.receptor].g
                        tau = 3 if e.receptor=='exc' else 8
                        self.assertAlmostEqual(actual, e.increment_uS*np.exp(-dt/tau), places=11)
                        fine_g_before = actual
                    else:
                        self.assertAlmostEqual(net.point.gi[0], e.increment_uS*np.exp(-dt/8), places=11)
            self.assertIsNotNone(fine_g_before)
            self.assertEqual([gid for _,gid in net.emissions].count('point'),1)
            self.assertEqual([gid for _,gid in net.emissions].count('fine'),1)
            self.assertEqual(len(net.deliveries),3)
            self.assertEqual(len({e.event_id for e in net.deliveries}),3)
            self.assertEqual([e.edge_index for e in net.deliveries],[0,1,2])

    def test_hybrid_zero_input(self):
        with HybridNetwork(['p','f'], ['p'], {'f':None}, [ConductanceEdge('f','p','exc',0.01,0.1)]) as net:
            for _ in range(400):
                self.assertEqual(net.step(), set())
            self.assertEqual(net.deliveries,[])
            self.assertEqual(net.emissions,[])


if __name__ == '__main__':
    unittest.main(verbosity=2)
