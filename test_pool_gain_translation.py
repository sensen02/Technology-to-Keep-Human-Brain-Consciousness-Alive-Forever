"""Pure arrays and synthetic binary MN spikes; no neural simulation or physics."""
import unittest
import numpy as np
from engine.embodied.pool_gain_translation import (
    ORIGINAL_FIR, MAX_GAIN, pool_gain_weights, gains_from_weights,
    make_pool_gain_translator)
from engine.embodied.motor_translation import FiniteMotorTranslator


def manifest():
    # Reversed actuator order tests canonical SORTED pool order, not row order.
    rows = [dict(actuator_index=i, local_indices=[65-i]) for i in range(66)]
    rows += [dict(actuator_index=66, local_indices=[0]),
             dict(actuator_index=67, local_indices=[])]
    return dict(n_neurons=67, n_actuators=68, rows=rows)


def consume(t, spikes, start=0.):
    return t.consume(spikes, spike_dt_s=.0005, start_s=start,
                     end_s=start+len(spikes)*.0005)[0]


class PoolGainTests(unittest.TestCase):
    def test_unit_gain_exact_original(self):
        np.testing.assert_array_equal(pool_gain_weights(np.ones(66)),
                                      np.tile(ORIGINAL_FIR, (66, 1)))
        m=manifest(); a=make_pool_gain_translator(m); b=FiniteMotorTranslator(m)
        rng=np.random.default_rng(4)
        for i in range(6):
            s=(rng.random((10,67))<.03).astype(int)
            np.testing.assert_array_equal(consume(a,s,i*.005),consume(b,s,i*.005))

    def test_each_of_66_gains_changes_only_its_row_preserves_shape(self):
        base=pool_gain_weights(np.ones(66))
        for i in range(66):
            g=np.ones(66);g[i]=1.5;w=pool_gain_weights(g)
            np.testing.assert_array_equal(np.delete(w,i,axis=0),np.delete(base,i,axis=0))
            np.testing.assert_array_equal(w[i],1.5*np.asarray(ORIGINAL_FIR))

    def test_bounds_preserve_shape_no_bin_clipping(self):
        self.assertEqual(MAX_GAIN,10.)
        w=pool_gain_weights(np.linspace(0,10,66))
        self.assertTrue(np.all((w>=0)&(w<=4)))
        np.testing.assert_array_equal(w[-1],10*np.asarray(ORIGINAL_FIR))
        np.testing.assert_array_equal(w[0],np.zeros(5))

    def test_invalid_inputs_rejected(self):
        for bad in [np.ones(65),np.ones((66,1)),np.full(66,np.nan),
                    np.full(66,np.inf),np.full(66,-.1),np.full(66,10.001)]:
            with self.assertRaises(ValueError):pool_gain_weights(bad)
        for bad in [np.zeros((65,5)),np.full((66,5),np.nan)]:
            with self.assertRaises(ValueError):gains_from_weights(bad)

    def test_general_and_shared_temporal_relationships(self):
        rng=np.random.default_rng(10);g=rng.uniform(0,10,66);w=pool_gain_weights(g)
        np.testing.assert_allclose(gains_from_weights(w),g,rtol=1e-14,atol=0)
        # Shared-time kernels belong here ONLY if proportional to original FIR.
        np.testing.assert_allclose(gains_from_weights(np.tile(2*np.asarray(ORIGINAL_FIR),(66,1))),2)
        bad=np.tile(ORIGINAL_FIR,(66,1));bad[:,1]+=.05
        with self.assertRaises(ValueError):gains_from_weights(bad)
        bad=w.copy();bad[4,3]+=.01
        with self.assertRaises(ValueError):gains_from_weights(bad)

    def test_shared_actuator_mapping_sorted_pool_identity(self):
        g=np.ones(66);g[0]=.5;t=make_pool_gain_translator(manifest(),g)
        self.assertIs(type(t),FiniteMotorTranslator)
        self.assertEqual(t.pools[0],(0,));self.assertEqual(t.actuator_pool[65],0)
        self.assertEqual(t.actuator_pool[66],0);self.assertEqual(t.actuator_pool[67],-1)
        s=np.zeros((10,67),int);s[-1,0]=1;u=consume(t,s)
        self.assertEqual(u[65],u[66]);self.assertGreater(u[65],0)
        np.testing.assert_array_equal(u[:65],np.zeros(65));self.assertEqual(u[67],0)

    def test_zero_gain_zero_spikes_and_outside_motor_whitelist(self):
        for g in (np.ones(66),np.zeros(66)):
            t=make_pool_gain_translator(manifest(),g)
            s=np.zeros((10,67),int);s[:,66]=1
            np.testing.assert_array_equal(consume(t,s),np.zeros(68))
        t=make_pool_gain_translator(manifest(),np.zeros(66))
        np.testing.assert_array_equal(consume(t,np.ones((10,67),int)),np.zeros(68))

    def test_gain_before_existing_ctrl_clip_not_after(self):
        # Low activity remains proportional; high activity obeys unchanged [0,1].
        s=np.zeros((80,67),int);s[0,0]=1
        a=consume(make_pool_gain_translator(manifest()),s)
        b=consume(make_pool_gain_translator(manifest(),np.full(66,.5)),s)
        np.testing.assert_array_equal(b,a*.5)
        saturated=np.zeros((10,67),int);saturated[-1,0]=1
        full=consume(make_pool_gain_translator(manifest()),saturated)
        half=consume(make_pool_gain_translator(manifest(),np.full(66,.5)),saturated)
        self.assertEqual(full[65],1.);self.assertAlmostEqual(half[65],.8)
        self.assertNotEqual(half[65],full[65]*.5)
        high=consume(make_pool_gain_translator(manifest(),np.full(66,10.)),np.ones((10,67),int))
        self.assertTrue(np.all((high>=0)&(high<=1)));self.assertEqual(high[0],1.)

    def test_no_mutable_gain_alias_reset_and_finite_memory(self):
        g=np.ones(66);t=make_pool_gain_translator(manifest(),g);before=t.weights
        g[:]=0;np.testing.assert_array_equal(t.weights,before)
        exposed=t.weights;exposed[:]=0;np.testing.assert_array_equal(t.weights,before)
        s=np.zeros((10,67),int);s[-1,0]=1;consume(t,s)
        np.testing.assert_array_equal(consume(t,np.zeros((80,67),int),.005),np.zeros(68))
        t.reset();np.testing.assert_array_equal(consume(t,np.zeros((10,67),int)),np.zeros(68))


if __name__=='__main__':unittest.main()
