"""Independent passive-network regression; no biological calibration claimed."""
import math
import numpy as np
from engine.electrode_frontend import (ContactGeometry, InterfaceImpedance,
    thermal_noise_variance, BOLTZMANN_J_PER_K)

def main():
    checks=[]
    def check(name, ok):
        checks.append((name,bool(ok)))
        print(('PASS' if ok else 'FAIL'),name)
    for load in (1e6,1e7,1e9,math.inf):
        z=InterfaceImpedance(ContactGeometry(7),amplifier_input_ohm=load)
        f=np.logspace(-6,8,2000); ze=z.impedance_ohm(f)
        zp=ze if math.isinf(load) else 1/(1/ze+1/load)
        psd=(z.noise_spectral_density_V_per_rtHz(f)*abs(z.transfer(f)))**2
        psd+=z.amplifier_noise_spectral_density_V_per_rtHz(f)**2
        check('fluctuation dissipation load='+str(load),np.allclose(psd,4*BOLTZMANN_J_PER_K*300*zp.real,rtol=1e-12,atol=0))
        check('signal divider load='+str(load),np.allclose(z.transfer(f),np.ones_like(ze) if math.isinf(load) else load/(load+ze),rtol=1e-12))
        d=thermal_noise_variance(z,f_lo_Hz=300,f_hi_Hz=5000,include_amplifier=True)
        check('uV conversion load='+str(load),d['rms_uV']==math.sqrt(d['variance_V2'])*1e6)
        check('API agrees load='+str(load),z.noise_rms_uV(300,5000,include_amplifier=True)==d['rms_uV'])
    z=InterfaceImpedance(ContactGeometry(7))
    d=thermal_noise_variance(z,f_lo_Hz=1e-6,f_hi_Hz=10000)
    check('historical ideal-contact 24.018uV not 24V',abs(d['rms_uV']-24.01816009741456)<1e-8)
    for n in (0,1,True,2.5):
        try:
            thermal_noise_variance(z,n_grid=n)
        except ValueError:
            check('reject grid '+str(n),True)
        else:
            check('reject grid '+str(n),False)
    assert all(ok for _,ok in checks)
    print('%d/%d PASS'%(sum(ok for _,ok in checks),len(checks)))
if __name__=='__main__': main()
