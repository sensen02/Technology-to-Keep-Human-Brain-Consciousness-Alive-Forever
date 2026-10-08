"""Run standalone recording diagnostics and deterministic acceptance checks."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from engine.electrode_recording import RecordingConfig, RecordingChain, BANDS, shared_reference_covariance
from engine.electrode_frontend import InterfaceImpedance, ContactGeometry, BOLTZMANN_J_PER_K

OUT = Path(__file__).resolve().parent / "outputs/embodied_body"

def selftest():
    passed = []
    def check(name, cond):
        if not cond:
            raise AssertionError(name)
        passed.append(name)
    def invalid(name, fn):
        try:
            fn()
        except (ValueError, TypeError):
            passed.append(name)
        else:
            raise AssertionError(name)
    for key in ("fs_Hz", "voltage_nV_rtHz", "current_fA_rtHz", "voltage_1f_corner_Hz", "current_1f_corner_Hz", "diameter_um", "pitch_um"):
        for val in (float("nan"), float("inf"), -1.):
            invalid(f"invalid {key}={val}", lambda key=key, val=val: RecordingConfig(**{key: val}))
    for kw in ({"hp_order":4}, {"lp_order":2}, {"digital_order":True}, {"fs_Hz":20000}, {"band":"bad"}, {"diameter_um":8}, {"pitch_um":21}):
        invalid(f"invalid {kw}", lambda kw=kw: RecordingConfig(**kw))
    covariance = shared_reference_covariance([[1.,2.,3.],[2.,3.,4.]], [0.5,1.])
    check("reference covariance diagonal", np.allclose(np.diagonal(covariance,axis1=-2,axis2=-1),[[1.5,2.5,3.5],[3.,4.,5.]]))
    check("reference covariance offdiagonal", np.allclose(covariance[:,0,1],[.5,1.]))
    check("reference covariance symmetric PSD", np.allclose(covariance,covariance.swapaxes(-1,-2)) and np.all(np.linalg.eigvalsh(covariance)>=0))
    check("zero reference independent", np.allclose(shared_reference_covariance([1.,2.],0.),np.diag([1.,2.])))
    for bad in (([],0.),([np.nan],0.),([1.],-1.),([1.],np.inf),([1.,2.],[1.,2.,3.])):
        invalid(f"invalid covariance {bad}", lambda bad=bad: shared_reference_covariance(*bad))
    c = RecordingChain(); f = np.array([1., 300., 5000.])
    for bad in ([0.], [-1.], [np.nan], [np.inf]):
        invalid(f"invalid noise frequency {bad}", lambda bad=bad: c.input_psds(bad))
    invalid("invalid budget density", lambda: c.budget(0))
    invalid("invalid budget bounds", lambda: c.budget(f_max=100.))
    invalid("invalid duration", lambda: c.demo(duration_s=np.nan))
    invalid("invalid current", lambda: c.demo(current_nA=np.inf))
    ideal = RecordingChain(interface=InterfaceImpedance(c.contact))
    check("ideal load H=1", np.all(ideal.interface.transfer(f)==1))
    ze = c.interface.impedance_ohm(f); rin = c.interface.amplifier_input_ohm
    check("finite load divider", np.allclose(c.interface.transfer(f),rin/(rin+ze)))
    kt = 4*BOLTZMANN_J_PER_K*c.interface.temperature_K
    p = c.input_psds(f); zl = 1/(1/ze+1/rin)
    check("passive resistor correct", np.allclose(p["input_resistor_thermal"],kt/rin*np.abs(zl)**2, rtol=1e-12, atol=0))
    check("contact loaded", np.allclose(p["contact_thermal"],kt*ze.real*np.abs(c.interface.transfer(f))**2, rtol=1e-12, atol=0))
    check("ideal Rin noise zero", np.all(ideal.input_psds(f)["input_resistor_thermal"]==0))
    check("unit area identity", np.isclose(c.contact.area_m2,c.contact.area_cm2*1e-4))
    white = RecordingChain(replace(c.config, voltage_1f_corner_Hz=0., current_1f_corner_Hz=0.))
    check("nV squared identity", np.allclose(white.input_psds(f)["amplifier_voltage"],1e-16,rtol=1e-12,atol=0))
    check("fA squared identity", np.allclose(white.input_psds(f)["amplifier_current"],1e-28*np.abs(zl)**2,rtol=1e-12,atol=0))
    check("analog HP order2 LP order4", len(c.hp_zpk[1])==2 and len(c.lp_zpk[1])==4)
    check("digital causal SOS stable", all(np.all(np.abs(np.roots(row[3:]))<1) for row in c.digital_sos))
    b = c.budget()
    check("noise sum", np.isclose(b["total_V2"],sum(b["components_V2"].values()),rtol=1e-12,atol=0))
    check("alias partition sum", np.isclose(b["total_V2"],b["baseband_V2"]+b["alias_V2"],rtol=1e-12,atol=0))
    fine = c.budget(2400)
    check("grid convergence <0.01%", abs(fine["total_V2"]/b["total_V2"]-1)<1e-4)
    extended = c.budget(1200, f_min=1e-7, f_max=3e7)
    check("tail convergence <0.01%", abs(extended["total_V2"]/b["total_V2"]-1)<1e-4)
    zero = c.demo(0.)
    check("zero signal exact", zero["signal_rms_uV"]==0 and np.all(zero["digital_V"]==0))
    demos = [c.demo(i) for i in (.01,.1,1.)]
    rms = np.array([d["signal_rms_uV"] for d in demos])
    check("current monotonic and linear", np.all(np.diff(rms)>0) and np.allclose(rms/rms[0],[1,10,100]))
    check("biphasic charge balanced", all(abs(d["discrete_net_charge_C"])<1e-24 for d in demos))
    check("source return unit transfer", 1e3<demos[0]["source_return_transfer_ohm"]<1e5)
    d4 = c.demo(.1, oversample=4); d16 = c.demo(.1, oversample=16)
    check("time analog oversampling4/8 convergence <0.1%", abs(demos[1]["signal_rms_uV"]/d4["signal_rms_uV"]-1)<.001)
    check("time analog oversampling8/16 convergence <0.1%", abs(d16["signal_rms_uV"]/demos[1]["signal_rms_uV"]-1)<.001)
    for band in BANDS:
        cb = RecordingChain(replace(c.config, band=band)).budget()
        check(f"finite {band} noise and aliases", np.isfinite(cb["noise_rms_uV"]) and cb["alias_V2"]>0)
        chain = RecordingChain(replace(c.config, band=band))
        coarse = chain.budget(2400); refined = chain.budget(4800)
        check(f"{band} total grid convergence <0.01%", abs(refined["total_V2"]/coarse["total_V2"]-1)<1e-4)
        check(f"{band} alias grid convergence <0.1%", abs(refined["alias_V2"]/coarse["alias_V2"]-1)<1e-3)
        expanded = chain.budget(2400, f_min=1e-7, f_max=3e7)
        check(f"{band} alias tail convergence <0.01%", abs(expanded["alias_V2"]/coarse["alias_V2"]-1)<1e-4)
    return passed

def main():
    tests = selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    c = RecordingChain()
    bands = {name: RecordingChain(replace(c.config,band=name)).budget(2400) for name in BANDS}
    convergence = [c.budget(n) for n in (300,600,1200,2400)]
    tail = [c.budget(2400,f_min=lo,f_max=hi) for lo,hi in ((1e-5,3e5),(1e-6,3e6),(1e-7,3e7))]
    sweep = []
    for band in BANDS:
        for en in (3.,10.,30.):
            for inoise in (1.,10.,100.):
                for corner in (0.,10.,100.,1000.):
                    chain = RecordingChain(replace(c.config,band=band,voltage_nV_rtHz=en,current_fA_rtHz=inoise,voltage_1f_corner_Hz=corner,current_1f_corner_Hz=corner))
                    sweep.append({"band":band,"ASSUMED_voltage_nV_rtHz":en,"ASSUMED_current_fA_rtHz":inoise,"ASSUMED_both_1f_corners_Hz":corner,"noise_rms_uV":chain.budget(600)["noise_rms_uV"]})
    demos = [c.demo(i) for i in (.01,.1,1.)]
    rows = []
    for band in BANDS:
        chain = RecordingChain(replace(c.config,band=band))
        for i in (.01,.1,1.):
            d = chain.demo(i)
            rows.append({k:v for k,v in d.items() if not isinstance(v,np.ndarray)} | {"band":band,"noise_rms_uV":bands[band]["noise_rms_uV"],"RMS_over_RMS_SNR":d["signal_rms_uV"]/bands[band]["noise_rms_uV"]})
    report = {"status":"hand-built tested engineering diagnostic, not validated hardware or biology",
              "reference_covariance":{"helper":"diag(channel_PSD)+reference_PSD*11T; independent channels/shared unity-gain reference", "integrated_into_budget":False,"demonstration_V2_per_Hz":shared_reference_covariance([1e-16,2e-16],.5e-16).tolist(),"CMRR":"not implemented"},
              "recording_only":True,"fixed_geometry":c.contact.as_dict()|{"pitch_um":20.},
              "chain":{"sampling_Hz":30000,"analog_HP_order":2,"analog_LP_order":4,"digital_bandpass_prototype_order":2,"digital":"causal SOS, not anti-alias","ADC":"ideal sampling only; quantization/clipping/jitter planned"},
              "noise_assumptions":{"voltage_white_nV_rtHz":[3,10,30],"current_white_fA_rtHz":[1,10,100],"both_1f_corner_Hz":[0,10,100,1000],"nominal":[10,10,100],"uncorrelated_sources":True},
              "interface_assumptions":{"Rin_ohm":1e9,"temperature_K":300,"conductivity_S_m":.3,"c_dl_uF_per_cm2":20,"rho_ct_ohm_cm2":385,"sweeps":"interface sweeps owned by frontend; these are nominal assumptions, not measured here"},
              "signal_diagnostic":"area averaged homogeneous ohmic source at z20um, equal opposite return z120um; temporally charge-balanced biphasic toy .01/.1/1nA, not fly membrane currents",
              "RMS_definition":"full 2s signal window, including silence/tails; compared to stationary expected noise RMS for same chain/window, not peak amplitude",
              "bands":bands,"convergence":convergence,"tail_convergence":tail,"ASSUMED_amplifier_sweep":sweep,"signal_comparison":rows,"tests":{"passed":len(tests),"checks":tests}}
    (OUT/"electrode_recording.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    f=np.geomspace(.01,1e6,3000)
    for band in BANDS:
        cc=RecordingChain(replace(c.config,band=band))
        axs[0,0].semilogx(f,20*np.log10(np.maximum(np.abs(cc.analog_response(f)),1e-15)),label=band)
    axs[0,0].axvline(15000,color="k",ls="--",label="Nyquist 15 kHz")
    axs[0,0].set(xlabel="Frequency (Hz)",ylabel="Analog gain (dB)",ylim=(-100,2),title="Before ADC: HP2 + LP4 Butterworth")
    axs[0,0].legend()
    for k,v in c.input_psds(f).items():
        axs[0,1].loglog(f,np.sqrt(v)*1e9,label=k.replace("_"," "))
    axs[0,1].set(xlabel="Frequency (Hz)",ylabel="Input ASD (nV/√Hz)",title="Loaded input noises (ASSUMED amplifier)")
    axs[0,1].legend(fontsize=8)
    d=demos[1]; mask=(d["time_s"]>=.028)&(d["time_s"]<.038)
    for key,label in (("raw_V","raw area average"),("analog_V","analog sampled"),("digital_V","causal digital")):
        axs[1,0].plot(d["time_s"][mask]*1000,d[key][mask]*1e6,label=label)
    axs[1,0].set(xlabel="Time (ms)",ylabel="Voltage (µV)",title="0.1 nA biphasic source + opposite return")
    axs[1,0].legend()
    for band in BANDS:
        r=[row for row in rows if row["band"]==band]
        axs[1,1].loglog([x["toy_peak_current_nA"] for x in r],[x["signal_rms_uV"] for x in r],"o-",label=f"{band} signal RMS")
        axs[1,1].axhline(bands[band]["noise_rms_uV"],ls="--",label=f"{band} noise RMS")
    axs[1,1].set(xlabel="Toy current (nA), NOT measured fly current",ylabel="RMS voltage (µV)",title="Identical chain; full 2 s observation")
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat: ax.grid(alpha=.2)
    fig.suptitle("Fixed 7 µm diameter / 20 µm pitch • recording-only engineering diagnostic\nIdeal ADC; no reference covariance / CMRR / biological validation")
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(OUT/"electrode_recording.png",dpi=150); plt.close(fig)
    print(json.dumps({"tests_passed":len(tests),"bands":bands,"signal_comparison":rows,"outputs":[str(OUT/"electrode_recording.json"),str(OUT/"electrode_recording.png")]},indent=2))

if __name__ == "__main__": main()
