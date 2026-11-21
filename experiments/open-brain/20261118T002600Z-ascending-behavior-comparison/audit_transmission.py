"""Independent record audit: native brain DN events really crossed the fixed neck."""
from neural_runtime import *


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    parameters=read(FIXED/'PARAMETERS.json')['ports']
    catalog=read(BIDIR/'CATALOG.json')['ports']
    bound=read(FIT/'BINDING.json')['records'];an_roots=[r['banc_root'] for r in bound]
    sources_state=torch.load(FIT/'FIT/update250.pt',weights_only=True,map_location='cpu')['state']
    anchors=read(BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json')['probes']
    up_targets={r for root in an_roots for r,_ in catalog[root]['outgoing']}
    motor_probes={r for roots in anchors.values() for r in roots}
    assert not up_targets & motor_probes
    records={}
    for case in ('FORMAL/seed10901','FORMAL/seed11311','REPLAY/seed10901'):
        folder=HERE/case;meta=read(folder/'RESULT.json');verify(meta['source_sha256'])
        with np.load(folder/'NEURAL.npz',allow_pickle=False) as a:
            values={k:a[k].copy() for k in a.files}
        checks={}
        for i,root in enumerate(meta['dn_roots']):
            delay=parameters[root]['delay_ticks']+1
            checks['DN_'+root]=bool(np.array_equal(values['dn_emitted'][:-delay,:,i],values['dn_arrived'][delay:,:,i]) and not values['dn_arrived'][:delay,:,i].any())
        for i,root in enumerate(an_roots):
            delay=parameters[root]['delay_ticks']
            checks['AN_'+root]=bool(np.array_equal(values['an_emitted'][:-delay,:,i],values['an_arrived'][delay:,:,i]) and not values['an_arrived'][:delay,:,i].any())
        f=np.zeros((2,3));features=np.zeros((len(values['rates']),2,3));alpha=-np.expm1(-DT/np.array([.005,.020,.050]))
        for tick,arrivals in enumerate(values['an_arrived'][:,2]):
            f+=alpha*(arrivals[:,None]/DT-f);features[tick]=f
        x=torch.from_numpy(features).double();level=(x*sources_state['mix_logits'].softmax(-1)).sum(-1)
        gain=sources_state['log_gain'].clamp(-2,2).exp();predicted=(1000*(-torch.expm1(-gain*level/1000))).numpy()
        rate_error=float(np.max(abs(predicted-values['rates'][:,2])))
        checks['candidate_rates_from_postneck_only']=rate_error<=1e-12
        events=np.zeros((len(predicted),2),dtype=bool);phase=np.zeros(2)
        for tick,row in enumerate(predicted):phase+=row*DT;events[tick]=phase>=1.;phase-=events[tick]
        checks['candidate_brain_input_events']=bool(np.array_equal(events,values['brain_an_events'][:,2]))
        checks['no_up_brain_events_zero']=not bool(values['brain_an_events'][:,3].any())
        records[case]=dict(pass_=all(checks.values()),checks=checks,postneck_rate_max_error_hz=rate_error)
    write(HERE/'TRANSMISSION_AUDIT.json',dict(pass_=all(r['pass_'] for r in records.values()),records=records,
        dn_latency='per-port measured propagation plus one explicit .1ms inter-module scheduling step',
        an_direct_to_motor_readout_overlap=0,new_simulation_steps=0,optimizer_updates=0))
    print('actual live DN/AN transmission and postneck-only encoder',all(r['pass_'] for r in records.values()),flush=True)


if __name__=='__main__':main()
