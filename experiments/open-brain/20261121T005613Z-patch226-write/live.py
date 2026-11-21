from common import *
from omnisim.scenes.acute_implant import load_acute_state,FIRST_AXON

def main(output_name):
    g=patterns()
    world,_,_,state=load_acute_state(FULL)
    results={}; gates={}
    try:
        gains=[state['section_exposure'][r].effect(p.points_m).gain for r,p in zip(roots,paths)]
        np.testing.assert_allclose(gains,geom['route_gain'],atol=1e-12,rtol=0)
        for name,i in zip(arms,g):
            model=ElectrodeObservation(tips,[AxonProbe(p.root,p.points_m,p.speed_m_s,owner=k)
                        for k,p in enumerate(paths)],contact_radius_m=.15e-6,sensor_patches=[],
                        section_exposure=state['section_exposure'])
            world.add_module(name,model)
            seen=set(); raw=[]
            for phase in (0,1):
                model.pulse_pattern(world,(-1)**phase*i,phase*.0001,first_axon_material=FIRST_AXON)
                events=[e for e in model.events if e['kind']!='below_threshold']
                model.events=[]
                raw.extend(events)
                new={e['axon'] for e in events if e['kind']=='axon_fired'}-seen
                # All events on a second initiation of the same root fall in its refractory period.
                if phase==0:accepted=[]
                accepted.extend(e for e in events if e['axon'] in new)
                seen.update(new)
            model.events=sorted(accepted,key=lambda e:e['time_s'])
            results[name]=dict(raw=raw,accepted=accepted)
        world.initialize_device()
        world.step(.0022)
        events=world.drain_events()
        for name,expected in zip(arms,[[],AN[:1],AN[1:],AN]):
            brain_end={p.root:'a' if p.points_m[0,1]<p.points_m[-1,1] else 'b' for p in paths}
            received=[dict(kind=e.kind,time_s=e.time_s,**e.data) for e in events if e.source==name]
            assert len(received)==len(results[name]['accepted'])
            arrived=[e for e in received if e['kind']=='axon_arrived' and e['end']==brain_end[e['axon']]]
            results[name]['brain_end']=arrived
            gates[name]=sorted(expected)==sorted(e['axon'] for e in arrived)
        write(output_name,dict(active_cells=world.active_cells,observed_s=world.time_s,
             gates=gates,arms=results,check_max_gain_error=float(np.max(abs(geom['route_gain']-gains)))))
        print(gates,flush=True)
    finally:world.close()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    main('registered-replay.json' if args.replay else 'registered.json')
