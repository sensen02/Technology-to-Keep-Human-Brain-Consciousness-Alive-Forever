"""Fixed conductance-based neurons on reviewed BANC cervical pathways."""
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

import os
HERE=Path(os.environ['P1_ARCHIVE_ROOT']).resolve()/'experiments/open-brain/20261117T164207Z-fixed-neck'
PARENT=HERE.parent/'20261117T051249Z-bidirectional'
sys.path.insert(0,str(PARENT))
from circuit import Circuit as LegacyCircuit
from build_catalog import hash_file

DT_MS=.1
RELEASE_MS=.3
RECEPTORS=('acetylcholine','gaba','glutamate')
RISE_MS=np.array([.2,.5,.5])
DECAY_MS=np.array([1.1,5.4,5.])
REVERSAL_MV=np.array([0.,-60.,-60.])
REST_MV=-52.
RESET_MV=-52.
THRESHOLD_MV=-45.
CAPACITANCE_PF=20.
LEAK_NS=1.
REFRACTORY_MS=2.2
# Charge chosen to match the legacy per-synapse small-signal membrane integral at rest.
CHARGE_NS_MS=.275*5.*CAPACITANCE_PF/20./np.abs(REVERSAL_MV-REST_MV)


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


@lru_cache(maxsize=1)
def full_catalog():
    return json.loads((PARENT/'CATALOG.json').read_text())


@lru_cache(maxsize=1)
def parameters():
    return json.loads((HERE/'PARAMETERS.json').read_text())


def ancestor_path(tree,start,end):
    result=[]
    node=start
    seen=set()
    while node in tree and node not in seen:
        result.append(node)
        if node==end:
            return result
        seen.add(node)
        node=tree[node][0]
    return None


def measure_path(port):
    path=PARENT/'geometry'/port['object']['filename']
    assert hash_file(path)==port['object']['sha256']
    values=np.loadtxt(path,comments='#')
    tree={int(row[0]):(int(row[6]),row[2:5]) for row in values}
    a,b=port['geometry']['crossing_children']
    chain=ancestor_path(tree,b,a) or ancestor_path(tree,a,b)
    assert chain,(a,b,path)
    point0=np.asarray(port['geometry']['front_nm'])
    point1=np.asarray(port['geometry']['rear_nm'])
    # Both boundary markers lie on their child's parent edge. Walk the actual
    # branch between them, including its interior curves in every direction.
    low,high=(point1,point0) if chain[0]==b else (point0,point1)
    coords=np.vstack([low,*[tree[node][1] for node in chain[1:]],high])
    length=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum()/1000.)
    direct=float(np.linalg.norm(np.asarray(point1)-point0)/1000.)
    assert length>=direct-1e-8 and length<1000.
    return dict(length_um=length,straight_um=direct,swc=str(path),sha256=hash_file(path),
                crossing_children=[a,b],radius_used=False)


def build_parameters():
    catalog=full_catalog()
    ports={}
    for root,port in catalog['ports'].items():
        path=measure_path(port)
        velocity=.10
        physical_ms=path['length_um']*.001/velocity
        ticks=int(math.ceil(physical_ms/DT_MS-1e-12))
        ports[root]=dict(path,velocity_m_s=velocity,velocity_basis='engineering prior; no cell-specific measurement',
                         propagation_ms=physical_ms,delay_ticks=ticks,
                         implemented_delay_ms=ticks*DT_MS,uncertainty_velocity_m_s=[.05,.20],
                         unresolved=['spike-initiation-to-anterior-plane','posterior-plane-to-synaptic-terminal'])
    sources={
      'kinetics':'https://pmc.ncbi.nlm.nih.gov/articles/PMC13126034/',
      'cholinergic_rise':'https://projects.iq.harvard.edu/files/wilson-lab/files/gouwenswilson2009.pdf',
      'giant_fiber_only_velocity_context':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6709211/',
      'upstream_baseline':'https://pmc.ncbi.nlm.nih.gov/articles/PMC11821855/'}
    write(HERE/'PARAMETERS.json',dict(dt_ms=DT_MS,ports=ports,release_delay_ms=RELEASE_MS,
                receptor_names=RECEPTORS,rise_ms=RISE_MS.tolist(),decay_ms=DECAY_MS.tolist(),
                reversal_mv=REVERSAL_MV.tolist(),charge_ns_ms=CHARGE_NS_MS.tolist(),
                capacitance_pf=CAPACITANCE_PF,leak_ns=LEAK_NS,rest_mv=REST_MV,reset_mv=RESET_MV,
                threshold_mv=THRESHOLD_MV,refractory_ms=REFRACTORY_MS,
                fixed=True,plasticity=False,sources=sources,
                basis={'structure':'measured from registered SWC and BANC counts',
                       'kinetics':'cross-cell literature assumptions; inhibitory rise engineering assumption',
                       'membrane':'Eon passive timescale with engineering capacitance',
                       'conductance_amplitude':'legacy small-signal area compatibility at resting voltage',
                       'delays':'only between registered cervical planes; distal and proximal delays unknown'}))


class FixedCircuit:
    """Same public event interface with per-port transit and fixed membrane currents."""
    def __init__(self,catalog,batch=1,dt_ms=DT_MS,velocity_multiplier=1.,max_extra_delay=0):
        legacy=LegacyCircuit(catalog,batch)
        for name in ('ports','port_roots','roots','index','port_index','port_columns','receptor_roots',
                     'brain_roots','allowed_drive_roots','allowed_drive_columns','brain_columns','dn_ports'):
            setattr(self,name,getattr(legacy,name))
        del legacy
        self.batch=batch
        self.dt_ms=dt_ms
        self.neck_delay=np.array([math.ceil(parameters()['ports'][root]['propagation_ms']/(
                                  velocity_multiplier*dt_ms)-1e-12) for root in self.port_roots],dtype=int)
        self.max_extra_delay=int(max_extra_delay)
        if self.max_extra_delay<0:raise ValueError('negative delay budget')
        self._pending=np.zeros((int(self.neck_delay.max())+self.max_extra_delay+1,batch,len(self.ports)),dtype=bool)
        self._tick=0
        self._voltage=np.full((batch,len(self.roots)),REST_MV,dtype=np.float64)
        self._refrac=np.zeros_like(self._voltage)
        self._spikes=np.zeros_like(self._voltage,dtype=bool)
        self._rise=np.zeros((3,batch,len(self.roots)),dtype=np.float64)
        self._decay=np.zeros_like(self._rise)
        self._synaptic_queue=np.zeros((max(1,math.ceil(RELEASE_MS/dt_ms)),3,batch,len(self.roots)))
        self._srise=np.exp(-dt_ms/RISE_MS)[:,None,None]
        self._sdecay=np.exp(-dt_ms/DECAY_MS)[:,None,None]
        self._avg_rise=(RISE_MS/dt_ms*(1.-self._srise[:,0,0]))[:,None,None]
        self._avg_decay=(DECAY_MS/dt_ms*(1.-self._sdecay[:,0,0]))[:,None,None]
        self.receptor_groups=self._receptor_identity()
        self.local=[self._matrix(kind,False) for kind in RECEPTORS]
        self.neck=[self._matrix(kind,True) for kind in RECEPTORS]
        self.last_conductance=np.zeros_like(self._rise)
        self.last_current=np.zeros_like(self._rise)
        self.pre_reset_voltage=self._voltage.copy()
        self.params_sha256=hash_file(HERE/'PARAMETERS.json')

    def _receptor_identity(self):
        # Every current source in this contracted graph has an auditable identity.
        catalog=full_catalog()['ports']
        result={root:row['transmitter'].lower() for root,row in catalog.items()}
        import pyarrow.feather as feather
        metadata=feather.read_table(PARENT.parents[2]/'data/banc888/banc_888_meta.feather',
                    columns=['banc_888_id','neurotransmitter_verified','neurotransmitter_predicted'])
        candidates=set(self.receptor_roots)
        for row in metadata.to_pylist():
            root=row['banc_888_id']
            if root in candidates:
                verified=row['neurotransmitter_verified'] or row['neurotransmitter_predicted']
                if verified:
                    result[root]=verified.lower()
        missing=self.receptor_roots-result.keys()
        assert not missing,missing
        assert all(result[r] in RECEPTORS for r in self.receptor_roots|set(self.port_roots))
        return result

    def _matrix(self,kind,cross):
        pres,posts,counts=[],[],[]
        for root,port in self.ports.items():
            edges=port['outgoing'] if cross else port['incoming_receptors']
            for other,count in edges:
                pre,post=(root,other) if cross else (other,root)
                if self.receptor_groups[pre]==kind:
                    pres.append(self.port_index[pre] if cross else self.index[pre])
                    posts.append(self.index[post])
                    counts.append(count)
        cols=len(self.ports) if cross else len(self.roots)
        return csr_matrix((np.array(counts,dtype=float),(posts,pres)),shape=(len(self.roots),cols))

    def validate_inputs(self,roots,side):
        return LegacyCircuit.validate_inputs(self,roots,side)

    def update_port_sources(self, drive, arrivals):
        """Allow an explicitly declared source-cell integrator before axon transit."""

    def step(self,drive,cut,eon_output=None,extra_delay=None):
        if drive.shape!=(self.batch,len(self.roots)) or drive.dtype!=np.float32 or not np.isfinite(drive).all():
            raise ValueError('drive must be finite float32 and indexed by registered cells')
        valid_shapes = ((len(self.ports),), (self.batch, len(self.ports)))
        if cut.shape not in valid_shapes or cut.dtype!=np.bool_:
            raise ValueError('cut must index the registered cervical axons')
        active=set(np.flatnonzero(np.any(drive!=0,axis=0)))
        if not active<=self.allowed_drive_columns:
            raise ValueError('driving a downstream cell bypasses the anatomical circuit')
        if eon_output is not None:
            if eon_output.shape!=(self.batch,len(self.ports)) or eon_output.dtype!=np.bool_:
                raise ValueError('invalid external source events')
            columns=np.flatnonzero(eon_output.any(axis=0))
            if any(self.port_roots[c] not in self.brain_roots for c in columns) or active&self.brain_columns:
                raise ValueError('external source is restricted to DN roots and exclusive of DN injection')
        cut = np.broadcast_to(cut, (self.batch, len(self.ports)))
        self._pending &= ~cut[None, :, :]
        head=self._tick%len(self._pending)
        arrivals=self._pending[head].copy()
        self._pending[head]=False
        slot=self._tick%len(self._synaptic_queue)
        # Each kernel receives its total integrated conductance only after release.
        due=self._synaptic_queue[slot].copy()
        self._synaptic_queue[slot]=0.
        for k,(local,neck) in enumerate(zip(self.local,self.neck,strict=True)):
            counts=(local@self._spikes.T).T+(neck@arrivals.T).T
            self._synaptic_queue[slot,k]=counts*CHARGE_NS_MS[k]
        increment=due/(DECAY_MS-RISE_MS)[:,None,None]
        self._rise+=increment
        self._decay+=increment
        g=np.maximum(0.,self._decay*self._avg_decay-self._rise*self._avg_rise)
        total=LEAK_NS+g.sum(axis=0)
        equilibrium=(LEAK_NS*REST_MV+(g*REVERSAL_MV[:,None,None]).sum(axis=0))/total
        effective_v=self._voltage+drive
        updated=equilibrium+(effective_v-equilibrium)*np.exp(-self.dt_ms*total/CAPACITANCE_PF)
        refractory=self._refrac>1e-12
        self._refrac=np.maximum(0.,self._refrac-self.dt_ms)
        updated[refractory]=RESET_MV
        self.pre_reset_voltage=updated.copy()
        self._spikes=(updated>=THRESHOLD_MV)&~refractory
        self._refrac[self._spikes]=REFRACTORY_MS
        self._voltage=np.where(self._spikes,RESET_MV,updated)
        self.last_conductance=g.copy()
        self.last_current=g*(REVERSAL_MV[:,None,None]-self._voltage[None,:,:])
        self._rise*=self._srise
        self._decay*=self._sdecay
        self.update_port_sources(drive, arrivals)
        events=self._spikes[:,self.port_columns].copy()
        if eon_output is not None:
            events[:,self.dn_ports]=eon_output[:,self.dn_ports]
        delay_at_emission=np.broadcast_to(self.neck_delay,cut.shape)
        if extra_delay is not None:
            if (extra_delay.shape!=cut.shape or not np.issubdtype(extra_delay.dtype,np.integer)
                    or (extra_delay<0).any() or (extra_delay>self.max_extra_delay).any()):
                raise ValueError('invalid lane-port delay')
            delay_at_emission=delay_at_emission+extra_delay
        active_events=events & ~cut
        for delay in np.unique(delay_at_emission[active_events]):
            target=(self._tick+int(delay))%len(self._pending)
            sending=active_events & (delay_at_emission==delay)
            if (self._pending[target] & sending).any():
                raise ValueError('two events collide in the same registered axon tick')
            self._pending[target] |= sending
        self._tick+=1
        return events,arrivals,self._spikes.copy(),self._voltage.copy()


if __name__=='__main__':
    build_parameters()
