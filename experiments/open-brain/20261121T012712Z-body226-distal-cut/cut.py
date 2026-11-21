"""Derive a brain-side local anatomy and trace the short remaining section on the source SWCs."""
from record import OUT,BASE,geom,paths,roots,read,write,np
from common import FULL
from omnisim.modules.banc_volume import NeckVolume
from omnisim.modules.planar_cmos import cut_volume
from collections import defaultdict
from pathlib import Path

def extensions(port,anchor,plane):
    a=np.loadtxt(port['swc'])
    idx={int(n):i for i,n in enumerate(a[:,0])}
    xyz=a[:,2:5]; adj=defaultdict(list)
    for i,parent in enumerate(a[:,6]):
        if parent in idx:
            j=idx[parent];adj[i].append(j);adj[j].append(i)
    c=idx[port['crossing_children'][1]]
    parent=idx[a[c,6]]
    i,j=sorted((c,parent),key=lambda k:xyz[k,1])
    start=xyz[i]+(xyz[j]-xyz[i])*((anchor[1]-xyz[i,1])/(xyz[j,1]-xyz[i,1]))
    assert np.linalg.norm(start-anchor)<2.,(anchor,start)
    stack=[(i,j,[anchor])]; completed=[]
    while stack:
        prev,node,chain=stack.pop()
        pt=xyz[node]
        if pt[1]>=plane:
            last=chain[-1]
            hit=last+(pt-last)*((plane-last[1])/(pt[1]-last[1]))
            completed.append(np.array(chain+[hit]));continue
        if pt[1]<anchor[1]-1000:continue
        stack.extend((node,k,chain+[pt]) for k in adj[node] if k!=prev)
    return completed

def main():
    ends=geom['actual_base_m']+geom['actual_axis']*geom['actual_length_um'][:,None]*1e-6
    tip=ends[:,1].max()*1e6
    # Conservative metal envelope using maximum radius at each end (includes inactive needles).
    radius=geom['base_radius_m']*1e6
    limit=float(np.maximum(geom['actual_base_m'][:,1]*1e6+radius,ends[:,1]*1e6+radius).max())
    h=float(geom['cell_m'])*1e6
    index=int(np.ceil((limit+10)/h));plane=round(index*h,8)
    with np.load(FULL/'anatomy.npz') as data:
        volume=NeckVolume(data['labels'],data['lower'],data['roots'])
        cut=cut_volume(volume,index)
        np.savez_compressed(OUT/'cut_anatomy.npz',labels=cut.labels,lower=cut.lower,roots=np.array(cut.roots),cell_um=h,cut_um=plane,source=data['source'])
        on_face=np.unique(cut.labels[:,-1,:])
        reg=sum(v in on_face for v in range(1,53))
    bank=read(BASE/'20261117T164207Z-fixed-neck/PARAMETERS.json')['ports']
    candidates={}; status={}
    for p in paths:
        anchor=p.points_m[np.argmax(p.points_m[:,1])]*1e9
        chains=extensions(bank[p.root],anchor,plane*1000)
        status[p.root]=len(chains)
        for i,q in enumerate(chains): candidates[p.root+'_'+str(i)]=q*1e-9
    np.savez_compressed(OUT/'extensions.npz',**candidates)
    cache=FULL.parents[1]/'synapse-amr/synapses-v888-v2.npz'
    with np.load(cache) as z: sites=z['sites']
    pre=sites[:,4]/1000;post=sites[:,7]/1000
    keep=(pre<plane)&(post<plane);cutoff=(pre>=plane)&(post>=plane)
    belt=((pre>=plane-10)&(pre<plane))|((post>=plane-10)&(post<plane))
    np.savez_compressed(OUT/'synapse_masks.npz',synapse_id=sites[:,0],retained=keep,removed=cutoff,crossing=~(keep|cutoff),in_margin=belt)
    write('cut.json',dict(distal_tip_center_um=float(tip),conservative_metal_y_max_um=limit,cut_y_um=plane,min_clearance_um=plane-limit,original_shape=list(volume.labels.shape),retained_shape=list(cut.labels.shape),removed_slices=int(volume.labels.shape[1]-cut.labels.shape[1]),registered_at_retained_face=int(reg),swc_cut_branches=status,cached_synapses=len(pre),retained=int(keep.sum()),removed=int(cutoff.sum()),crossing=int((~(keep|cutoff)).sum()),in_ten_um_margin=int(belt.sum()),scope='cached circuit pairs; y-plane geometry only'))
    print('plane',plane,'counts',dict((i,list(status.values()).count(i)) for i in set(status.values())))
if __name__=='__main__':main()
