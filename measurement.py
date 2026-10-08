"""Annular fluorescence half-height radius, adapted from upstream fig2.nb.
Raster approximation at 1.6 pixels/um; black postinjury ablated core.
No threshold fitting. Missing outer crossing is censored, not clipped.
"""
import numpy as np
from matplotlib.path import Path
from scipy.sparse import csr_matrix


def fluorescence(c):
    n, k, c0 = 2.96, .167, .1
    lo, hi = .09971653246990764, .14249861653280668
    return ((c**n-c0**n)*hi+(c0**n+k**n)*lo)/(c**n+k**n)


def measure(c, baseline_c, polygons, gcamp, ablated, pixels_per_um=1.6):
    """Return radii, profiles, physical bin centers and censor mask.

    Complete centered annuli only; original interpolated pixel-index convention
    replaced by physical bin centers and piecewise linear outer crossing.
    """
    ppu=pixels_per_um
    vertices=np.concatenate(polygons)
    extent=np.max(np.abs(vertices))+1
    axis=np.arange(-extent,extent,1/ppu)+.5/ppu
    xx,yy=np.meshgrid(axis,axis)
    points=np.c_[xx.ravel(),yy.ravel()]
    labels=np.full(len(points),-1,dtype=int)
    for i,poly in enumerate(polygons):
        inside=Path(np.vstack([poly,poly[0]])).contains_points(points)
        labels[inside]=i
    bins=np.floor(np.linalg.norm(points,axis=1)*ppu).astype(int)
    # Stop before first incomplete annulus; avoid rectangular-domain artifacts.
    complete=[]
    for b in range(1,int(extent*ppu)):
        hit=bins==b
        if hit.any() and np.all(labels[hit]>=0): complete.append(b)
        else: break
    if not complete: raise ValueError('No complete annuli in tissue')
    maxbin=complete[-1]
    keep=(labels>=0)&(bins<=maxbin)
    weights=csr_matrix((np.ones(keep.sum()),(bins[keep],labels[keep])),shape=(maxbin+1,len(polygons)))
    counts=np.asarray(weights.sum(axis=1)).ravel()
    f=fluorescence(np.asarray(c))*np.asarray(gcamp)
    # Static expression heterogeneity is not an injury-induced response.
    # This numerical no-response gate is distinct from the half-height threshold.
    baseline_f=fluorescence(baseline_c)*gcamp
    induced=np.max(np.abs(f[:,~ablated]-baseline_f[~ablated]),axis=1)>1e-6
    f[:,ablated]=0
    profiles=np.asarray(weights@f.T).T/np.maximum(counts,1)
    baseline=np.asarray(weights@(fluorescence(baseline_c)*gcamp))/np.maximum(counts,1)
    r=(np.arange(maxbin+1)+.5)/ppu
    base=float(np.mean(baseline[9:]))
    radii=[]; censored=[]; reasons=[]
    # Distinct censor causes are recorded separately, never merged into one flag.
    for frame,prof in enumerate(profiles):
        if not induced[frame]:
            radii.append(np.nan); censored.append(True); reasons.append('no_response'); continue
        peak=9+int(np.argmax(prof[9:]))
        half=.5*(base+prof[peak])
        value=np.nan; reason='ok'
        if prof[peak]<=base+1e-9:
            reason='peak_not_above_baseline'
        else:
            for j in range(peak,len(r)-1):
                if prof[j]>=half and prof[j+1]<half:
                    value=r[j]+(r[j+1]-r[j])*(half-prof[j])/(prof[j+1]-prof[j]); break
            if not np.isfinite(value):
                # Descending half-height crossing lies beyond the sampled field.
                reason='outer_crossing_beyond_field'
        radii.append(value); censored.append(not np.isfinite(value)); reasons.append(reason)
    return {'radius':np.array(radii),'profiles':profiles,'r':r,'censored':np.array(censored),
            'censor_reason':np.array(reasons),'baseline':base,
            'max_complete_radius':float(r[-1]),'pixels_per_um':ppu}
