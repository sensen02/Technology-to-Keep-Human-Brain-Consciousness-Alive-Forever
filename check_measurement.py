"""Independent synthetic-image checks for observation code, not biology."""
import numpy as np
from measurement import measure,fluorescence

def check():
    # 1um square pixels as cells on large tissue; construct smooth radial signal.
    axis=np.arange(-40,41,dtype=float)
    pos=np.array([(x,y) for x in axis for y in axis])
    polys=pos[:,None,:]+np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    r=np.linalg.norm(pos,axis=1)
    base=np.full(len(pos),.1)
    targetF=fluorescence(base)+.015*np.exp(-r*r/(2*10**2))
    n,k,c0,lo,hi=2.96,.167,.1,.09971653246990764,.14249861653280668
    c=(((hi-lo)*c0**n+(targetF-lo)*k**n)/(hi-targetF))**(1/n)
    result=measure(c[None,:],base,polys,np.ones(len(pos)),np.zeros(len(pos),dtype=bool))
    # Peak is searched outside inner9pixel bins, not at origin.
    peak_radius=result['r'][9]
    expected=np.sqrt(peak_radius**2+2*10**2*np.log(2))
    measured=float(result['radius'][0])
    assert abs(measured-expected)<1.,(measured,expected)
    flat=measure(base[None,:],base,polys,np.ones(len(pos)),np.zeros(len(pos),dtype=bool))
    assert flat['censored'][0]
    return {'synthetic_gaussian_radius_um':measured,'expected_with_excluded_core_um':float(expected),'error_um':abs(measured-expected),'flat_profile_censored':True}

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
