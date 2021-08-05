import numpy as np
import collections

def _xv_from_uni(xi,zmax,gct):
    """
    Generate a z vertex displacement from a uniform random variable 
    both zmax and gct should be in cm
    """
    if xi > 0.:
        return gct*np.log(xi) + zmax
    else:
        return -100.*zmax
    
xv_from_uni = np.vectorize(_xv_from_uni)

def _det_hit_condition(_pv, _pl, det_rad, zmax, xv, Ethr=1.):
    """
    returns true if lepton hits a circular detector of radius det_rad, 
    if it originates from a vector that decays a distance xv from the detector
    pv = relativistic 4 vector momentum
    pl = relativistic 4 lepton momentum
    det_rad = detector radius in cm
    xv = z distance of the vector decay vertex from the detector in cm
    """
    
    #Ethr = 1. # E137 Ecal detector threshold energy
    
    pv = np.array(_pv)
    pl = np.array(_pl)
    # xv < 0 corresponds to decays beyond the detector
    if xv < 0:
        return False
    
    #print pv
    pvt = pv[1:3]
    pvz = pv[3]
    
    plt = pl[1:3]
    plz = pl[3]
    
    # transverse displacement of vector when it decays
    #print xv#,(zmax-xv)*pvt,pvz
    vec_delta = (zmax-xv)*pvt/pvz
    
    #print pvt/pvz, np.linalg.norm(vec_delta)
    # point at which lepton momentum crosses the detector x-y plane

    rvec = vec_delta + xv*plt/plz
    
    #print rvec, np.sqrt(np.dot(rvec,rvec))
    
    return (pl[0] >= Ethr) and (np.dot(rvec,rvec) < (det_rad)**2.)
    
# Vectorized version of the above
def det_hit_condition(pv, pl, det_rad, zmax, xvs, Ethr=1.):
    if not isinstance(xvs, collections.Iterable):
        xvs = np.array([xvs])
        
    return np.array([_det_hit_condition(pv, pl, det_rad, zmax, xv, Ethr) for xv in xvs])
