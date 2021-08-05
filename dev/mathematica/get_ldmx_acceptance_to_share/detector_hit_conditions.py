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

def _det_hit_condition(_pa, _pr, det_rad, zmax, xv, Ethr=1.):
    """
    returns true if lepton hits a circular detector of radius det_rad, 
    if it originates from a vector that decays a distance xv from the detector
    pa = relativistic 4 vector momentum of the axion 
    pr = relativistic 4 lepton momentum of the recoil electron
    det_rad = detector radius in cm
    xv = z distance of the vector decay vertex from the detector in cm
    """
    
    #Ethr = 1. # E137 Ecal detector threshold energy
    
    pa = np.array(_pa)
    pr = np.array(_pr)
    # xv < 0 corresponds to decays beyond the detector
    if xv < 0:
        return False
    
    
    #return (pl[0] >= Ethr) and (np.dot(rvec,rvec) < (det_rad)**2.)
    return (pr[0] <= Ethr)
    
# Vectorized version of the above
def det_hit_condition(pv, pl, det_rad, zmax, xvs, Ethr=1.):
    if not isinstance(xvs, collections.Iterable):
        xvs = np.array([xvs])
        
    return np.array([_det_hit_condition(pv, pl, det_rad, zmax, xv, Ethr) for xv in xvs])
