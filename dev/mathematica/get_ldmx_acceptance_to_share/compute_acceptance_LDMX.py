import numpy as np
import sys
from detector_hit_conditions import *

def get_acceptance_from_four_vectors(ma, ctau, four_vector_list, zmin, zmax, det_rad, Ethr=1.):
    evt_fraction_detected = []
    for pa, pr in four_vector_list:
        if pa[3] < 0:
            continue

        gct = (pa[3]/ma)*ctau

        np.seterr(over='raise')
        try: 
            xi_high = np.exp(-zmin/gct)
            xi_low = np.exp(-zmax/gct)
        except:
            print "zmin, zmax, gct = ", zmin, zmax, gct
            sys.exit(0)
        
        f = 0.
        fs = []
        if xi_high > 0.:
            
            xis = np.linspace(xi_low,xi_high,num=100.)
            xvs = xv_from_uni(xis,zmax,gct)

            coef = (xi_high-xi_low)/len(xis)
            #fs.append(coef*np.sum(det_hit_condition(pa, pr, det_rad, zmax, xvs, Ethr).astype('float')))

            #f = max(fs)
            f = coef*np.sum(det_hit_condition(pa, pr, det_rad, zmax, xvs, Ethr).astype('float'))

            
        evt_fraction_detected.append(f)

    evt_fraction_detected = np.array(evt_fraction_detected)
    
    return np.sum(evt_fraction_detected)/len(evt_fraction_detected)
