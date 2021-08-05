import re
import os
import decimal
import argparse
import reformat
import numpy as np
from sympy import Symbol
from sympy.stats import sample, Uniform, Exponential
import matplotlib.pyplot as plt

def eps_dls():

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='infile', help='input file')
    #parser.add_argument('-o', action='store', dest='outdir', help='output dir')
    parser.add_argument('-m', action='store', dest='mass',   help='Ap mass')
    parser.add_argument('-s', action='store', dest='seed',  help='seed')
    args = parser.parse_args()
    
    # Sampling prereqs
    mAp = float( args.mass )
    # Looks like this is 10X, not sure if eps should = 1 or 0.1
    epss = np.logspace( -5, -3, 3 )
    decay_widths = [ reformat.gamma_ap_tot(mAp, eps) for eps in epss]
    
    # Decay time
    c_speed = 299_792_458_000 # mm/s 
    t = Symbol('t')
    # hbar = 6.582e-25 GeV*s
    taus = [6.582e-25 / decay_width  for decay_width in decay_widths]
    decay_rvs = [Exponential(t, 1/tau) for tau in taus]
    decay_ts = [ sample(
                        decay_rv,
                        numsamples=10_000,
                        seed=np.random.seed( int( args.seed ) )
                    ) for decay_rv in decay_rvs]
    
    # Decay lengths
    dls = [ [] for eps in epss ]
    
    indx = 0
    for indx in range(len(epss)):
    
        # Open original and output files
        with open(args.infile, 'r') as ogfile:
    
            ##################################################
            # Edit header (techincaly until </init>
            # Many conditions shouldn't check in events sec.
            ##################################################
            for line in ogfile:
        
                # Break from header/init
                if line == '</init>\n':
                    break
                
            ##################################################
            # Edit events
            ##################################################
            event_num = 0
            event_line = 0
            current_line = 0
            for line in ogfile: # Picks up where last loop leaves off
                current_line += 1
        
                # Scale relevant lines
                if line == '<event>\n':
                    event_num += 1
                    event_line = 0
                    if event_num % 1000 == 0:
                        print( 'Reformatting event: {}'.format(event_num) )
        
                else: event_line += 1
                if 1 < event_line < 9:
        
                    line = reformat.rescaleLine(line, tokens=range(6,11))
        
                    
                if event_line == 6: # Take note of Ap info for projection
                    px,py,pz,en = [
                                    float(v) for v in reformat.numsInLine(line)[6:10] 
                                    ]
                    Ap_3mom = np.array((px,py,pz))
        
                # Skip mgrwt. add appripriate vertex, and end event
                elif event_line == 16 :
    
                    # Add verticies
                    t = next(decay_ts[indx]) * (en/mAp)
        
                    dls[indx].append( Ap_3mom[2]*c_speed / mAp * t )
        
        cut = 4000
        plt.hist(
                    np.clip(dls[indx], 0, cut),
                    histtype='step',
                    log=True,
                    bins=50,
                    range = (0,cut),
                    density=True,
                    label='eps = {}'.format(epss[indx])
                    )
        indx += 1
    
    plt.legend(loc='upper center')
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')
    plt.title(r"$m_{A'} =$" + str(mAp) + " GeV")
    plt.show()
