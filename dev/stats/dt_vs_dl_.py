import re
import os
import decimal
import argparse
import numpy as np
from sympy import Symbol
from scipy import interpolate
from sympy.stats import sample, Uniform, Exponential
import matplotlib.pyplot as plt
import phys_form


def main():

    """ Add interactivity for LHE analysis """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', action='store', dest='mAp', type=float)
    parser.add_argument('-e', action='store', dest='eps', type=float)
    parser.add_argument('-g', action='store', dest='gamma', type=float)
    parser.add_argument('-i', action='store', dest='infile',)
    args = parser.parse_args()

    # Constants
    mAp, eps, gamma = args.mAp, args.eps, args.gamma
    hbar = 6.582e-25 # GeV*s
    c_speed = 299_792_458_000 # mm/s

    # Detector limits
    zmin = 300
    zmax = 4000 - 300

    # Will store information here
    nEvents = 10_000
    dt_zs = []
    dl_zs = []
    report = {
            }

    ##################################################
    # Analysis
    ##################################################

    # Decay time
    t = Symbol('t')
    decay_width = phys_form.gamma_ap_tot(mAp, eps)
    tau = hbar / decay_width
    decay_t_rv = Exponential(t, 1/tau)
    decay_t = sample(
                        decay_t_rv,
                        numsamples=nEvents,
                        seed=np.random.seed( 2 )
                        )

    # Decay z
    z = Symbol('z')
    gctau = gamma*c_speed*tau
    print(gctau)
    if True: return
    decay_l_rv = Exponential(z, 1/gctau)
    decay_l = sample(
                        decay_l_rv,
                        numsamples=nEvents,
                        seed=np.random.seed( 2 )
                        )

    # Open file for analysis
    with open(args.infile, 'r') as ogfile:
    
        # Skip from header/init
        for line in ogfile:
            if line == '</init>\n':
                break
        
        ##################################################
        #  events
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

            if event_line == 6: # Take note of Ap info for projection
                line = phys_form.rescaleLine(line, tokens=range(6,11))
                px,py,pz,en = [
                                float(v) for v in phys_form.numsInLine(line)[6:10] 
                                ]
                Ap_3mom = np.array((px,py,pz))

                # Decay time
                t = next(decay_t)*(en/mAp)
                dt_zs.append( Ap_3mom[2]*c_speed / mAp * t)

                # Decay length
                l = next(decay_l)
                dl_zs.append( phys_form.dot(
                                            phys_form.unit(Ap_3mom) * l,
                                            (0,0,1)
                                            )
                                )

    # Fill report
    #report['something'] = something

    # Print report
    for k,v in report.items():
        print( '{}: {}'.format(k,v) )

    plot( Times=dt_zs, Lengths=dl_zs )
    plt.title(r"$m_{A'} =$" + str(mAp) + r" GeV, $\epsilon$ = " + str(eps))
    plt.show()

def plot(**kwargs):

    """ Plot Z distributions of decay times and decay lengths """

    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')

    #print([ len(d) for d in kwargs.values() ]) #rem
    for lab, data in kwargs.items():
        plt.hist(
                    data,
                    histtype = 'step',
                    log = True,
                    density = True,
                    range = (0,4000),
                    bins = 50,
                    label = lab
                    )

    plt.legend(loc='upper right')

if __name__=='__main__': main()
