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

    # Detector limits
    zmin = 300
    zmax = 4000 - 300

    # Will store information here
    nEvents = 10_000
    dt_zs = []
    dl_zs = []
    report = {}

    ##################################################
    # Analysis
    ##################################################

    # Decay time
    t = Symbol('t')
    tau = phys_form.tau(mAp, eps)
    decay_t_rv = Exponential(t, 1/tau)
    decay_t = sample(
                        decay_t_rv,
                        numsamples=nEvents,
                        seed=np.random.seed( 2 )
                        )

    # Decay length
    z = Symbol('z')
    gctau = gamma * phys_form.c_speed * tau
    #print(gctau) # For when you only want to know gcctau
    #if True: return
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
                t = next(decay_t)
                dt_zs.append( Ap_3mom[2]*phys_form.c_speed / mAp * t )

                # Decay length
                l = next(decay_l)
                dl_zs.append( phys_form.unit(Ap_3mom)[2] * l )

    # Fill report
    report['gamma * c * tau [mm]'] = gctau

    # Print report
    for k,v in report.items():
        print( '{}: {}'.format(k,v) )

    plot( mAp=mAp, eps=eps, Times=dt_zs, Lengths=dl_zs )
    plt.show()

def plot(**kwargs):

    """ Plot Z distributions of decay times and decay lengths """

    # Individual
    plt.figure(1)
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')
    plt.title(
                r"$m_{A'} =$" + str( kwargs['mAp'] ) + ' GeV, '
                + r'$\epsilon$ = ' + str( kwargs['eps'] )
                )

    ns = []
    for lab, data in kwargs.items():
        if type(data) == float: continue
        plt.figure(0)
        plt.hist(
                            np.clip(data,0,4000),
                            histtype = 'step',
                            log = True,
                            density = True,
                            range = (0,4000),
                            bins = 50,
                            label = lab
                            )[:2]
        plt.figure(1)
        n, bins = plt.hist(
                            np.clip(data,0,4000),
                            histtype = 'step',
                            range = (0,4000),
                            bins = 50,
                            label = lab
                            )[:2]
        ns.append(n)

    if True:
        print('Events per bin:')
        for x,y in zip( ns[0], ns[1] ):
            print(x,y)

    plt.legend(loc='upper right')

    # Ratio
    plt.figure(2)
    plt.ylabel(r'$n_{Lengths}/n_{Times}$')
    plt.xlabel(r'Decay Z [mm]')
    plt.title(
                r"$m_{A'} =$" + str( kwargs['mAp'] ) + ' GeV, '
                + r'$\epsilon$ = ' + str( kwargs['eps'] )
                )


    plt.step(
            bins[:-1],
            ns[1]/ns[0],
            )

    # Plot ratio errorbars
    #plt.scatter()

    plt.ylim( 0, plt.ylim()[1] )
    

if __name__=='__main__': main()
