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
    gctau = gamma * phys_form.c_speed * phys_form.tau(mAp, eps)
    #print(gctau) # For when you only want to know gctau
    #if True: return

    # Detector limits
    zmin = 300
    zmax = 4000 - 300

    # Will store information here
    nEvents = 10_000
    dl_zs = []
    report = {
            'gctau [mm]': gctau
            }

    ##################################################
    # Analysis
    ##################################################

    # Decay length
    z = Symbol('z')
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
                px,py,pz = [
                            float(v) for v in phys_form.numsInLine(line)[6:9] 
                            ]
                Ap_3mom = np.array((px,py,pz))

                # Decay length
                l = next(decay_l)
                dl_zs.append( phys_form.unit(Ap_3mom)[2] * l )

    # Fill report

    # Print report
    for k,v in report.items():
        print( '{}: {}'.format(k,v) )

    plot( dl_zs, mAp, eps, gamma*phys_form.c_speed)
    plt.show()

def plot(zs, mass, ep, gc):

    """ Plot Z distributions of decay times and decay lengths """

    # Individual
    plt.figure(1)
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')
    plt.title(
                r"$m_{A'} =$" + str( mass ) + ' GeV, '
                + r'$\epsilon$ = ' + str( ep )
                )

    plt.hist(
                np.clip(zs,0,4000),
                histtype = 'step',
                log = True,
                density = True,
                range = (0,4000),
                bins = 50,
                )

    plt.figure(0)
    ns, bins = plt.hist(
                        np.clip(zs,0,4000),
                        histtype = 'step',
                        range = (0,4000),
                        bins = 50,
                        )[:2]
    expos = np.exp( -1*bins[:-1] / (gc * phys_form.tau(mass,ep) )) \
                                    / (gc * phys_form.tau(mass,ep) )

    plt.figure(2)
    epsilons = np.logspace(-4, -5, num=10)
    for e in epsilons:
        gct = gc * phys_form.tau(mass, e)
        plt.step(
                bins[:-1],
                ns * ( np.exp( -1*bins[:-1] / gct) / gct) / expos,
                label = r'$\epsilon = $' + f'{e}'
                )

    plt.legend(loc='upper center')
    plt.ylim( 0, plt.ylim()[1] )

if __name__=='__main__': main()
