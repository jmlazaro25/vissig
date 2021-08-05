import re
import os
import decimal
import argparse
import numpy as np
from sympy import Symbol
from scipy import interpolate
from sympy.stats import sample, Uniform, Exponential
import phys_form

def main():

    """ Add interactivity for LHE analysis """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', action='store', dest='mAp', type=float)
    parser.add_argument('-i', action='store', dest='infile')
    parser.add_argument('--plot', action='store_true', dest='plot')
    args = parser.parse_args()
    mAp = args.mAp

    # Will store information here
    nEvents = 10_000
    ap_momentum = np.zeros( (nEvents,4) ) # px, py, pz, en (bc of lhes)
    es_momentum = np.zeros( (nEvents,2,4) ) # positron, electron in order
    report = {
            }

    ##################################################
    # Analysis
    ##################################################

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

            if 5 < event_line < 9: # A', positron, or electron momentum
                line = phys_form.rescaleLine(line, tokens=range(6,11))
                px,py,pz,en = [
                                float(v) for v in phys_form.numsInLine(line)[6:10] 
                                ]

                if event_line == 6: # A'
                    ap_momentum[event_num -1] = np.array((px,py,pz,en))

                elif event_line == 7: # positron
                    es_momentum[event_num -1,0] = np.array((px,py,pz,en))

                elif event_line == 8: # electron
                    es_momentum[event_num -1,1] = np.array((px,py,pz,en))

    # Fill report
    report['Average p'] = np.average( ap_momentum, axis=0 )
    report['Average A\' angle wrt z [deg]'] = np.average( [
                                                angle(
                                                        ap_momentum[e,:3],
                                                        units='degrees'
                                                        ) \
                                                for e in range(nEvents)
                                                ] )
    report['Average max(e+,e- angle wrt z)'] = np.average( [ max( [
                                                angle(
                                                        es_momentum[e,p,:3],
                                                        units='degrees'
                                                        ) \
                                                for p in (0,1) ] ) \
                                                for e in range(nEvents)
                                                ] )

    # Print report
    for k,v in report.items():
        print( '{}: {}'.format(k,v) )

    if args.plot:
        pass

# From bdt/mods/physics
import math

def angle(vec, units, vec2=[0,0,1]):

    """ Angle between vectors (with z by default) """

    if units=='degrees': return math.acos( dot( unit(vec), unit(vec2) ) )*180.0/math.pi
    elif units=='radians': return math.acos( dot( unit(vec), unit(vec2) ) )
    else: print('\nSpecify valid angle unit ("degrees" or "randians")')

def mag(iterable):

    """ Magnitude of whatever """

    return math.sqrt(sum([x**2 for x in iterable]))

def unit(arrayy):

    """ Return normalized np array """

    return np.array(arrayy)/mag(arrayy)

def dot(i1, i2):

    """ Dot iterables """

    return sum( [i1[i]*i2[i] for i in range( len(i1) )] )

if __name__=='__main__': main()
