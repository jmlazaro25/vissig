import re
import os
import decimal
import argparse
import numpy as np
from sympy import Symbol
from scipy import interpolate
from sympy.stats import sample, Uniform, Exponential

def main():

    """ Add interactivity for LHE analysis """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', action='store', dest='mAp', type=float)
    parser.add_argument('-e', action='store', dest='eps', type=float)
    parser.add_argument('-l', action='store', dest='gctau', type=float)
    parser.add_argument('-i', action='store', dest='infile',)
    args = parser.parse_args()
    mAp, eps, gctau = args.mAp, args.eps, args.gctau

    # Will store information here
    nEvents = 10_000
    decay_zs = [] # Only those within bounds on first sample
    #positions = np.zeros( (nEvents, 3) )
    report = {
            }

    ##################################################
    # Analysis (similar to writeBremDecay, but dev-y as opposed to pro-y)
    ##################################################

    # Creation XYZ
    Sym = Symbol('q')
    x_rv = Uniform(Sym, -10 , 10 )
    y_rv = Uniform(Sym, -40 , 40 )
    #z_rv = Uniform(Sym, -0.175, 0.175)
    Xs = sample( x_rv, numsamples=nEvents, seed=np.random.seed( 2 ) )
    Ys = sample( y_rv, numsamples=nEvents, seed=np.random.seed( 2 ) )
    #Zs = sample( z_rv, numsamples=nEvents, seed=np.random.seed( 2 ) )

    # Detector limits
    zmin = 300
    zmax = 4000 - 300

    # Decay time
    c_speed = 299_792_458_000 # mm/s
    t = Symbol('t')
    decay_width = gamma_ap_tot(mAp, eps)
    tau = 6.582e-25 / decay_width # hbar = 6.582e-25 GeV*s
    decay_rv = Exponential(t, 1/tau)
    decay_t = sample(
                        decay_rv,
                        numsamples=nEvents*rsample_factor,
                        seed=np.random.seed( 2 )
                        )

    # Decay 

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
                line = rescaleLine(line, tokens=range(6,11))
                px,py,pz,en = [
                                float(v) for v in numsInLine(line)[6:10] 
                                ]
                Ap_3mom = np.array((px,py,pz))

            """ Might care about these later
            elif event_line < 9: # decay electrons

                # Null parents
                line = replaceNums(line, [2,3], [-1,-1])
            """

            # Skip mgrwt. add appropriate vertex, and end event
            if event_line == 16 :

                # Add verticies
                #x,y,z,t = next(Xs), next(Ys), next(Zs), next(decay_t)*(en/mAp)
                x,y,z = next(Xs), next(Ys), 0
                c_vertex = np.array( (x,y,z) )

                t = -1 # While prep
                d_vertex = c_vertex + Ap_3mom*c_speed / mAp * t
                resamps = -1
                while not ( zmin <= d_vertex[2] <= zmax ):
                    resamps += 1
                    """
                    if resamps == 50:
                        print(f'50 samples on event {event_num}: {Ap_3mom}')
                    """
                    if resamps == resamples_allowed:
                        #print(f'500 samples on event {event_num}; p = {Ap_3mom}')
                        break
                    try:
                        t = next(decay_t)*(en/mAp)
                        d_vertex = c_vertex + Ap_3mom*c_speed / mAp * t
                    except: # ran out of decay_t
                        # Fill report
                        report['Resampled events (normalized)'] \
                                    = np.count_nonzero(resamples) / event_num
                        report['Average resamples'] \
                                    = np.average(resamples)*nEvents / event_num
                        report['Max resamples'] = max(resamples)
                        report['Events before quitting'] = event_num

                        # Print report
                        print(resamples[:event_num])
                        for k,v in report.items():
                            print( '{}: {}'.format(k,v) )

                        quit()

                if resamps == 0:
                    decay_zs.append(d_vertex[2])
                elif resamps < resamples_allowed:
                    decay_zs_resampled.append(d_vertex[2])

                resamples[event_num-1] = resamps

    # Fill report
    report['Resampled events (normalized)'] \
                                        = np.count_nonzero(resamples) / nEvents
    report['Average resamples'] = np.average(resamples)
    report['Max resamples'] = max(resamples)

    # Print report
    for k,v in report.items():
        print( '{}: {}'.format(k,v) )

def plot():

    """ Plot resamples """

    """
    import matplotlib.pyplot as plt
    plt.hist(resamples)
    plt.title(r"$m_{A'} =$" + str(mAp) + r" GeV, $\epsilon$ = " + str(eps))

    # Plot decay_zs, decay_zs, and combo
    #plt.figure()
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ns, bins, p  = ax1.hist(
                [decay_zs,decay_zs_resampled],
                histtype='step',
                log=True,
                bins=50,
                range = (zmin,zmax),
                density=True,
                label=['decay_zs','decay_zs_resampled']
                )

    plt.bar( bins[:-1 ] , ns[1] / ns[0], label='nolog', width=68)
    plt.bar( bins[:-1 ] , np.log(ns[1]/ns[0]) / np.log(10),
            label='log', width=68 )
    
    plt.legend(loc='upper center')
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')

    plt.hist(
                decay_zs_resampled,
                histtype='step',
                log=True,
                bins=50,
                range = (zmin,zmax),
                density=True,
                label='decay_zs_resampled'
                )

    plt.legend(loc='upper center')
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')

    plt.hist(
                decay_zs + decay_zs_resampled,
                histtype='step',
                log=True,
                bins=50,
                range = (zmin,zmax),
                density=True,
                label='Combination'
                )

    plt.legend(loc='upper right')
    plt.ylabel(r'A.U.')
    plt.xlabel(r'Decay Z [mm]')

    plt.title(r"$m_{A'} =$" + str(mAp) + r" GeV, $\epsilon$ = " + str(eps))

    plt.show()
    """

    pass

if __name__=='__main__': main()
