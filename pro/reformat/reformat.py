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
    parser.add_argument('-i', action='store', dest='infile',)
    args = parser.parse_args()
    mAp, eps = args.mAp, args.eps
    rsample_factor = 10

    # Will store information here
    nEvents = 10_000
    resamples_allowed = 500
    resamples = np.zeros( nEvents )
    decay_zs = [] # Only those within bounds on first sample
    decay_zs_resampled = [] # Within bounds after < resamples_allowed
    positions = np.zeros( (nEvents, 3) )
    report = {
            'Resampled events (normalized)': 0,
            'Average resamples': 0,
            'Max resamples': 0
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

    # Plot resamples
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


def writeBremDecay(lhe, mAp, eps, seed, outdir=None, nevents=10_000):

    """ Break A'->ee LHEs into brem and decay files and reformat/rescale """

    # Write to input directory if outdir not provided
    if outdir is None: outdir = '/'.join( lhe.split('/')[:-1] )

    # Create outdir if needed
    if not os.path.exists(outdir): os.makedirs(outdir)

    # Outfile names
    genname = lhe.split('/')[-1].split('.lhe')[0] \
            + '_run{}'.format(seed) \
            + '_eps{}'.format(eps)
    bremfile = '{}/{}_brem.lhe'.format(outdir,genname)
    decayfile = '{}/{}_decay.lhe'.format(outdir,genname)
    print( 'Reformatting:\n{}\nInto:\n{}\n{}'.format(lhe,bremfile,decayfile) )

    # Creation XYZ
    Sym = Symbol('q')
    x_rv = Uniform(Sym, -10 , 10 )
    y_rv = Uniform(Sym, -40 , 40 )
    #z_rv = Uniform(Sym, -0.175, 0.175)
    Xs = sample( x_rv, numsamples=nevents, seed=np.random.seed( seed ) )
    Ys = sample( y_rv, numsamples=nevents, seed=np.random.seed( seed ) )
    #Zs = sample( z_rv, numsamples=nevents, seed=np.random.seed( seed ) )

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
                        numsamples=nevents,
                        seed=np.random.seed( seed )
                        )

    # Will store information here
    nevents_used = 0

    # Open original and output files
    with open(lhe, 'r') as ogfile, \
            open(bremfile, 'w') as bremf,  \
            open(decayfile, 'w') as decayf:
    
        ##################################################
        # Edit header (techincaly until </init>
        # Many conditions shouldn't check in events sec.
        ##################################################
        scaling_mass = False
        for line in ogfile:

            # ebeams
            if re.search(r'ebeam',line):
                line = rescaleLine(line)

            # Masses
            if line[:10] == 'BLOCK MASS':
                scaling_mass = True # Indicate following lines should be scaled
                continue
            if line[0] == '#':
                scaling_mass = False
            if scaling_mass:
                line = rescaleLine(line, tokens=[1])

            # Decay Width
            if re.match(r'DECAY +622', line):
                line = replaceNums(line, [1], [decay_width])

            # Break from header/init
            if line == '</init>\n':
                bremf.write(line)
                decayf.write(line)
                break
        
            bremf.write(line)
            decayf.write(line)
        
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
                event_brem_lines = []
                event_decay_lines = ['<event>\n']

                if event_num % 1000 == 0:
                    print( 'Reformatting event: {}'.format(event_num) )

            else: event_line += 1
            if 1 < event_line < 9:

                line = rescaleLine(line, tokens=range(6,11))
            
            # Event info line
            if event_line ==1:
            
                # Correct particle number
                event_brem_lines.append( replaceNums(line, [0], [5]) )
                event_decay_lines.append( replaceNums(line, [0], [2]) )

            elif event_line < 7: # If first 5 write to bremfile
                event_brem_lines.append(line)

                if event_line == 6: # Take note of Ap info for projection
                    px,py,pz,en = [
                                    float(v) for v in numsInLine(line)[6:10] 
                                    ]
                    Ap_3mom = np.array((px,py,pz))

            elif event_line < 9: # decay electrons

                # Null parents
                event_decay_lines.append( replaceNums(line, [2,3], [-1,-1]) )

            # Skip mgrwt. add appropriate vertex, and end event
            elif event_line == 16 :

                # Prepare vertex samples
                #x,y,z,t = next(Xs), next(Ys), next(Zs), next(decay_t)*(en/mAp)
                x,y,z,t = next(Xs), next(Ys), 0, next(decay_t)*(en/mAp)
                c_vertex = np.array( (x,y,z) )
                d_vertex = c_vertex + Ap_3mom*c_speed / mAp * t

                # If not in allowed z, don't write event
                if not ( zmin <= d_vertex[2] <= zmax ): continue
                nevents_used += 1 # Else, count event as used

                # If it is allowed, catch up the writing
                for ln in event_brem_lines: bremf.write(ln)
                for ln in event_decay_lines: decayf.write(ln)

                # Then add the verticies
                bremf.write( '#vertex {} {} {}\n'.format(x,y,z) )
                decayf.write( '#vertex {} {} {} {}\n'.format(*d_vertex,t) )

                # And end event
                bremf.write(line)
                decayf.write(line)
        
            # End both
            elif line == '</LesHouchesEvents>\n':
                bremf.write(line)
                decayf.write(line)

    print(f'Using {nevents_used} events')

    return bremfile, decayfile, nevents_used

# <From Nikita Blinov> ##################################################
# Hadronic R ratio used to compute hadronic width of the A'
Rvals=np.loadtxt("reformat/r_fixed.dat")
#Rvals=np.loadtxt("r_fixed.dat")
Rvals_interp = interpolate.interp1d(Rvals[:,0],Rvals[:,1],kind='linear');

def Rfunc(s):
    if np.sqrt(s) >= 0.36:
        return Rvals_interp(np.sqrt(s))
    else:
        return 0.

def gamma_ap_to_ll(mAp,ml,eps):
    if mAp < 2.*ml:
        return 0.
    
    aEM=1/137.035999679

    beta=1. - (2*ml/mAp)**2
    
    return (1./3.)*(aEM*eps**2)*mAp*np.sqrt(beta)*(1 + (2*ml**2)/mAp**2)
    
# Total decay rate of Ap into electrons and muons and hadrons
# Valid for mAp > 2*me 
# Width is returned in GeV
def gamma_ap_tot(mAp, epsilon):
    me = 0.51099895/1000.
    mmu = 105.6583745/1000.
    return gamma_ap_to_ll(mAp,me,epsilon) \
            + gamma_ap_to_ll(mAp,mmu,epsilon)*(1. + Rfunc(mAp**2))

# </From Nikita Blinov> ##################################################

##################################################
# Line editing 
##################################################

def numsInLine(line_w_nums):

    """ Find numbers in line """

    nums =  re.findall(r' [\d,\.,e,\-,\+]+', line_w_nums) # Seems close enough

    return [ num[1:] for num in nums ]

def rescaleLine(line_w_nums, scale=decimal.Decimal('0.1'), tokens=[0]):

    """ Replace numbers at given tokens (indicies) with scaled versions """

    numbers = numsInLine(line_w_nums)
    numbers = [ numbers[i] for i in tokens ] # Get numbers at desired indicies
    scaled_line = line_w_nums

    for number in numbers: 
        scaled_line = re.sub(re.sub(r'\+', r'\\+', number), # looks like - equiv not needed
                            str(decimal.Decimal(number)*scale), scaled_line,
                            count=1)

    return scaled_line

def replaceNums(line_w_nums, tokens, new_vals):

    """ Replace numbers at given tokens (indicies) with specific new_vals """

    numbers = numsInLine(line_w_nums)
    numbers = [ numbers[i] for i in tokens ]# Numbers we care about
    new_line = line_w_nums

    for number, new_val in zip(numbers,new_vals):
        new_line = re.sub(number, str(new_val), new_line, count=1)

    return new_line

if __name__=='__main__': main()
