import re
import os
import decimal
import argparse
import numpy as np
from sympy import Symbol
from scipy import interpolate
from sympy.stats import sample, Uniform, Exponential
from reformat import phys_form

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
    decay_vs = '{}/{}_decay.dat'.format(outdir,genname)
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
    t = Symbol('t')
    decay_width = phys_form.gamma_ap_tot(mAp, eps)
    tau = phys_form.tau(mAp, eps)
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
            open(decayfile, 'w') as decayf, \
            open(decay_vs, 'w') as decayvs:
    
        ##################################################
        # Edit header (techincaly until </init>
        # Many conditions shouldn't check in events sec.
        ##################################################
        scaling_mass = False
        for line in ogfile:

            # ebeams
            if re.search(r'ebeam',line):
                line = phys_form.rescaleLine(line)

            # Masses
            if line[:10] == 'BLOCK MASS':
                scaling_mass = True # Indicate following lines should be scaled
                continue
            if line[0] == '#':
                scaling_mass = False
            if scaling_mass:
                line = phys_form.rescaleLine(line, tokens=[1])

            # Decay Width
            if re.match(r'DECAY +622', line):
                line = phys_form.replaceNums(line, [1], [decay_width])

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

                line = phys_form.rescaleLine(line, tokens=range(6,11))
            
            # Event info line
            if event_line ==1:
            
                # Correct particle number
                event_brem_lines.append( phys_form.replaceNums(line, [0], [5]) )
                event_decay_lines.append(phys_form.replaceNums(line, [0], [2]) )

            elif event_line < 7: # If first 5 write to bremfile
                event_brem_lines.append(line)

                if event_line == 6: # Take note of Ap info for projection
                    px,py,pz = [
                                float(v) for v in phys_form.numsInLine(line)[6:9] 
                                ]
                    Ap_3mom = np.array((px,py,pz))

            elif event_line < 9: # decay electrons

                # Null parents
                event_decay_lines.append( phys_form.replaceNums(line, [2,3], [-1,-1]) )

            # Skip mgrwt. add appropriate vertex, and end event
            elif event_line == 16 :

                # Prepare vertex samples
                #x,y,z,t = next(Xs), next(Ys), next(Zs), next(decay_t)*(en/mAp)
                x,y,z,t = next(Xs), next(Ys), 0, next(decay_t)
                c_vertex = np.array( (x,y,z) )
                d_vertex = c_vertex + Ap_3mom*phys_form.c_speed / mAp * t

                # If not in allowed z, don't write event
                if not ( zmin <= d_vertex[2] <= zmax ): continue
                nevents_used += 1 # Else, count event as used

                # If it is allowed, catch up the writing
                for ln in event_brem_lines: bremf.write(ln)
                for ln in event_decay_lines: decayf.write(ln)

                # Then add the verticies
                bremf.write( '#vertex {} {} {}\n'.format(x,y,z) )
                decayf.write( '#vertex {} {} {} {}\n'.format(*d_vertex,t) )
                decayvs.write( '{} {} {} {}\n'.format(*d_vertex,t) )

                # And end event
                bremf.write(line)
                decayf.write(line)
        
            # End both
            elif line == '</LesHouchesEvents>\n':
                bremf.write(line)
                decayf.write(line)

    print(f'Using {nevents_used} events')

    return bremfile, decayfile, nevents_used

if __name__=='__main__': main()
