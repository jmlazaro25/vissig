import os
import sys
from math import floor

def split(lhe, outdir, epf):

    """ Split large LHE files for faster sim """

    # lines per event
    lpe = 17

    # Create outdir if needed and clean in already exists
    if not os.path.exists(outdir): os.makedirs(outdir)
    else: os.system(f'rm -f {outdir}/*')

    # Count number of events in big file
    n_events = 0
    with open(lhe, 'r') as ogfile:
        for line in ogfile:
            if line == '<event>\n': n_events += 1
        ogfile.seek(0) # Go back to the beggining for writing others

    # Give warning if things aren't nice (maybe handle later)
    nfiles = n_events/epf
    if not nfiles.is_integer():
        sys.exit(
                    f'Trying to make {nfiles} files'
                    + f' (n_events = {n_events}, epf = {epf}).'
                    )

    # Open outfiles
    genname = lhe.split('/')[-1].split('.lhe')[0]
    files = [
                open(f'{outdir}/{genname}_p{f_num}.lhe', 'w') \
                for f_num in range(int(nfiles))
                ]
    print( 'Reformatting:\n{}\nInto:\n{}'.format(lhe,[f.name for f in files]) )

    # Loop through og again to write to new files
    with open(lhe, 'r') as ogfile:
    
        # Write header to all files
        for line in ogfile:

            for f in files: f.write(line)

            if line == '</init>\n':
                break
        
        # Split events
        event_line = 0
        for line in ogfile: # Picks up where last loop leaves off

            event_line += 1
            file_num = event_line/(epf*lpe)

            if file_num.is_integer():
                files[ int(file_num) - 1].write(line)
                files[ int(file_num) - 1].write('</LesHouchesEvents>\n')
                files[ int(file_num) - 1].close()
                if file_num == nfiles: break

            else: files[ floor( file_num ) ].write(line)


    return [f.name for f in files]

'''
def main():

    """ For interactive use while batch is prepared """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', action='store', dest='infile')
    parser.add_argument('-o', action='store', dest='outdir')
    parser.add_argument('--epf', action='store', dest='epf', type=int) # events per file
    args = parser.parse_args()

    split(args.infile, args.outdir, args.epf)
'''

def main():

    mAps = (0.005, 0.01, 0.05, 0.1)

    for mAp in mAps:
        split(f'lhes/{mAp}/unweighted_events.lhe', f'{mAp}/split', 10_000)

if __name__ == '__main__': main()
