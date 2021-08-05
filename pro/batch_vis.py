import os
import logging
import subprocess
from glob import glob
from time import sleep
from argparse import ArgumentParser
from configparser import ConfigParser

def main():
    
    # Syntax 
    parser = ArgumentParser()
    parser.add_argument('-m', action='store', dest='mAp', type=float)
    parser.add_argument('-e', action='store', dest='eps', type=float)
    parser.add_argument('-n', dest='batch_size', type=int, default=1)
    args = parser.parse_args()

    # Configure logger
    logging.basicConfig(format='[ submitJobs ][ %(levelname)s ]: %(message)s',
            level=logging.DEBUG)

    # Determine how many jobs to submit at a time
    logging.info('Submitting jobs in batches of %s' % args.batch_size)

    # Relevant dirs
    nfsdir = '/nfs/slac/g/ldmx/users/jmlazaro'
    pwd = '/nfs/slac/g/ldmx/users/jmlazaro/samples/v3/vissig/lhes' # bc home isn't home
    lheout = f'{pwd}/{args.mAp}/reformatted'
    rootout = f'{nfsdir}/samples/v3/vissig/train/{args.mAp}'
    if not os.path.exists(lheout): os.makedirs(lheout) 
    if not os.path.exists(rootout):
        print(f'Making {outdir}')
        os.makedirs(rootout) 

    # Command that will be used to submit jobs to the batch system
    batch_command = ('bsub '

                     + '-W 100 '

                     #+ '-q short '
                     #+ '-W 60 '

                     + '-n 3 '
                     + '-R "select[centos7] span[hosts=1]" '
                     + f'singularity run --home {nfsdir} {nfsdir}/ldmx_pro_visdecay.sif {nfsdir} '
                     + f'fire {pwd}/vis.py '
                     + f'-m {args.mAp} '
                     + f'-e {args.eps} '
                     + f'--lheout {lheout} '
                     + f'--rootout {rootout} '
                    )

    # Build list of complete commands
    job_commands = [] # Remnant from more complex batch system but still could be useful for debugging
    for fil in sorted(
                        glob(f'{pwd}/0.01/split/*'),
                        key=lambda x: int( x.split('_p')[1].split('.lhe')[0] )
                        ):

        job_commands.append(batch_command
                            + f'-i {fil} '
                            + f'-r {len(job_commands)}'  
                            )

    job_commands = job_commands[:2]

    # Submit them
    for command in job_commands:

        print(command)
        subprocess.Popen(command, shell=True).wait()

        # If a batch of jobs has been submitted, don't submit another batch
        # until all jobs are running. 
        if (job_commands.index(command) + 1)%args.batch_size == 0:
            
            # Initially, wait 10s before submitting other jobs. This lets the
            # batch system catch up so we get an accurate count of pending
            # jobs.
            sleep(10)

            # Check how many jobs are pending
            cjobs = int(
                    subprocess.check_output('bjobs -p | wc -l', shell=True)
                    )
            print('cjobs: %s' % cjobs)
            while cjobs != 0:
                logging.info('%s jobs are still pending' % cjobs)
                sleep(30)
                cjobs = int(
                        subprocess.check_output('bjobs -p | wc -l', shell=True)
                        )
                continue

if __name__ == '__main__': main()
