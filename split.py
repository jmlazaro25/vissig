import os
import ROOT as r
from glob import glob
r.gSystem.Load('libFramework.so')

def load(fil,treeName='LDMX_Events'):

    # Load ROOT tree

    twee = r.TChain(treeName)
    twee.Add(fil)

    return twee

def main():
   
    # Not triggered (we'll want a different one than for stan)
    gs = ('pn','0.001','0.01','0.1','1.0')
    bkg_train_min = 1_250_000
    sig_train_min = 312_500

    indir = '/nfs/slac/g/ldmx/users/jmlazaro/samples/v3/4gev/vissig/train'
    outdir = '/nfs/slac/g/ldmx/users/jmlazaro/samples/v3/4gev/vissig/test'

    for g in gs:

        files = glob( indir + '/{}/*.root'.format(g))
        n_evs = [ load(f).GetEntries() for f in files ]

        if g == gs[0]: train_min = bkg_train_min
        train_min = sig_train_min

        # Report some stuff
        print('\n\n\n{}: '.format(g))
        print('n_evs: {}'.format(n_evs))
        print('min:',train_min)

        for i in range( len( files ) ):

            if sum( n_evs[:i+1] ) >= train_min:
                ci = i
                break

        # Get number need of events still needed for training
        cut = train_min - sum( n_evs[:ci] )

        # Break up last file if needed
        if cut > 0:

            # Load original
            ogTree = load( files[ci] )

            # Put some in training
            trainF = r.TFile( files[ci][:-5] + '_p1.root', 'RECREATE')
            trainT = ogTree.CloneTree(0)
            for entry in range(cut):
                ogTree.GetEntry(entry)
                trainT.Fill()
            trainF.cd()
            trainT.Write()
            trainF.Close()

            # Put some in testing
            testF = r.TFile( 
                            '{}/{}/'.format(outdir, g) \
                            + files[ci].split('/')[-1][:-5] + '_p2.root',
                            'RECREATE'
                            )
            testT = ogTree.CloneTree(0)
            for entry in range( cut, ogTree.GetEntries() ):
                ogTree.GetEntry(entry)
                testT.Fill()
            testF.cd()
            testT.Write()
            testF.Close()

            # Move original into 'cut' directory to avoid confusion
            cutdir = indir + '/../cut' 
            if not os.path.exists(cutdir): os.makedirs(cutdir)
            print('mv {} {}'.format(files[ci], cutdir))
            os.system( 'mv {} {}'.format(files[ci], cutdir) )
    
        # Move all others into testing
        if not os.path.exists( f'{outdir}/{g}' ): os.makedirs( f'{outdir}/{g}' )
        for f in files[ci+1:]:
            print('mv {} {}/{}'.format(f, outdir, g))
            os.system( 'mv {} {}/{}'.format(f, outdir, g) )

if __name__ == '__main__': main()
