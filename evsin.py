import ROOT as r
from glob import glob
from argparse import ArgumentParser
r.gSystem.Load('libFramework.so')

# Load a tree from a group of input files
def load(group,treeName='LDMX_Events'):

    # Load a group of files into a readable tree

    twee = r.TChain(treeName)
    for f in group:
        twee.Add(f)

    return twee

def main():
    
    parser = ArgumentParser()
    parser.add_argument('dirs', nargs='+', action='store', default=[])
    args = parser.parse_args()

    for d in args.dirs:

        files = glob( d + '/*.root')
        n_evs = load(files).GetEntries()
        print('{}: {}'.format(d,n_evs))

if __name__ == '__main__': main()
