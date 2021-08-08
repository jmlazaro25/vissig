from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-r', dest='run', type=int)
parser.add_argument('-m', dest='mAp', type=float)
parser.add_argument('-e', dest='eps', type=float)
parser.add_argument('-z', dest='zlims', type=int, nargs=2)
parser.add_argument('-i', dest='infile')
parser.add_argument('--lheout', dest='lheout')
parser.add_argument('--rootout', dest='rootout')
args = parser.parse_args()

# Set up process
from LDMX.Framework import ldmxcfg

proc = 'v12' # Consider arg
outname = (
            f'mAp{args.mAp}'
            + f'_run{args.run}'
            + f'_eps{args.eps}'
            + f'_z{args.zlims[0]}-{args.zlims[1]}'
            + f'_lp{proc}'
            )
p = ldmxcfg.Process(f'{proc}')
p.outputFiles = [ f'{args.rootout}/{outname}.root' ]
p.maxEvents = 10_000
p.maxEvents = 100
p.logFrequency = 1
p.termLogLevel = 0
p.run = args.run # Handles random seeds

# Set up simulation
from LDMX.SimCore import simulator
from LDMX.Ecal import EcalGeometry
from LDMX.Hcal import HcalGeometry
import LDMX.Ecal.ecal_hardcoded_conditions as ecal_conditions
import LDMX.Hcal.hcal_hardcoded_conditions as hcal_conditions
sim = simulator.simulator('visible_signal')
sim.description = "A' -> ee visible signal decay"
sim.setDetector('ldmx-det-v12',True)
#sim.beamSpotSmear = [20., 80., 0] # Built into reformat.write

# Generate lhe files on the fly from the input brem+decay file
import reformat.reformat as reformat
dark_brem_file, ap_decay_file, n_allowed_events = reformat.writeBremDecay(
                                                        args.infile,
                                                        args.mAp,
                                                        args.eps,
                                                        args.zlims,
                                                        args.run, # Random seed
                                                        args.lheout,
                                                        outname
                                                        )

# Reduce number of events to produce if needed
p.maxEvents = min( p.maxEvents, n_allowed_events)

# Generators 
from LDMX.SimCore import generators
dark_brem = generators.lhe('dark_brem', dark_brem_file)
ap_decay  = generators.lhe('ap_decay' , ap_decay_file)
sim.generators = [ dark_brem, ap_decay ]

# Producers
import LDMX.Ecal.digi as ecal_digi
import LDMX.Ecal.vetos as ecal_vetos
import LDMX.Hcal.hcal as hcal_py
import LDMX.Hcal.digi as hcal_digi
from LDMX.Recon.simpleTrigger import simpleTrigger

p.sequence=[
        sim,
        ecal_digi.EcalDigiProducer(),
        ecal_digi.EcalRecProducer(),
        ecal_vetos.EcalVetoProcessor(),
        hcal_digi.HcalDigiProducer(),
        hcal_digi.HcalRecProducer(),
        hcal_py.HcalVetoProcessor(),
        ]
print(p)
