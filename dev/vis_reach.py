import numpy as np
import sys
import array
tmpargv = sys.argv
sys.argv = []
import getopt
import ROOT
from ROOT import gROOT, TFile, TTree, TChain, gDirectory, TLine, gStyle, TCanvas, TLegend, TH1, TH1F, TH2F, TF1, TLatex
sys.argv = tmpargv

def openPDF(outfile,canvas):
	canvas.Print(outfile+".pdf[")

def closePDF(outfile,canvas):
	canvas.Print(outfile+".pdf]")

def N_ap(m, eps, eot):
    return 7. * pow(eps/1.e-5, 2) * pow(0.1/m, 2) * eot / 1.e16

def N_sig(Naprime, zmin, zmax, gctau):
    return Naprime * (np.exp(-zmin / gctau) - np.exp(-zmax / gctau))

def GammaCTau(E, m, eps):
    return 65. * (E/8.) * pow(1.e-5 / eps, 2) * pow(0.1/m, 2)

label = ""

options, remainder = getopt.gnu_getopt(sys.argv[1:], 'hl:')

# Parse the command line arguments
for opt, arg in options:
		if opt=='-l':
			label = arg
		if opt=='-h':
			print_usage()
			sys.exit(0)

gStyle.SetOptStat(0)
c = TCanvas("c","c",800,600)

outfile = remainder[0]

ebeam = 8. #GeV
zmin = 43. #cm
zmax = 315. #cm
#zmin = 43 #cm
#zmax = 84 #cm
eot = 1.e16
minSignal = 14

NepsBins = 100
epsmin = -7
epsmax = -2

nMass = 100
#massmin = 10
#massmax = 1000
massmin = -2
massmax = 0

Medges = array.array('d')
Epsedges = array.array('d')
for i in range(0,nMass+1):
    #Medges.append(massmin/1.e3+(i-0.5)*(massmax/1.e3-massmin/1.e3)/float(nMass-1))
    Medges.append(10**(massmin+(i-0.5)*(massmax-massmin)/float(nMass-1)))
for j in range(0,NepsBins+1):
	Epsedges.append(10**(epsmin+(j-0.5)*(epsmax-epsmin)/float(NepsBins-1)))

NAprime = TH2F("NAprime", "NAprime", nMass, Medges, NepsBins, Epsedges)
GamCTau = TH2F("GamCTau", "GamCTau", nMass, Medges, NepsBins, Epsedges)
detectable = TH2F("detectable", "detectable", nMass, Medges, NepsBins, Epsedges)

for i in range(0, nMass):
    #mass = (massmax - massmin)/float(nMass - 1) * i + massmin
    logmass = (massmax - massmin)/float(nMass - 1) * i + massmin
    mass = 10**logmass
    #massArr.append(mass)
    for j in range(0, NepsBins):
        logeps = (epsmax - epsmin)/float(NepsBins - 1) * j + epsmin
        eps = 10**logeps
        #epsarr.append(eps)
        Naprime = N_ap(mass, eps, eot)
        gctau = GammaCTau(ebeam, mass, eps)
        nsig = N_sig(Naprime, zmin, zmax, gctau)
        #print(nsig)
        NAprime.Fill(mass, eps, Naprime)
        GamCTau.Fill(mass, eps, gctau)
        detectable.Fill(mass, eps, nsig)

openPDF(outfile,c)
c.SetLogx(1)
c.SetLogy(1)
c.SetLogz(1)
NAprime.Draw("COLZ")
NAprime.SetTitle("Number of A's Produced, {0:.0f} GeV Beam, {1} EOT {2}".format(ebeam, eot, label))
NAprime.GetXaxis().SetTitle("mass [GeV]  ")
NAprime.GetYaxis().SetTitle("#epsilon")
c.Print(outfile+".pdf")
GamCTau.Draw("COLZ")
GamCTau.SetTitle("Gamma CTau, {0:.0f} GeV Beam {1}".format(ebeam, label))
GamCTau.GetXaxis().SetTitle("mass [GeV]  ")
GamCTau.GetYaxis().SetTitle("#epsilon")
c.Print(outfile+".pdf")
detectable.Draw("COLZ")
detectable.SetTitle("Number of Signal Events, {0:.0f} GeV Beam, {1} EOT {2}".format(ebeam, eot, label))
detectable.GetXaxis().SetTitle("mass [GeV]  ")
detectable.GetYaxis().SetTitle("#epsilon")
c.Print(outfile+".pdf")

nlevels = 1
contour = array.array('d')
contour.append(minSignal)
detectable.SetContour(1, contour)
detectable.Draw("cont2")
detectable.SetTitle("Contour for {0} Signal Events, {1:.0f} GeV Beam, {2} EOT {3}".format(minSignal, ebeam, eot, label))
detectable.GetXaxis().SetTitle("mass [GeV]  ")
detectable.GetYaxis().SetTitle("#epsilon")
c.Print(outfile+".pdf")

closePDF(outfile,c)
