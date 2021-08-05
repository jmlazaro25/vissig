import numpy as np
from subprocess import call
import os
import sys
sys.path.insert(1,'../')

import banner
import pc
import pylhe

import kinematics as kin

import collections

from scipy import interpolate, integrate, optimize

# Extract the axion from banner
def get_mass_from_pc(rd,pid=622, banner_tag='HPS'):
    bfname = os.path.join(rd,str(os.path.basename(rd)+'_' + banner_tag + '_banner.txt'))
        
    if not os.path.isfile(bfname):
        print "# Banner corresponding to ", rd, "not found!"
        
    b = banner.Banner(banner_path=bfname)
    param_card = pc.ParamCard(b['slha'])
    
    return param_card['mass'].get(pid).value

def get_raw_counts(rd):
    evt_file = rd + '/evt.lhe'
    

    with open(evt_file, "w") as outfile:
        call(['gunzip', '-c', rd + '/unweighted_events.lhe.gz'],stdout=outfile)
        
    call(['sed', '-i', 's/&/and/', evt_file])
    
    # Extract the Branching fraction to rho pi from the banner
    bfname = os.path.join(rd,str(os.path.basename(rd)+'_HPS'+'_banner.txt'))
        
    if not os.path.isfile(bfname):
        print "# Banner corresponding to ", rd, "not found!"
        
    b = banner.Banner(banner_path=bfname)
    param_card = pc.ParamCard(b['slha'])
    """
    # mAp decay branching fractions 
    d1 = param_card['decay'].decay_table[666][1].lhacode[1]
    
    if d1 != 624 and d2 != 625:
        print 'Wrong decay mode!'
        exit(0)
        
    br = param_card['decay'].decay_table[622][1].value
    """
    br = 1.
    
    evt_weight_list = []
    
    ax_energy_list = []   
    ax_rap_list = []
    ax_th_list = []
    ax_pz_list = []
    
    Em_recoil_list = []
    
    counter = 0
    
    for evt in pylhe.readLHE(evt_file):
        counter = counter + 1
        
        # Note that we rescale the event weight to get rid of the branching fraction
        evt_weight_list.append((evt['eventinfo']['weight'])/br)
        
        if counter % 10000 == 0:
            print "Processing event ", counter

        for part in evt['particles']:
            #print part
            #exit(0)
            mom = [part['e'], part['px'], part['py'], part['pz']]

            if part['id'] == 622:
                ax_energy_list.append(mom[0])
                ax_rap_list.append(kin.rap(mom))
                ax_th_list.append(kin.theta(mom))
                ax_pz_list.append(mom[3])
                
            if part['id'] == 11 and part['status'] == 1 and part['mother1']!=part['mother2']:
                Em_recoil_list.append(mom[0])
            
    os.remove(evt_file)
         

    evt_weight_list = np.array(evt_weight_list)
        
    ax_energy_list = np.array(ax_energy_list)
   
    ax_rap_list = np.array(ax_rap_list)    
    ax_pz_list = np.array(ax_pz_list)
    ax_th_list = np.array(ax_th_list)
    
    Em_recoil_list = np.array(Em_recoil_list)

    return evt_weight_list, ax_energy_list, ax_rap_list, ax_th_list, ax_pz_list, Em_recoil_list

def get_xsec(evt_weight_list):
    return np.sum(evt_weight_list)/len(evt_weight_list)

def get_xsec_after_cut(Ebeam,xcut,evt_weight_list,Em_recoil_list,ax_pz_list):
    num_evt_total = len(evt_weight_list)

    
    ids = Em_recoil_list/Ebeam < xcut
    
    ax_pz_after_cut = ax_pz_list[ids]
    
    ax_pz_avg_after_cut = np.sum(ax_pz_after_cut)/len(ax_pz_after_cut)
    xsec_after_cut = np.sum(evt_weight_list[ids])/num_evt_total
    
    return xsec_after_cut, ax_pz_avg_after_cut

def get_momentum_list(rd, banner_tag='FT'):
    """
    Returns a list of [pa, pem, pep] for all events
    """
    evt_file = rd + '/evt.lhe'
    

    with open(evt_file, "w") as outfile:
        call(['gunzip', '-c', rd + '/unweighted_events.lhe.gz'],stdout=outfile)
        
    call(['sed', '-i', 's/&/and/', evt_file])
    
    # Extract the Branching fraction to rho pi from the banner
    bfname = os.path.join(rd,str(os.path.basename(rd)+'_' + banner_tag + '_banner.txt'))
        
    if not os.path.isfile(bfname):
        print "# Banner corresponding to ", rd, "not found!"
        
    b = banner.Banner(banner_path=bfname)
    param_card = pc.ParamCard(b['slha'])
    """
    # mAp decay branching fractions 
    d1 = param_card['decay'].decay_table[666][1].lhacode[1]
    
    if d1 != 624 and d2 != 625:
        print 'Wrong decay mode!'
        exit(0)
        
    br = param_card['decay'].decay_table[622][1].value
    """
    br = 1.
  
    counter = 0
    
    four_vector_list = []
    evt_weight_list = []

    for evt in pylhe.readLHE(evt_file):
        counter = counter + 1

        # Note that we rescale the event weight to get rid of the branching fraction
        evt_weight_list.append((evt['eventinfo']['weight'])/br)
        
       
        if counter % 10000 == 0:
            print "Processing event ", counter
       
        
        pp = []
        pr = []
        
        for part in evt['particles']:
            #print part
            #exit(0)
            mom = np.array([part['e'], part['px'], part['py'], part['pz']])

            # Get axion momentum
            if part['id'] == 622:
                pa = mom
            # Get recoil electon momentum
            if part['id'] == 11 and part['status'] == 1 and part['mother1']!=part['mother2']:
                pr = mom 

            """  
            # Get pz/pt for the decay products of the rho
            if part['id'] == -11 and part['status'] == 1:
                pp = mom

            if part['id'] == 11 and part['status'] == 1 and part['mother1']==part['mother2']:
                pm = mom
            """
         # Dummy decay product 4 vectors for now
        four_vector_list.append([pa, pr])
    
    four_vector_list = np.array(four_vector_list)
    evt_weight_list = np.array(evt_weight_list)
       
    return evt_weight_list, four_vector_list
   
