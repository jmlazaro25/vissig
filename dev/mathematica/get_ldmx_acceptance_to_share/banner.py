from __future__ import division
import copy
import logging
import numbers
import os
import sys
import re
import math

#dict
class Banner(dict):
    """ """

    ordered_items = ['mgversion', 'mg5proccard', 'mgproccard', 'mgruncard',
                     'slha', 'mggenerationinfo', 'mgpythiacard', 'mgpgscard',
                     'mgdelphescard', 'mgdelphestrigger','mgshowercard','run_settings']

    capitalized_items = {
            'mgversion': 'MGVersion',
            'mg5proccard': 'MG5ProcCard',
            'mgproccard': 'MGProcCard',
            'mgruncard': 'MGRunCard',
            'mggenerationinfo': 'MGGenerationInfo',
            'mgpythiacard': 'MGPythiaCard',
            'mgpgscard': 'MGPGSCard',
            'mgdelphescard': 'MGDelphesCard',
            'mgdelphestrigger': 'MGDelphesTrigger',
            'mgshowercard': 'MGShowerCard' }
    
    def __init__(self, banner_path=None):
        """ """
        if isinstance(banner_path, Banner):
            dict.__init__(self, banner_path)
            self.lhe_version = banner_path.lhe_version
            return     
        else:
            dict.__init__(self)
        
        if banner_path:
            self.read_banner(banner_path)

    ############################################################################
    #  READ BANNER
    ############################################################################
    pat_begin=re.compile('<(?P<name>\w*)>')
    pat_end=re.compile('</(?P<name>\w*)>')

    tag_to_file={'slha':'param_card.dat',
      'mgruncard':'run_card.dat',
      'mgpythiacard':'pythia_card.dat',
      'mgpgscard' : 'pgs_card.dat',
      'mgdelphescard':'delphes_card.dat',      
      'mgdelphestrigger':'delphes_trigger.dat',
      'mg5proccard':'proc_card_mg5.dat',
      'mgproccard': 'proc_card.dat',
      'init': '',
      'mggenerationinfo':'',
      'scalesfunctionalform':'',
      'montecarlomasses':'',
      'initrwgt':'',
      'madspin':'madspin_card.dat',
      'mgshowercard':'shower_card.dat',
      'run_settings':''
      }
    
    def read_banner(self, input_path):
        """read a banner"""

        if isinstance(input_path, str):
            if input_path.find('\n') ==-1:
                input_path = open(input_path)
            else:
                def split_iter(string):
                    return (x.groups(0)[0] for x in re.finditer(r"([^\n]*\n)", string, re.DOTALL))
                input_path = split_iter(input_path)
                
        text = ''
        store = False
        for line in input_path:
            if self.pat_begin.search(line):
                if self.pat_begin.search(line).group('name').lower() in self.tag_to_file:
                    tag = self.pat_begin.search(line).group('name').lower()
                    store = True
                    continue
            if store and self.pat_end.search(line):
                if tag == self.pat_end.search(line).group('name').lower():
                    self[tag] = text
                    text = ''
                    store = False
            if store:
                if line.endswith('\n'):
                    text += line
                else:
                    text += '%s%s' % (line, '\n')
                
            #reaching end of the banner in a event file avoid to read full file 
            if "</init>" in line:
                break
            elif "<event>" in line:
                break
    
    def __getattribute__(self, attr):
        """allow auto-build for the run_card/param_card/... """
        try:
            return super(Banner, self).__getattribute__(attr)
        except:
            if attr not in ['run_card', 'param_card', 'slha', 'mgruncard', 'mg5proccard', 'mgshowercard', 'foanalyse']:
                raise
            return self.charge_card(attr)

