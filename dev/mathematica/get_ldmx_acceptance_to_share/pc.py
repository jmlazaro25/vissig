from __future__ import division

import itertools
import xml.etree.ElementTree as ET
import math
import StringIO
import os
import re
import shutil
import logging


class InvalidParamCard(Exception):
    """ a class for invalid param_card """
    pass

class Parameter (object):
    """A class for a param_card parameter"""
    
    def __init__(self, param=None, block=None, lhacode=None, value=None, comment=None):
        """Init the parameter"""

        self.format = 'float'
        if param:
            block = param.lhablock
            lhacode = param.lhacode
            value = param.value
            comment = param.comment
            format = param.format

        self.lhablock = block
        if lhacode:
            self.lhacode = lhacode
        else:
            self.lhacode = []
        self.value = value
        self.comment = comment

    def set_block(self, block):
        """ set the block name """
        
        self.lhablock = block

    def load_str(self, text):
        """ initialize the information from a str"""

        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.comment = text, ""


        data = data.split()
        if any(d.startswith('scan') for d in data):
            position = [i for i,d in enumerate(data) if d.startswith('scan')][0]
            data = data[:position] + [' '.join(data[position:])] 
        if not len(data):
            return
        try:
            self.lhacode = tuple([int(d) for d in data[:-1]])
        except Exception:
            self.lhacode = tuple([int(d) for d in data[:-1] if d.isdigit()])
            self.value= ' '.join(data[len(self.lhacode):])
        else:
            self.value = data[-1]
        
        # convert to number when possible
        try:
            self.value = float(self.value)
        except:
            self.format = 'str'
            pass
        else:
            if self.lhablock == 'modsel':
                self.format = 'int'
                self.value = int(self.value)

    def load_decay(self, text):
        """ initialize the decay information from a str"""

        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.comment = text, ""


        data = data.split()
        if not len(data):
            return
        self.lhacode = [int(d) for d in data[2:]]
        self.lhacode.sort()
        self.lhacode = tuple([len(self.lhacode)] + self.lhacode)
        
        self.value = float(data[0]) 
        self.format = 'decay_table'

    def __str__(self):
        """ return a SLAH string """

        format = self.format
        if self.format == 'float':
            try:
                value = float(self.value)
            except:
                format = 'str'
        
        if format == 'float':
            if self.lhablock == 'decay' and not isinstance(self.value,basestring):
                return 'DECAY %s %e # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
            elif self.lhablock == 'decay':
                return 'DECAY %s Auto # %s' % (' '.join([str(d) for d in self.lhacode]), self.comment)
            elif self.lhablock and self.lhablock.startswith('qnumbers'):
                return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
            else:
                return '      %s %e # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
        elif format == 'int':
            return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
        elif format == 'str':
            if self.lhablock == 'decay':
                return 'DECAY %s %s # %s' % (' '.join([str(d) for d in self.lhacode]),self.value, self.comment)
            return '      %s %s # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
        elif self.format == 'decay_table':
            return '      %e %s # %s' % ( self.value,' '.join([str(d) for d in self.lhacode]), self.comment)
        elif self.format == 'int':
            return '      %s %i # %s' % (' '.join([str(d) for d in self.lhacode]), int(self.value), self.comment)
        else:
            if self.lhablock == 'decay':
                return 'DECAY %s %d # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)
            else:
                return '      %s %d # %s' % (' '.join([str(d) for d in self.lhacode]), self.value, self.comment)


class Block(list):
    """ list of parameter """
    
    def __init__(self, name=None):
        if name:
            self.name = name.lower()
        else:
            self.name = name
        self.scale = None
        self.comment = ''
        self.decay_table = {}
        self.param_dict={}
        list.__init__(self)

    def get(self, lhacode, default=None):
        """return the parameter associate to the lhacode"""
        if not self.param_dict:
            self.create_param_dict()
        
        if isinstance(lhacode, int):
            lhacode = (lhacode,)
            
        try:
            return self.param_dict[tuple(lhacode)]
        except KeyError:
            if default is None:
                raise KeyError, 'id %s is not in %s' % (tuple(lhacode), self.name)
            else:
                return Parameter(block=self, lhacode=lhacode, value=default,
                                                           comment='not define')
        
    def remove(self, lhacode):
        """ remove a parameter """
        list.remove(self, self.get(lhacode))
        # update the dictionary of key
        return self.param_dict.pop(tuple(lhacode))
    
    def __eq__(self, other, prec=1e-4):
        """ """
        if len(self) != len(other):
            return False
        
        return not any(abs(param.value-other.param_dict[key].value)> prec * abs(param.value)
                        for key, param in self.param_dict.items())
        
    def __ne__(self, other, prec=1e-4):
        return not self.__eq__(other, prec)
        
    def append(self, obj):
        
        assert isinstance(obj, Parameter)
        if not hasattr(self, 'name'): #can happen if loeaded from pickle
            self.__init__(obj.lhablock)
        assert not obj.lhablock or obj.lhablock == self.name

        #The following line seems/is stupid but allow to pickle/unpickle this object
        #this is important for madspin (in gridpack mode)
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}
            
        if tuple(obj.lhacode) in self.param_dict:
            if self.param_dict[tuple(obj.lhacode)].value != obj.value:
                raise InvalidParamCard, '%s %s is already define to %s impossible to assign %s' % \
                    (self.name, obj.lhacode, self.param_dict[tuple(obj.lhacode)].value, obj.value)
            return
        list.append(self, obj)
        # update the dictionary of key
        self.param_dict[tuple(obj.lhacode)] = obj

    def create_param_dict(self):
        """create a link between the lhacode and the Parameter"""
        for param in self:
            self.param_dict[tuple(param.lhacode)] = param
        
        return self.param_dict

    def def_scale(self, scale):
        """ """
        self.scale = scale

    def load_str(self, text):
        "set inforamtion from the line"
        
        if '#' in text:
            data, self.comment = text.split('#',1)
        else:
            data, self.commant = text, ""

        data = data.lower()
        data = data.split()
        self.name = data[1] # the first part of data is model
        if len(data) == 3:
            if data[2].startswith('q='):
                #the last part should be of the form Q=
                self.scale = float(data[2][2:])
            elif self.name == 'qnumbers':
                self.name += ' %s' % data[2]
        elif len(data) == 4 and data[2] == 'q=':
            #the last part should be of the form Q=
            self.scale = float(data[3])                
            
        return self
    
    def keys(self):
        """returns the list of id define in this blocks"""
        
        return [p.lhacode for p in self]

    def __str__(self):
        """ return a str in the SLAH format """ 
        
        text = """###################################""" + \
               """\n## INFORMATION FOR %s""" % self.name.upper() +\
               """\n###################################\n"""

        #special case for decay chain
        if self.name == 'decay':
            for param in self:
                pid = param.lhacode[0]
                param.set_block('decay')
                text += str(param)+ '\n'
                if self.decay_table.has_key(pid):
                    text += str(self.decay_table[pid])+'\n'
            return text
        elif self.name.startswith('decay'):
            text = '' # avoid block definition
        #general case 
        elif not self.scale:
            text += 'BLOCK %s # %s\n' % (self.name.upper(), self.comment)
        else:
            text += 'BLOCK %s Q= %e # %s\n' % (self.name.upper(), self.scale, self.comment)
        
        text += '\n'.join([str(param) for param in self])
        return text + '\n'


class ParamCard(dict):
    """ a param Card: list of Block """
    mp_prefix = 'MP__'

    header = \
    """######################################################################\n""" + \
    """## PARAM_CARD AUTOMATICALY GENERATED BY MG5                       ####\n""" + \
    """######################################################################\n"""


    def __init__(self, input_path=None):
        self.order = []
        
        if isinstance(input_path, ParamCard):
            self.read(input_path.write())
            self.input_path = input_path.input_path 
        else:
            self.input_path = input_path
            if input_path:
                self.read(input_path)
        
    def read(self, input_path):
        """ read a card and full this object with the content of the card """

        if isinstance(input_path, str):
            if '\n' in input_path:
                input = StringIO.StringIO(input_path)
            else:
                input = open(input_path)
        else:
            input = input_path #Use for banner loading and test


        cur_block = None
        for line in input:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            line = line.lower()
            if line.startswith('block'):
                cur_block = Block()
                cur_block.load_str(line)
                self.append(cur_block)
                continue
            
            if line.startswith('decay'):
                if not self.has_block('decay'):
                    cur_block = Block('decay')
                    self.append(cur_block)
                else:
                    cur_block = self['decay']
                param = Parameter()
                param.set_block(cur_block.name)
                param.load_str(line[6:])
                cur_block.append(param)
                continue

            if cur_block is None:
                continue            
                    
            if cur_block.name == 'decay':
                # This is a decay table
                id =  cur_block[-1].lhacode[0]
                cur_block = Block('decay_table_%s' % id)
                self['decay'].decay_table[id] = cur_block
            
            

            
            if cur_block.name.startswith('decay_table'):
                param = Parameter()
                param.load_decay(line)
                try:
                    cur_block.append(param)
                except InvalidParamCard:
                    pass
            else:
                param = Parameter()
                param.set_block(cur_block.name)
                param.load_str(line)
                cur_block.append(param)
                  
        return self
    
    def analyze_param_card(self):
        """ Analyzes the comment of the parameter in the param_card and returns
        a dictionary with parameter names in values and the tuple (lhablock, id)
        in value as well as a dictionary for restricted values.
        WARNING: THIS FUNCTION RELIES ON THE FORMATTING OF THE COMMENT IN THE
        CARD TO FETCH THE PARAMETER NAME. This is mostly ok on the *_default.dat
        but typically dangerous on the user-defined card."""
    
        pname2block = {}
        restricted_value = {}

        for bname, block in self.items():
            for lha_id, param in block.param_dict.items():
                all_var = []
                comment = param.comment
                # treat merge parameter
                if comment.strip().startswith('set of param :'):
                    all_var = list(re.findall(r'''[^-]1\*(\w*)\b''', comment))
                # just the variable name as comment
                elif len(comment.split()) == 1:
                    all_var = [comment.strip().lower()]
                # either contraction or not formatted
                else:
                    split = comment.split()
                    if len(split) >2 and split[1] == ':':
                        # NO VAR associated
                        restricted_value[(bname, lha_id)] = ' '.join(split[1:])
                    elif len(split) == 2:
                        if re.search(r'''\[[A-Z]\]eV\^''', split[1]):
                            all_var = [comment.strip().lower()]
                    elif len(split) >=2 and split[1].startswith('('):
                        all_var = [split[0].strip().lower()]
                    else:
                        if not bname.startswith('qnumbers'):
                            logger.debug("not recognize information for %s %s : %s",
                                      bname, lha_id, comment)
                        # not recognized format
                        continue

                for var in all_var:
                    var = var.lower()
                    if var in pname2block:
                        pname2block[var].append((bname, lha_id))
                    else:
                        pname2block[var] = [(bname, lha_id)]
        
        return pname2block, restricted_value
    
    def write(self, outpath=None):
        """schedular for writing a card"""
  
        # order the block in a smart way
        blocks = self.order_block()
        text = self.header
        text += ''.join([str(block) for block in blocks])

        if not outpath:
            return text
        elif isinstance(outpath, str):
            file(outpath,'w').write(text)
        else:
            outpath.write(text) # for test purpose
    
    def create_diff(self, new_card):
        """return a text file allowing to pass from this card to the new one
           via the set command"""
        
        diff = ''
        for blockname, block in self.items():
            for param in block:
                lhacode = param.lhacode
                value = param.value
                new_value = new_card[blockname].get(lhacode).value
                if not misc.equal(value, new_value, 6, zero_limit=False):
                    lhacode = ' '.join([str(i) for i in lhacode])
                    diff += 'set param_card %s %s %s # orig: %s\n' % \
                                       (blockname, lhacode , new_value, value)
        return diff 


 
        
    def append(self, obj):
        """add an object to this"""
        
        assert isinstance(obj, Block)
        self[obj.name] = obj
        if not obj.name.startswith('decay_table'): 
            self.order.append(obj)
        
        
        
    def has_block(self, name):
        return self.has_key(name)
    
    def order_block(self):
        """ reorganize the block """
        return self.order
    
    def rename_blocks(self, name_dict):
        """ rename the blocks """
        
        for old_name, new_name in name_dict.items():
            self[new_name] = self.pop(old_name)
            self[new_name].name = new_name
            for param in self[new_name]:
                param.lhablock = new_name
                
    def remove_block(self, name):
        """ remove a blocks """
        assert len(self[name])==0
        [self.order.pop(i) for i,b in enumerate(self.order) if b.name == name]
        self.pop(name)
        
    def remove_param(self, block, lhacode):
        """ remove a parameter """
        if self.has_param(block, lhacode):
            self[block].remove(lhacode)
            if len(self[block]) == 0:
                self.remove_block(block)
    
    def has_param(self, block, lhacode):
        """check if param exists"""
        
        try:
            self[block].get(lhacode)
        except:
            return False
        else:
            return True
        
    def copy_param(self,old_block, old_lha, block=None, lhacode=None):
        """ make a parameter, a symbolic link on another one """
        
        # Find the current block/parameter
        old_block_obj = self[old_block]
        parameter = old_block_obj.get(old_lha)        
        if not block:
            block = old_block
        if not lhacode:
            lhacode = old_lha
            
        self.add_param(block, lhacode, parameter.value, parameter.comment)
        
    def add_param(self,block, lha, value, comment=''):
        
        parameter = Parameter(block=block, lhacode=lha, value=value, 
                              comment=comment)
        try:
            new_block = self[block]
        except KeyError:
            # If the new block didn't exist yet
            new_block = Block(block)
            self.append(new_block)
        new_block.append(parameter)
        
             
    def mod_param(self, old_block, old_lha, block=None, lhacode=None, 
                                              value=None, comment=None):
        """ change a parameter to a new one. This is not a duplication."""

        # Find the current block/parameter
        old_block = self[old_block]
        try:
            parameter = old_block.get(old_lha)
        except:
            if lhacode is not None:
                lhacode=old_lha
            self.add_param(block, lhacode, value, comment)
            return
        

        # Update the parameter
        if block:
            parameter.lhablock = block
        if lhacode:
            parameter.lhacode = lhacode
        if value:
            parameter.value = value
        if comment:
            parameter.comment = comment

        # Change the block of the parameter
        if block:
            old_block.remove(old_lha)
            if not len(old_block):
                self.remove_block(old_block.name)
            try:
                new_block = self[block]
            except KeyError:
                # If the new block didn't exist yet
                new_block = Block(block)
                self.append(new_block)            
            new_block.append(parameter)
        elif lhacode:
            old_block.param_dict[tuple(lhacode)] = \
                                  old_block.param_dict.pop(tuple(old_lha))


    def check_and_remove(self, block, lhacode, value):
        """ check that the value is coherent and remove it"""
        
        if self.has_param(block, lhacode):
            param = self[block].get(lhacode)
            if param.value != value:
                error_msg = 'This card is not suitable to be convert to SLAH1\n'
                error_msg += 'Parameter %s %s should be %s' % (block, lhacode, value)
                raise InvalidParamCard, error_msg   
            self.remove_param(block, lhacode)




class ParamCardIterator(ParamCard):
    """A class keeping track of the scan: flag in the param_card and 
       having an __iter__() function to scan over all the points of the scan.
    """

    logging = True
    def __init__(self, input_path=None):
        super(ParamCardIterator, self).__init__(input_path=input_path)
        self.itertag = [] #all the current value use
        self.cross = []   # keep track of all the cross-section computed 
        self.param_order = []
        
    def __iter__(self):
        """generate the next param_card (in a abstract way) related to the scan.
           Technically this generates only the generator."""
        
        if hasattr(self, 'iterator'):
            return self.iterator
        self.iterator = self.iterate()
        return self.iterator
    
    def next(self, autostart=False):
        """call the next iteration value"""
        try:
            iterator = self.iterator
        except:
            if autostart:
                iterator = self.__iter__()
            else:
                raise
        try:
            out = iterator.next()
        except StopIteration:
            del self.iterator
            raise
        return out
    
    def iterate(self):
        """create the actual generator"""
        all_iterators = {} # dictionary of key -> block of object to scan [([param, [values]), ...]
        auto = 'Auto'
        pattern = re.compile(r'''scan\s*(?P<id>\d*)\s*:\s*(?P<value>[^#]*)''', re.I)
        # First determine which parameter to change and in which group
        # so far only explicit value of the scan (no lambda function are allowed)
        for block in self.order:
            for param in block:
                if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                    try:
                        key, def_list = pattern.findall(param.value)[0]
                    except:
                        raise Exception, "Fail to handle scanning tag: Please check that the syntax is valid"
                    if key == '': 
                        key = -1 * len(all_iterators)
                    if key not in all_iterators:
                        all_iterators[key] = []
                    try:
                        all_iterators[key].append( (param, eval(def_list)))
                    except SyntaxError:
                        raise Exception, "Fail to handle your scan definition. Please check your syntax."
                    
        keys = all_iterators.keys() # need to fix an order for the scan
        param_card = ParamCard(self)
        #store the type of parameter
        for key in keys:
            for param, values in all_iterators[key]:
                self.param_order.append("%s#%s" % (param.lhablock, '_'.join(`i` for i in param.lhacode)))
        
        # do the loop
        lengths = [range(len(all_iterators[key][0][1])) for key in keys]
        for positions in itertools.product(*lengths):
            self.itertag = []
            if self.logging:
                logger.info("Create the next param_card in the scan definition", '$MG:color:BLACK')
            for i, pos in enumerate(positions):
                key = keys[i]
                for param, values in all_iterators[key]:
                    # assign the value in the card.
                    param_card[param.lhablock].get(param.lhacode).value = values[pos]
                    self.itertag.append(values[pos])
                    if self.logging:
                        logger.info("change parameter %s with code %s to %s", \
                                   param.lhablock, param.lhacode, values[pos])
            # retrun the current param_card up to next iteration
            yield param_card
        
    
    def store_entry(self, run_name, cross):
        """store the value of the cross-section"""
        self.cross.append({'bench' : self.itertag, 'run_name': run_name, 'cross':cross})
        

    def write_summary(self, path):
        """ """
        
        ff = open(path, 'w')
        ff.write("#%-19s %-20s %-20s\n" % ('run_name',' '.join(self.param_order), 'cross(pb)'))
        for info in self.cross:
            bench = [str(p) for p in info['bench']]
            cross = info['cross']
            name = info['run_name']
            ff.write("%-20s %-20s %-20s \n" % (name,' '.join(bench) ,cross))
            #ff.write("%s %s %s \n" % (name,' '.join(bench) ,cross))
            



                         
