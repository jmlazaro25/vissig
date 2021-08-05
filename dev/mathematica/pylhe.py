import os

# Author: Lukas Heinrich
# Downloaded from https://pypi.python.org/pypi/pylhe/0.0.2

class LHEFile(object):
  def __init__(self):
    pass

class LHEEventInfo(object):
  fieldnames = ['nparticles', 'pid', 'weight', 'scale', 'aqed', 'aqcd']
  
  @classmethod
  def fromstring(cls,string):
    return dict(zip(cls.fieldnames,map(float,string.split())))
    

class LHEParticle(dict):
  fieldnames = ['id','status','mother1','mother2','color1','color2','px','py','pz','e','m','lifetime','spin']
  
  def __init__(self):
    pass

  @classmethod
  def fromstring(cls,string):
    return dict(zip(cls.fieldnames,map(float,string.split())))

def loads():
  pass
  
import xml.etree.ElementTree as ET
def readLHE(file):
  try:
    for event,element in ET.iterparse(file,events=['end']):      
      if element.tag == 'event':
          eventdict = {}
          data = element.text.split('\n')[1:-1]
          eventdata,particles = data[0],data[1:]
          eventdict['eventinfo']=LHEEventInfo.fromstring(eventdata)
          eventdict['particles'] = []
          for p in particles:
            eventdict['particles']+=[LHEParticle.fromstring(p)]
          yield eventdict

  except ET.ParseError:
    print("WARNING. Parse Error.")
    return
"""    
import networkx as nx
import pypdt
import tex2pix
import subprocess
import shutil
def visualize(event,outputname):
  g = nx.DiGraph()
  for i,p in enumerate(event['particles']):
      g.add_node(i+1,attr_dict=p)
      name = pypdt.particle(p['id']).name
      greek = ['gamma','nu','mu','tau','rho','Xi','Sigma','Lambda','omega','Omega','Alpha','psi','phi','pi']
      for greekname in greek:
        if greekname in name:
          name = name.replace(greekname,'\\'+greekname)
      if 'susy-' in name:
        name = name.replace('susy-','\\tilde ')
      g.node[i+1].update(texlbl = "${}$".format(name))
  for i,p in enumerate(event['particles']):
     if(p['mother1']>0):g.add_edge(int(p['mother1']),i+1)
     if(p['mother2']>0):g.add_edge(int(p['mother2']),i+1)
  nx.write_dot(g,'event.dot')
  p = subprocess.Popen(['dot2tex','event.dot'], stdout = subprocess.PIPE)
  tex2pix.Renderer(texfile = p.stdout).mkpdf(outputname)
  subprocess.check_call(['pdfcrop',outputname,outputname])
  # shutil.move('event-cropped.pdf','event.pdf')
  os.remove('event.dot')
"""
