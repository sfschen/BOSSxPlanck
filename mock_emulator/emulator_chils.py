import numpy as np
import json
from taylor_approximation import taylor_approximate


class Chi_LS(object):
    
    def __init__(self, emufilename):
        self.load(emufilename)
        
    def load(self, emufilename):
        # Load Taylor series json file from emufilename:
        json_file = open(emufilename,'r')
        emu = json.load( json_file )
        json_file.close()
        
        # repackage into form we need, i.e. a dictionary of arrays
        self.emu_dict = {'x0': emu['x0'],\
                         'derivs': [np.array(ll) for ll in emu['derivs']]}
        
        del(emu)
    
    def __call__(self, cpars):

        return float(taylor_approximate(cpars,\
                                  self.emu_dict['x0'],\
                                  self.emu_dict['derivs'], order=3))
