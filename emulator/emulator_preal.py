import numpy as np
import json
from taylor_approximation import taylor_approximate


class Emulator_Preal(object):
    
    def __init__(self, filebase):

        self.load(filebase)
        
        self.cpars = np.zeros(3)
        self.ptabs  = {}
        
    
    def load(self, filebase):
        
        # Load Taylor Series json file
        json_file = open('emu/preal.json', 'r')
        emu = json.load( json_file )
        json_file.close()
        
        # repackage into form we need, i.e. arrays
        
        self.emu_dict = {}
        
        for zstr in emu.keys():
            
            self.emu_dict[zstr] = {'kvec': np.array(emu[zstr]['kvec']),\
                                   'x0': emu[zstr]['x0'],\
                                   'derivs': [np.array(ll) for ll in emu[zstr]['derivs']]}
        
        del emu
        
    
    def update_cosmo(self, cpars):
        '''
        If the cosmology is not the same as the old one, update the ptables.
        '''
        if not np.allclose(cpars, self.cpars):
            self.cpars = cpars
            for zstr in self.emu_dict.keys():
                self.ptabs[zstr] = taylor_approximate(cpars,\
                                                      self.emu_dict[zstr]['x0'],\
                                                      self.emu_dict[zstr]['derivs'], order=4)
                
        
    
    def __call__(self, params, spectra='Pgg'):
        
        zstr = str(params[-1])
        cpars = params[:3]
        bpars = params[3:-1]
        
        self.update_cosmo(cpars)
        
        if spectra == 'Pgg':
            b1, b2, bs,b3, alpha, sn,= bpars
            bias_monomials = np.array([1, b1, b1**2,\
                                       b2, b1*b2, b2**2,\
                                       bs, b1*bs, b2*bs, bs**2,\
                                       b3, b1*b3])
        elif spectra == 'Pgm':
            b1, b2, bs, b3, alpha = bpars
            sn = 0
            bias_monomials = np.array([1, 0.5*b1, 0,\
                               0.5*b2, 0, 0,\
                               0.5*bs, 0, 0, 0,\
                               0.5*b3, 0])
        
        elif spectra =='Pmm':
            alpha = bpars
            sn = 0
            bias_monomials = np.array([1, 0, 0,\
                               0, 0, 0,\
                               0, 0, 0, 0,\
                               0, 0])
            
        kvec = self.ptabs[zstr][:,0]
        za   = self.ptabs[zstr][:,-1]
    
        # the first row is kv, last row is za for countrterm
        res = np.sum(self.ptabs[zstr][:,1:-1] * bias_monomials,axis=1)\
              + alpha * kvec**2 * za + sn
    
        return kvec, res