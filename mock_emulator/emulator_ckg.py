import numpy as np
import json
from taylor_approximation import taylor_approximate


class Emulator_Ckg(object):
    def __init__(self, emufilename):
        self.load(emufilename)
        self.cpars = np.zeros(3)
        self.ctab = {}

        #
    def load(self, emufilename):
        '''Load the Taylor series from emufilename and repackage it into
           a dictionary of arrays.'''
        # Load Taylor series json file from emufilename:
        json_file = open(emufilename,'r')
        emu = json.load( json_file )
        json_file.close()
        
        # repackage into form we need, i.e. a dictionary of arrays
        self.emu_dict = {'ell': np.array(emu['ell']),\
                                'x0': emu['x0'],\
                                'derivs': [np.array(ll) for ll in emu['derivs']]}
        del(emu)


    def update_cosmo(self, cpars):
        '''If the cosmology is not the same as the old one, update the ptables.'''
        if not np.allclose(cpars, self.cpars):
            self.cpars = cpars
            self.ctab = taylor_approximate(cpars,\
                                           self.emu_dict['x0'],\
                                           self.emu_dict['derivs'], order=4)

    def __call__(self, params):
        '''Evaluate the Taylor series for the spectrum given by 'spectra'
           at the point given by 'params'.'''
        #zstr  = str(params[-1])
        cpars = params[:3]
        
        #
        self.update_cosmo(cpars)
        #
        bpars = params[3:-1] # this includes b1, b2, bs, alphaX, smag
        pvec =  np.concatenate( ([1], bpars) )

        ell = self.emu_dict['ell']
        res = np.sum(pvec[None,:] * self.ctab, axis=1)

        return ell,res
