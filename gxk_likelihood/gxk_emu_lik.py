#
# A variant of gxk_likelihood that uses an emulator to
# make all of the theory predictions.
#
import numpy as np
import sys
import os

from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from emulator_ckg      import Emulator_Ckg                 as Emulator
from cobaya.likelihood import Likelihood




class GxKLikelihood(Likelihood):
    # From yaml file.
    model:  str
    basedir: str
    clsfn:  str
    covfn:  str
    Emufn:  list
    zeff:   list
    suffx:  list
    wlafn:  list
    wlxfn:  list
    acut:   list
    xcut:   list
    #
    def initialize(self):
        """Sets up the class."""
        # Load the data and invert the covariance matrix.
        self.loadData()
        self.cinv = np.linalg.inv(self.cov)
        # Load each of the emulators.
        self.Emu = []
        for fn in self.Emufn:
            self.Emu.append(Emulator(self.basedir+fn))
    def get_requirements(self):
        """What we require."""
        reqs = {\
               'logA':     None,\
               'H0':       None,\
               'omegam':   None,\
               'sigma8': None,\
               }
        # Build the parameter names we require for each sample.
        for suf in self.suffx:
            for pref in ['bsig8','b2','bs','bn',\
                         'alpha_a','alpha_x',\
                         'SN','smag']:
                reqs[pref+'_'+suf] = None
        return(reqs)
    def logp(self,**params_values):
        """Return the log-likelihood."""
        pp  = self.provider
        OmM = pp.get_param('omegam')
        hub = pp.get_param('H0')/100.0
        logA= pp.get_param('logA')
        sigma8=pp.get_param('sigma8')
        # We want to store some of this information in "self" for
        # easy retrieval later.
        self.thy = {}
        # The windowed theory vectors will all be concatenated to
        # form a large vector, called "obs" here.
        obs = np.array([],dtype='float')
        for i,suf in enumerate(self.suffx):
            # Fill in the parameter list, starting with the
            # cosmological parameters.
            if self.model.startswith('clpt'):
                cpars = [OmM,hub,logA]
            else:
                raise RuntimeError("Unknown model.")
            # Extract some common parameters.
            zeff= float(self.zeff[i])
            b1  = pp.get_param('bsig8_'+suf)/sigma8 - 1.
            b2  = pp.get_param('b2_'+suf)
            bs  = pp.get_param('bs_'+suf)
            sn  = pp.get_param('SN_'+suf)
            smag= pp.get_param('smag_'+suf)
            #
            # Do some parameter munging depending upon the model name
            # to fill in the rest of pars.
            if self.model.startswith('clpt'):
                alpX  = pp.get_param('alpha_x_'+suf)
                bparsX= [b1,b2,bs,alpX]
            else:
                raise RuntimeError("Unknown model.")
            # and call APS to get a prediction,
            ell,clgk = self.Emu[i](cpars+bparsX+[smag,zeff])
            clgg     = np.zeros_like(clgk)
            thy = np.array([ell,clgg,clgk]).T
            self.thy[suf]=thy.copy()
            # then "observe" it, appending the observations to obs.
            obs = np.append(obs,self.observe(thy,self.wla[i],self.wlx[i]))
        self.obs = obs.copy()
        # Now compute chi^2 and return -ln(L)
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
        return(-0.5*chi2)
        #
    def loadData(self):
        """Load the data, covariance and windows from files."""
        dd        = np.loadtxt(self.basedir+self.clsfn)
        self.cov  = np.loadtxt(self.basedir+self.covfn)
        self.wla = []
        for fn in self.wlafn:
            self.wla.append(np.loadtxt(self.basedir+fn))
        self.wlx = []
        for fn in self.wlxfn:
            self.wlx.append(np.loadtxt(self.basedir+fn))
        # Now pack things and modify the covariance matrix to
        # "drop" some data points.
        Nsamp   = (dd.shape[1]-1)//2
        if Nsamp!=len(self.wla):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.wlx):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.acut):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xcut):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        self.xx = dd[:,0]
        self.dd = dd[:,1:].T.flatten()
        self.input_cov = self.cov.copy()
        for j in range(Nsamp):
            for i in np.nonzero(self.xx>self.acut[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx>self.xcut[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
        #
    def observe(self,tt,wla,wlx):
        """Applies the window function and binning matrices."""
        lmax = wla.shape[1]
        ells = np.arange(lmax)
        # Have to stack auto and cross.
        obs1 = np.dot(wla,np.interp(ells,tt[:,0],tt[:,1],right=0))
        obs2 = np.dot(wlx,np.interp(ells,tt[:,0],tt[:,2],right=0))
        obs  = np.concatenate([obs1,obs2])
        return(obs)
        #
