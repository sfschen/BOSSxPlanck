#
# A variant of gxk_likelihood that uses an emulator to
# make all of the theory predictions.
#
import numpy as np
import sys
import os
import yaml

from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from emulator_ckg      import Emulator_Ckg                 as Emulator
from cobaya.likelihood import Likelihood




class GxKLikelihood(Likelihood):
    # From yaml file.
    model:  str
    basedir: str
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    clsfn:  str
    covfn:  str
    Emufn:  list
    zeff:   list
    suffx:  list
    wlafn:  list
    wlxfn:  list
    amin:   list
    xmin:   list
    amax:   list
    xmax:   list
    #
    
    def initialize(self):
        """Sets up the class."""
        # Load the data and invert the covariance matrix.
        
        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.basedir+self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        
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
               'gamma': None,
               }
        # Build the parameter names we require for each sample.
        for suf in self.suffx:
            for pref in ['bsig8','b2','bs']:
                reqs[pref+'_'+suf] = None
        return(reqs)
    
    def full_predict(self, thetas=None):
        '''
        Combine observe and predict.
        '''
        obs = np.array([],dtype='float')
        
        for i,suf in enumerate(self.suffx):
            # Compute theory prediction
            thy = self.clgk_predict(i, suf,self.zeff[i],thetas=thetas)
            # then "observe" it, appending the observations to obs.
            obs = np.append(obs,self.observe(thy,self.wla[i],self.wlx[i]))

        return obs
    
    def logp(self,**params_values):
        """Return the log-likelihood."""
        
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        if not self.optimize:
            lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL
    
    def get_best_fit(self):
        '''
        Generate best fits including linear templates.
        '''

        self.thy_nl  = self.dd - self.Delta
        self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
        self.thy_lin = np.einsum('i,il', self.bf_thetas, self.templates)
        self.thy = self.thy_nl + self.thy_lin
        return self.thy

    
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
        if Nsamp!=len(self.amin):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xmin):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.amax):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xmax):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        self.xx = dd[:,0]
        self.dd = dd[:,1:].T.flatten()
        self.input_cov = self.cov.copy()
        for j in range(Nsamp):
            for i in np.nonzero(self.xx>self.amax[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx>self.xmax[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx<self.amin[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx<self.xmin[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
        #
        
    def clgk_predict(self, i, suf, zeff, thetas=None):
        '''
        Predict Clgk for sample 'suf'.
        '''
        
        pp  = self.provider
        OmM = pp.get_param('omegam')
        hub = pp.get_param('H0')/100.0
        logA= pp.get_param('logA')
        sigma8 = pp.get_param('sigma8')
        ck = (1 + pp.get_param('gamma'))/2.
        
        cpars = [OmM,hub,logA]
        
        # Extract some common parameters.
        b1  = pp.get_param('bsig8_'+suf)/sigma8 - 1.
        b2  = pp.get_param('b2_'+suf)
        bs  = pp.get_param('bs_'+suf)
        
        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alpX = self.linear_param_means['alpha_x_' + suf]
            # sn = self.linear_param_means['SN_' + suf] # we don't actually use this
            smag = self.linear_param_means['smag_' + suf]
        else:
            alpX = thetas['alpha_x_' + suf]
            # sn = thetas['SN_' + suf] # we don't actually use this
            smag = thetas['smag_' + suf]
            
        # alpX  = pp.get_param('alpha_x_'+suf)
        # sn  = pp.get_param('SN_'+suf)
        bparsX= [b1,b2,bs,alpX]
        
        # Magnification also gets modified by gravitational slip
        # rescale to smag' such that c(5s - 2) = 5s' - 2
        smag = ck * smag - 2*(ck - 1)
        #
        # Do some parameter munging depending upon the model name
        # to fill in the rest of pars.

        # and call APS to get a prediction,
        ell,clgk = self.Emu[i](cpars+bparsX+[smag,zeff])
        clgg     = np.zeros_like(clgk)
            
        #Correct for slip:
        clgk *= ck
            
        thy = np.array([ell,clgg,clgk]).T
        
        return thy
    
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
