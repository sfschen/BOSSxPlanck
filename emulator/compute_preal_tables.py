import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# k vector to use:
ki = np.logspace(-3.0,1.0,200)
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )


def compute_preal_tables(pars, z=0.61):
    
    # Commented out: old version using sigma8
    
    #OmegaM, h, sigma8 = pars
    OmegaM, h, lnAs = pars
    
    omega_b = 0.02242

    #lnAs =  3.047
    ns = 0.9665

    nnu = 1
    nur = 2.033
    mnu = 0.06
    omega_nu = 0.0106 * mnu
        
    omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'm_ncdm': mnu,
        'tau_reio': 0.0568,
        'omega_b': omega_b,
        'omega_cdm': omega_c}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    
    # Calculate growth rate
    fnu = pkclass.Omega_nu / pkclass.Omega_m()
    f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)

    # Calculate and renormalize power spectrum
    pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
    #print(pkclass.sigma8())
    #pi = (sigma8/pkclass.sigma8())**2 * pi
    
    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.2, jn=5,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1)
    
    modPT.make_ptable(f, 0, kv=kvec)
    
    return modPT.pktables[0]


def compute_phalofit(pars, z=0.61):
    
    # Commented out: old version using sigma8
    
    #OmegaM, h, sigma8 = pars
    OmegaM, h, lnAs = pars
    
    omega_b = 0.02242

    #lnAs =  3.047
    ns = 0.9665

    nnu = 1
    nur = 2.033
    mnu = 0.06
    omega_nu = 0.0106 * mnu
        
    omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'non linear': 'halofit',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'm_ncdm': mnu,
        'tau_reio': 0.0568,
        'omega_b': omega_b,
        'omega_cdm': omega_c}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()

    # Calculate and renormalize power spectrum
    pnl = np.array( [pkclass.pk(k*h, z ) * h**3 for k in ki] )

    return pnl
    