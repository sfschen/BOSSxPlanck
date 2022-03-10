#!/usr/bin/env python3
#
# Makes the plot of the P_{cb} and P_m power spectra for
# massive neutrino cosmologies, using CLASS.
#
import numpy as np
import matplotlib.pyplot as plt


from classy import Class

# Cosmological parameters (global):
OmegaM, h, sigma8 = 0.3, 0.67, 0.75


def make_class():
    """Generates a CLASS instance."""
    omega_b = 0.02242
    lnAs    =  3.047
    ns      = 0.9665
    #
    nnu = 1
    nur = 2.033
    mnu = 0.06
    omega_nu = 0.0106 * mnu
    omega_c  = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
    #
    pkparams = {\
        'output': 'mPk',\
        'P_k_max_h/Mpc': 20.,\
        'z_pk': '0.0,10',\
        'A_s': np.exp(lnAs)*1e-10,\
        'n_s': ns,\
        'h': h,\
        'N_ur': nur,\
        'N_ncdm': nnu,\
        'm_ncdm': mnu,\
        'tau_reio': 0.0568,\
        'omega_b': omega_b,\
        'omega_cdm': omega_c}
    #
    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    #
    return(pkclass)
    #




def make_plot():
    """Does the work of making the figure."""
    cc = make_class()
    #
    fnu      = cc.Omega_nu / cc.Omega_m()
    Omega_cb = OmegaM * (1-fnu)
    chi_z1   = cc.angular_distance(0.38) * (1.+0.38) * h
    chi_z3   = cc.angular_distance(0.59) * (1.+0.59) * h
    print("Comoving distance to z=0.38 is {:.1f}Mpc/h.".format(chi_z1))
    print("Comoving distance to z=0.59 is {:.1f}Mpc/h.".format(chi_z3))
    #
    # Make linear power spectra
    ki    = np.logspace(-3.0,1.0,200)
    p_cb1 = (sigma8/cc.sigma8())**2 * np.array( [cc.pk_cb( k*h, 0.38 ) * h**3 for k in ki] )
    p_mm1 = (sigma8/cc.sigma8())**2 * np.array( [cc.pk_lin(k*h, 0.38 ) * h**3 for k in ki] )
    p_cbm1= np.sqrt(p_cb1 * p_mm1)
    #
    p_cb3 = (sigma8/cc.sigma8())**2 * np.array( [cc.pk_cb( k*h, 0.59 ) * h**3 for k in ki] )
    p_mm3 = (sigma8/cc.sigma8())**2 * np.array( [cc.pk_lin(k*h, 0.59 ) * h**3 for k in ki] )
    p_cbm3= np.sqrt(p_cb3 * p_mm3)
    #
    fig,ax = plt.subplots(1,2,figsize=(8,3.25))
    #
    ax[0].semilogx(ki, p_mm1/ (p_cb1), 'C1',label=r'$\bf{z1}$')
    ax[0].semilogx(ki, p_cbm1/ (p_cb1),'C1--')
    ax[0].semilogx(ki, p_mm3/ (p_cb3), 'C2',label=r'$\bf{z3}$')
    ax[0].semilogx(ki, p_cbm3/ (p_cb3),'C2--')
    #
    ax[0].fill_between([0.02, 0.2], [0,0],[2,2], color='grey', alpha=0.2)
    ax[0].fill_between([50/chi_z1, 250/chi_z1], [0,0],[2,2], color='C1', alpha=0.5)
    ax[0].fill_between([50/chi_z3, 350/chi_z3], [0,0],[2,2], color='C2', alpha=0.3)
    ax[0].text(4e-3, 0.991, r'$M_\nu = 0.06$ eV')
    ax[0].legend()
    #
    ax[0].set_xlim(3e-3, 0.5)
    ax[0].set_ylim(0.990, 1.00)
    ax[0].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$')
    ax[0].set_ylabel(r'$P_{XY}/ P_{cb}$')
    #
    ax[1].semilogx(ki, OmegaM**2 * p_mm1/ (Omega_cb**2 * p_cb1), 'C1')
    ax[1].semilogx(ki, OmegaM * Omega_cb * p_cbm1/ (Omega_cb**2 * p_cb1),'C1--')
    ax[1].semilogx(ki, OmegaM**2 * p_mm3/ (Omega_cb**2 * p_cb3), 'C2')
    ax[1].semilogx(ki, OmegaM * Omega_cb * p_cbm3/ (Omega_cb**2 * p_cb3),'C2--')
    ax[1].plot(0,0, 'k', label='m,m')
    ax[1].plot(0,0, 'k--', label='cb,m')
    ax[1].fill_between([0.02, 0.2], [0,0],[2,2], color='grey', alpha=0.2)
    ax[1].fill_between([50/chi_z1, 250/chi_z1], [0,0],[2,2], color='C1', alpha=0.5)
    ax[1].fill_between([50/chi_z3, 350/chi_z3], [0,0],[2,2], color='C2', alpha=0.3)
    ax[1].legend()
    #
    ax[1].set_xlim(3e-3, 0.5)
    ax[1].set_ylim(0.999, 1.01)
    ax[1].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$')
    ax[1].set_ylabel(r'$\Omega_X \Omega_Y\ P_{XY}/ \Omega_{cb}^2 P_{cb}$')
    #
    plt.tight_layout()
    plt.savefig('neutrino_scales.pdf')
    #



if __name__=="__main__":
    make_plot()
    #
