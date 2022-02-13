#!/usr/bin/env python
#
# Plots the power spectra (angular or multipole) and correlation
# functions for the # data and a fiducial model for each hemisphere
# and redshift.
#
import numpy as np
import matplotlib.pyplot as plt

# Set up some labels, colors and linestyles to be
# shared by all of the figures.
zlst = [0.38,0.61]  # Effective redshifts.
clst = ['C0','C1']  # Colors for NGC/SGC
llst = ['--',':' ]




def make_pkl_plot():
    """Does the work of making the P_ell(k) figure."""
    rsddb  = "rsd_data/"
    fig,ax = plt.subplots(2,2,figsize=(8,3.1),sharex=True,\
               gridspec_kw={'height_ratios':[3,1]})
    # The multipole power spectra -- NEED TO FIX ERROR BARS.
    for i,zz in enumerate(zlst):
        for j,hemi in enumerate(['NGC','SGC']):
            offs=0.001 if j==0 else -0.001
            iz = 1 if i==0 else 3
            # The data itself.
            pk = np.loadtxt(rsddb+"pk/pk_{:s}_z{:d}.dat".format(hemi,iz))
            ax[0,i].errorbar(pk[:,0]+offs,pk[:,0]*pk[:,1],yerr=100,\
                             color=clst[j],fmt='s',mfc='None',label=hemi)
            ax[0,i].errorbar(pk[:,0]+offs,pk[:,0]*pk[:,2],yerr=100,\
                             color=clst[j],fmt='^',mfc='None')
            # Plot the theory as lines.
            thy = np.loadtxt(rsddb+"fits/"+
                    "best_fit_pkl_{:s}z{:d}.txt".format(hemi,iz))
            ax[0,i].plot(thy[:,0],thy[:,0]*thy[:,1],'-',color=clst[j])
            ax[0,i].plot(thy[:,0],thy[:,0]*thy[:,2],':',color=clst[j])
            # Now the ratio to the theory.
            ax[1,i].errorbar(pk[:,0]+offs,pk[:,1]/thy[:,1],yerr=100/thy[:,1],\
                             color=clst[j],fmt='s',mfc='None')
            ax[1,i].errorbar(pk[:,0]+offs,pk[:,2]/thy[:,2],yerr=100/thy[:,1],\
                             color=clst[j],fmt='^',mfc='None')
        ax[0,i].legend(title="$z={:.2f}$".format(zz),loc=1)
        #
        ax[1,i].axhline(1.0,ls=':',color='k')
        #
        for j in range(2):
            ax[j,i].set_xlim(0.005,0.20)
        ax[0,i].set_ylim(-100,2250)
        ax[1,i].set_ylim(0,2)
        ax[1,i].set_xlabel(r'$k\quad [h\ {\rm Mpc}^{-1}]$')
    ax[0,0].set_ylabel(r'$k P_\ell(k)\quad [h^2{\rm Mpc}^{-2}]$')
    ax[1,0].set_ylabel(r'Ratio')
    for j in range(2):
        ax[j,1].get_yaxis().set_visible(False)
    #
    plt.tight_layout()
    plt.savefig('plot_data_pkl.pdf')
    #





def make_xil_plot():
    """Does the work of making the xi_ell(s) figure."""
    rsddb  = "rsd_data/"
    fig,ax = plt.subplots(2,2,figsize=(8,3.1),sharex=True,\
               gridspec_kw={'height_ratios':[3,1]})
    # The multipole power spectra.
    for i,zz in enumerate(zlst):
        iz = 1 if i==0 else 3
        # The data itself -- NEED TO FIX ERROR BARS.
        xi = np.loadtxt(rsddb+"xi/z{:d}.xi".format(iz))
        ax[0,i].errorbar(xi[:,0],xi[:,0]**2*xi[:,1],yerr=10,\
                         color=clst[0],fmt='s',mfc='None',label=r'$\ell=0$')
        ax[0,i].errorbar(xi[:,0],xi[:,0]**2*xi[:,2],yerr=10,\
                         color=clst[0],fmt='^',mfc='None',label=r'$\ell=2$')
        ax[0,i].legend(title="$z={:.2f}$".format(zz),loc=1)
        # Plot theory as lines.
        thy = np.loadtxt(rsddb+"fits/best_fit_xil_z{:d}.txt".format(iz))
        ax[0,i].plot(thy[:,0],thy[:,0]**2*thy[:,1],'-',color=clst[0])
        ax[0,i].plot(thy[:,0],thy[:,0]**2*thy[:,2],':',color=clst[0])
        # Now the ratio to the theory.
        ax[1,i].errorbar(xi[:,0],xi[:,1]/thy[:,1],yerr=10/thy[:,1],\
                         color=clst[0],fmt='s',mfc='None')
        ax[1,i].errorbar(xi[:,0],xi[:,2]/thy[:,2],yerr=10/thy[:,2],\
                         color=clst[0],fmt='^',mfc='None')
        ax[1,i].axhline(1.0,ls=':',color='k')
        #
        for j in range(2):
            ax[j,i].set_xlim(80,130)
        ax[0,i].set_ylim(0,75)
        ax[1,i].set_ylim(0,2)
        ax[1,i].set_xlabel(r'$s\quad [h^{-1}{\rm Mpc}]$')
    ax[0,0].set_ylabel(r'$i^\ell s^2 \xi_\ell(s)\quad [h^{-2}{\rm Mpc}^{2}]$')
    ax[1,0].set_ylabel(r'Ratio')
    for j in range(2):
        ax[j,1].get_yaxis().set_visible(False)
    #
    plt.tight_layout()
    plt.savefig('plot_data_xil.pdf')
    #




def make_cls_plot():
    """Does the work of making the C_ell figure."""
    angdb  = "data/"
    fig,ax = plt.subplots(2,2,figsize=(8,3.1),sharex=True,\
               gridspec_kw={'height_ratios':[3,1]})
    # The angular power spectra.
    for i,zz in enumerate(zlst):
      for j,hemi in enumerate(["NGC","SGC"]):
        # Generate the file names
        pref = "gal_s"+str(j+1)
        pref+= "1" if i==0 else "3"
        offs = 3 if hemi=='NGC' else -3
        # and read the data.
        cls = np.loadtxt(angdb+pref+"_cls.txt")
        cov = np.loadtxt(angdb+pref+"_cov.txt")
        wlx = np.loadtxt(angdb+pref+"_wlx.txt")
        best= np.loadtxt(angdb+pref+"_mod.txt")
        # Compute the errors.
        Nbin = cls.shape[0]
        dcla = np.zeros(Nbin)
        dclx = np.zeros(Nbin)
        for k in range(Nbin):
            dcla[k] = np.sqrt( cov[k+0*Nbin,k+0*Nbin] )
            dclx[k] = np.sqrt( cov[k+1*Nbin,k+1*Nbin] )
        # Plot the cross-spectrum data and model.
        ax[0,i].plot(best[:,0],1e6*best[:,2],'-',color=clst[j],label=hemi)
        ax[0,i].errorbar(cls[:,0]+offs,1e6*cls[:,2],yerr=1e6*dclx,\
                         color=clst[j],fmt='s',mfc='None')
        ax[0,i].legend(title="$z={:.2f}$".format(zz),loc=1)
        # Plot the ratio of the data over the theory for the cross-spectrum.
        ells = np.arange(wlx.shape[1])
        obs  = np.dot(wlx,np.interp(ells,best[:,0],best[:,2],right=0))
        ax[1,i].errorbar(cls[:,0]+offs,cls[:,2]/obs,yerr=dclx/obs,\
                         color=clst[j],fmt='s',mfc='None')
        #
        ax[1,i].axhline(1.0,ls=':',color='k')
        #
        for j in [0,1]:
            ax[j,i].set_xlim(1,375)
        ax[0,i].set_yscale('log')
        ax[0,i].set_ylim(0.02,3)
        ax[1,i].set_ylim(0,2)
        ax[1,i].set_xlabel(r'Multipole')
    ax[0,0].set_ylabel(r'$10^6\ C_\ell$')
    ax[1,0].set_ylabel(r'Ratio')
    for j in range(2):
        ax[j,1].get_yaxis().set_visible(False)
    #
    plt.tight_layout()
    plt.savefig('plot_data_cls.pdf')
    #






if __name__=="__main__":
    make_pkl_plot()
    make_xil_plot()
    make_cls_plot()
    #
