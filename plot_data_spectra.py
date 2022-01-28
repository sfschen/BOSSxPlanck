#!/usr/bin/env python
#
# Plots the power spectra (angular or multipole) for the
# data and a fiducial model for each hemisphere and redshift.
#
import numpy as np
import matplotlib.pyplot as plt



def make_plot():
    """Does the work of making the figure."""
    db     = "data/"
    fig,ax = plt.subplots(4,2,figsize=(10,8),\
               gridspec_kw={'height_ratios':[3,1,3,1]})
    # Set up some labels, colors and linestyles.
    zlst = [0.38,0.61]  # Effective redshifts.
    clst = ['C0','C1']  # Colors for NGC/SGC
    llst = ['--',':' ]
    # First plot the multipole power spectra.
    for i,zz in enumerate(zlst):
        ax[0,i].plot([0,0.25],[1,100],color=clst[0],label='NGC')
        ax[0,i].plot([0,0.25],[100,1],color=clst[1],label='SGC')
        ax[0,i].legend(title="$z={:.2f}$".format(zz),loc=1)
        #
        ax[1,i].axhline(1.0,ls=':',color='k')
        #
        ax[0,i].set_xlim(0,0.25)
        ax[2,i].set_ylim(1,4)
        ax[2,i].set_yscale('log')
        ax[0,i].get_xaxis().set_visible(False)
        ax[1,i].set_xlim(0,0.25)
        ax[1,i].set_ylim(0,2)
        ax[1,i].set_xlabel(r'$k\quad [h\ {\rm Mpc}]$')
    # then the angular power spectra.
    for i,zz in enumerate(zlst):
      for j,hemi in enumerate(["NGC","SGC"]):
        # Generate the file names
        pref = "gal_s"+str(j+1)
        pref+= "1" if i==0 else "3"
        # and read the data.
        cls = np.loadtxt(db+pref+"_cls.txt")
        cov = np.loadtxt(db+pref+"_cov.txt")
        wlx = np.loadtxt(db+pref+"_wlx.txt")
        best= np.loadtxt(db+pref+"_mod.txt")
        # Compute the errors.
        Nbin = cls.shape[0]
        dcla = np.zeros(Nbin)
        dclx = np.zeros(Nbin)
        for k in range(Nbin):
            dcla[k] = np.sqrt( cov[k+0*Nbin,k+0*Nbin] )
            dclx[k] = np.sqrt( cov[k+1*Nbin,k+1*Nbin] )
        # Plot the data.
        ax[2,i].plot(best[:,0],1e6*best[:,1],'-',color=clst[j],label=hemi)
        ax[2,i].plot(best[:,0],1e6*best[:,2],':',color=clst[j])
        ax[2,i].errorbar(cls[:,0],1e6*cls[:,1],yerr=1e6*dcla,\
                         color=clst[j],fmt='s')
        ax[2,i].errorbar(cls[:,0],1e6*cls[:,2],yerr=1e6*dclx,\
                         color=clst[j],fmt='^')
        ax[2,i].legend(title="$z={:.2f}$".format(zz),loc=1)
        # Plot the ratio of the data over the theory for the cross-spectrum.
        ells = np.arange(wlx.shape[1])
        obs  = np.dot(wlx,np.interp(ells,best[:,0],best[:,2],right=0))
        ax[3,i].errorbar(cls[:,0],cls[:,2]/obs,yerr=dclx/obs,\
                         color=clst[j],fmt='^')
        #
        ax[3,i].axhline(1.0,ls=':',color='k')
        #
        ax[2,i].set_xlim(0,399)
        ax[2,i].set_ylim(0.01,250)
        ax[2,i].set_yscale('log')
        ax[2,i].set_ylabel(r'$10^6\ C_\ell$')
        ax[2,i].get_xaxis().set_visible(False)
        ax[3,i].set_xlim(0,399)
        ax[3,i].set_ylim(0,2)
        ax[3,i].set_xlabel(r'Multipole')
        ax[3,i].set_ylabel(r'Ratio')
    #
    plt.tight_layout()
    plt.savefig('plot_data_spectra.pdf')
    #



if __name__=="__main__":
    make_plot()
