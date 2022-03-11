# Load the Various Emulators:
import numpy as np

from emulator_preal import Emulator_Preal as Emulator
from emulator_chils import Chi_LS
from predict_cl import AngularPowerSpectra

# Last Scattering
basedir = '/global/cscratch1/sd/sfschen/BOSSxPlanck/BOSSxPlanck/emulator/'
chils_func= Chi_LS('./' + 'emu/chi_ls.json')

# Galaxy Spectra
Emu = Emulator(basedir+'emu/preal.json',
               basedir+'emu/phalofit.json')

def compute_Ckg_table(cpars, dndz, zeff):
    
    OmegaM, h, lnAs = cpars
    
    aps = AngularPowerSpectra(OmegaM,chils_func([OmegaM,h]),dndz,zeff)
    
    table = np.zeros( (1251,6) )

    # Calculate Reference Point
    b1, b2, bs, alpX = 0, 0, 0, 0
    smag = 0

    pars0 = np.array([b1, b2, bs, alpX, smag])

    bparsA= [0,0,0,0,0]
    bparsX= [b1,b2,bs,alpX]
    
    ell,clgg0,clgk0 = aps(Emu,\
                          cpars,bparsA,bparsX,\
                          smag,Lmax=1251)

    table[:,0] = clgk0

    for ii in range(5):

        pars = np.copy(pars0)
        pars[ii] += 1.0
    
        b1_temp, b2_temp, bs_temp, alpX_temp, smag_temp = pars
    
        bparsA= [0,0,0,0,0]
        bparsX= [b1_temp,b2_temp,bs_temp,alpX_temp]


        ell,clgg,clgk = aps(Emu,\
                            cpars,bparsA,bparsX,\
                            smag_temp,Lmax=1251)
    
        table[:,ii+1] = clgk - clgk0
    
    return table