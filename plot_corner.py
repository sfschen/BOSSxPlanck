#
# Makes the corner plot(s), should be run in the Cobaya environment.
#
import numpy as np
import matplotlib.pyplot as plt

from cobaya.yaml          import yaml_load_file
from cobaya.samplers.mcmc import plot_progress
from cobaya.model         import get_model
#
from getdist.mcsamples    import MCSamplesFromCobaya
from getdist.mcsamples    import loadMCSamples
import getdist.plots      as     gdplt



db   = "/global/cscratch1/sd/mwhite/BOSS/Cobaya/"


#
chains = []
clist  = []
llist  = []
legnd  = [r'RSD+BAO',r'+$\kappa g$',r'+$\kappa g$ (SZ)',r'+$\kappa g\ (\ell>80)$']
icol   = 0
for yaml in [\
            "rsd_bao_zall.yaml",\
            "rxk_ngc_lcdm.yaml",\
            "rxk_nosz_ngc_lcdm.yaml",\
            "rxk_ngc_lcdm_lcut1.yaml",\
            ]:
    info= yaml_load_file(db + yaml)
    cc  = loadMCSamples(db + info["output"],no_cache=True,\
                        settings={'ignore_rows':0.3})
    p   = cc.getParams()
    cc.addDerived(p.sigma8*(p.omegam/0.3)**0.5,name='S8',label='S_8')
    chains.append(cc)
    #
    col = 'C'+str(icol)
    clist.append(col)
    llist.append({'ls':'-','color':col})
    legnd.append(yaml[:-5])
    icol += 1
#
for cc in chains:
    print('\n'+cc.getName())
    print("R-1=",cc.getGelmanRubin())
    for k in ["omegam","H0","sigma8","S8"]:
        print( cc.getInlineLatex(k) )
    if cc.getName().find('slip')>=0:
        print( cc.getInlineLatex('gamma') )




# Do the corner plot.
g = gdplt.get_subplot_plotter()
g.triangle_plot(chains,\
                ["omegam","H0","sigma8"],\
                colors=clist,line_args=llist,\
                legend_labels=legnd,\
                filled=[True,True,False,False],\
                alphas=[0.40,0.40,1.000,1.000])
g.export('corner_plot.pdf')
#
