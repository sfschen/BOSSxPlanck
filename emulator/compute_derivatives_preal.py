import numpy as np
import sys
from taylor_approximation import compute_derivatives
from compute_preal_tables import kvec
import json

zs = [float(x) for x in sys.argv[1:]]

# Remake the data grid:
order = 4
Npoints = 2*order + 1

# these are OmegaM, h, sigma8
# param_str = ['omegam', 'h', 'sigma8']
#x0s = [0.31, 0.68, 0.73]
#dxs = [0.01, 0.01, 0.05]

# these are OmegaM, h, lnAs
param_str = ['omegam', 'h', 'logA']
x0s = [0.31, 0.68, 2.84]
dxs = [0.01, 0.01, 0.05]

Nparams = len(x0s)
output_shape = (len(kvec),14) # first row is kv

center_ii = (order,)*Nparams
P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Load data per z:

outfile = '/global/cscratch1/sd/sfschen/BOSSxPlanck/emulator/emu/preal.json'
fb = '/global/cscratch1/sd/sfschen/BOSSxPlanck/emulator/data/preal/'

emu_dict = {}

for z in zs:

    # Load Grid
    for ii in range(Npoints):
        for jj in range(Npoints):
            for kk in range(Npoints):
                P0grid[ii,jj,kk] = np.loadtxt(fb+'preal_z_%.2f/preal_%d_%d_%d.txt'%(z,ii,jj,kk))
            
            
    # Now compute the derivatives
    derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)

    list0 = [ dd.tolist() for dd in derivs0 ]

    emu_dict[z] = {'params': param_str,\
                   'x0': x0s,\
                   'kvec': kvec.tolist(),\
                   'derivs': list0}

json_file = open(outfile, 'w')
json.dump(emu_dict, json_file)
json_file.close()