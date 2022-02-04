#!/usr/bin/env python
#

import numpy as np
import sys
import json

from taylor_approximation import compute_derivatives

# Load the sample properites
db= sys.argv[1]
zeff = float(sys.argv[2])
dndz = np.loadtxt(sys.argv[3])
suffix = sys.argv[4]

# Remake the data grid:
order = 4
Npoints = 2*order + 1

# These are settings for OmegaM, h, lnAs
param_str = ['omegam', 'h', 'logA']
x0s = [0.31, 0.68, 2.84]
dxs = [0.01, 0.01, 0.05]

Nparams = len(x0s)

center_ii = (order,)*Nparams
output_shape = (1251,6)
P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Set up the paths and create an empty dictionary.
fb      = db + '/data/ckg/'
emu_dict= {}

for ii in range(Npoints):
    for jj in range(Npoints):
        for kk in range(Npoints):
            print(ii,jj,kk)
            fn = "ckg_" + suffix
            fn+= "/ckg_{:s}_{:d}_{:d}_{:d}.txt".format(suffix,ii,jj,kk)
            P0grid[ii,jj,kk] = np.loadtxt(fb+fn)
    
# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)
list0   = [ dd.tolist() for dd in derivs0 ]
# and add them to the dictionary.
emu_dict = {'params': param_str,\
            'x0': x0s,\
            'ell': np.arange(1251).tolist(),\
            'derivs': list0}

# Write the results to file.
outfile   = db + 'emu/ckg_%s.json' %(suffix)
json_file = open(outfile, 'w')
json.dump(emu_dict, json_file)
json_file.close()
