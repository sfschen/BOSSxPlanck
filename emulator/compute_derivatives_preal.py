import numpy as np
import sys
from taylor_approximation import compute_derivatives
from compute_preal_tables import kvec
import json

z = float(sys.argv[1])

# Remake the data grid:
order = 4
Npoints = 2*order + 1
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s)
dxs = [0.01, 0.01, 0.05]

output_shape = (len(kvec),14) # first row is kv

center_ii = (order,)*Nparams
P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Load data
for ii in range(Npoints):
    for jj in range(Npoints):
        for kk in range(Npoints):
            #print(ii,jj,kk)
            P0grid[ii,jj,kk] = np.loadtxt('data/preal/preal_z_%.2f/preal_%d_%d_%d.txt'%(z,ii,jj,kk))
            
            
# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)

# Now save:
outfile = 'emu/preal_z_%.2f.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]

outdict = {'params': ['omegam', 'h', 'sigma8'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs': list0}

json_file = open(outfile, 'w')
json.dump(outdict, json_file)
json_file.close()