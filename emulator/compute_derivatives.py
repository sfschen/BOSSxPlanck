import numpy as np
import sys
from taylor_approximation import compute_derivatives
from compute_pell_tables import compute_pell_tables, kvec
import json

z = float(sys.argv[1])

# Remake the data grid:
order = 4
Npoints = 2*order + 1
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s)
dxs = [0.01, 0.01, 0.05]

output_shape = (len(kvec),19)

center_ii = (order,)*Nparams
P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Load data
for ii in range(Npoints):
    for jj in range(Npoints):
        for kk in range(Npoints):
            #print(ii,jj,kk)
            P0grid[ii,jj,kk] = np.loadtxt('data/boss_z_%.2f/boss_p0_%d_%d_%d.txt'%(z,ii,jj,kk))
            P2grid[ii,jj,kk] = np.loadtxt('data/boss_z_%.2f/boss_p2_%d_%d_%d.txt'%(z,ii,jj,kk))
            P4grid[ii,jj,kk] = np.loadtxt('data/boss_z_%.2f/boss_p4_%d_%d_%d.txt'%(z,ii,jj,kk))
            
# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)
derivs2 = compute_derivatives(P2grid, dxs, center_ii, 5)
derivs4 = compute_derivatives(P4grid, dxs, center_ii, 5)

# Now save:
outfile = 'emu/boss_z_%.2f_pkells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['omegam', 'h', 'sigma8'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

json_file = open(outfile, 'w')
json.dump(outdict, json_file)
json_file.close()