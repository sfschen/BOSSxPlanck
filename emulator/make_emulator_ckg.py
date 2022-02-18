# Make the grid:
import numpy as np
import sys
import os
import json

from mpi4py import MPI
from compute_ckg_table import compute_Ckg_table
from taylor_approximation import compute_derivatives

# Load the sample properites
db= sys.argv[1]
zeff = float(sys.argv[2])
dndz = np.loadtxt(sys.argv[3])
suffix = sys.argv[4]

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
comm.Barrier()
#print("Hello I am process {:d} of {:d}.".format(mpi_rank, mpi_size))

# these are settings for OmegaM, h, lnAs
param_str = ['omegam', 'h', 'logA']
x0s = [0.31, 0.68, 2.84]
dxs = [0.01, 0.01, 0.05]

Nparams = len(x0s)
order = 4

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')
output_shape = (1251,6)

# Build Grid
print("Building grid.")

P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P0gridii =  np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]

        P0gridii[iis] = compute_Ckg_table(coord, dndz, zeff)

comm.Barrier()
comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)

# Now compute the derivatives
print("Computing derivatives.")

if mpi_rank == 0:

    derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)
    list0   = [ dd.tolist() for dd in derivs0 ]
    # and add them to the dictionary.
    emu_dict = {'params': param_str,\
            'x0': x0s,\
            'ell': np.arange(1251).tolist(),\
            'derivs': list0}

    # Write the results to file.
    outfile   = db + '/emu/ckg_%s.json' %(suffix)
    json_file = open(outfile, 'w')
    json.dump(emu_dict, json_file)
    json_file.close()