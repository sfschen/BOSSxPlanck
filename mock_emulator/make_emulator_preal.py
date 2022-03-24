#!/usr/bin/env python
#
# Generates a grid of theoretical predictions spanning a
# range of parameters.  These can then be finite-differenced
# to compute derivatives that are in turn used for Taylor
# series expansion as an emulation strategy.
#
import numpy as np
import sys
import os
import json

from mpi4py import MPI

from compute_preal_tables import compute_preal_tables, kvec
from taylor_approximation import compute_derivatives


if len(sys.argv)<=2:
    raise RuntimeError("Usage: "+sys.argv[0]+" <basedir> <z1 z2 z3 ...>")
db= sys.argv[1]
zs = sys.argv[2:]; zs = [float(z) for z in zs]

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
print("Hello I am process {:d} of {:d}.".format(mpi_rank, mpi_size))

# Set up the output k vector:

output_shape = (len(kvec),14) # 13 types of terms + kvec

# First construct the grid
order = 4
# these are settings for OmegaM, h, sigma8
#x0s = [0.31, 0.68, 0.73]# these are chosen to be roughly at the BOSS best fit value
#dxs = [0.01, 0.01, 0.05]

# these are settings for OmegaM, h, lnAs
param_str = ['omegam', 'h', 'logA']
x0s = [0.31, 0.68, 2.84]
dxs = [0.01, 0.01, 0.05]

Nparams = len(x0s)

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

output_shape = (len(kvec),14) # first row is kv

# Build grid
print("Building grid.")

P0grids = {}

for z in zs:

    P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
    P0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

    for nn, iis in enumerate(zip(*Inds)):
    
        if nn%mpi_size == mpi_rank:
            coord = [Coords[i][iis] for i in range(Nparams)]
            print(coord,iis)
        
            P0gridii[iis] = compute_preal_tables(coord,z=z)

    comm.Barrier()
    comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)
    
    P0grids[z] = np.array(P0grid)

# Now compute the derivatives
print("Computing derivatives.")
emu_dict = {}

if mpi_rank == 0:
    
    for z in zs:
    
        P0grid = P0grids[z]
        # Now compute the derivatives
        derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)
        list0   = [ dd.tolist() for dd in derivs0 ]
        # and add them to the dictionary.
        emu_dict[z] = {'params': param_str,\
                   'x0': x0s,\
                   'kvec': kvec.tolist(),\
                   'derivs': list0}

    # Write the results to file.
    outfile   = db + '/emu/preal.json'
    json_file = open(outfile, 'w')
    json.dump(emu_dict, json_file)
    json_file.close()

