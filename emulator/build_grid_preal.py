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

from mpi4py import MPI

if len(sys.argv!=3):
    raise RuntimeError("Usage: "+sys.argv[0]+" <basedir> <z>")
db= sys.argv[1]
z = float(sys.argv[2])

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print("Hello I am process {:d} of {:d}.".format(mpi_rank, mpi_size))

# Set up the output k vector:
from compute_preal_tables import compute_preal_tables, kvec

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

Fs = np.zeros( (Npoints,)*Nparams + output_shape )

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        
        preal = compute_preal_tables(coord,z=z)
        
        fb = db+'/data/preal/preal_z_%.2f/'%(z)
        if not os.path.isdir(fb):
            os.mkdir(fb)
        np.savetxt(fb+'preal_%d_%d_%d.txt'%(iis),preal)
