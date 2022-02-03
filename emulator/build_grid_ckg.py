# Make the grid:
import numpy as np
import sys
import os

from mpi4py import MPI
from compute_ckg_table import compute_Ckg_table

# Load the sample properites
db= sys.argv[1]
zeff = float(sys.argv[2])
dndz = np.loadtxt(sys.argv[3])
suffix = sys.argv[4]

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print("Hello I am process {:d} of {:d}.".format(mpi_rank, mpi_size))

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

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        #print(coord,iis)
        
        pnl = compute_Ckg_table(coord, dndz, zeff)
        
        fb = db+'data/ckg/ckg_%s/'%(suffix)
        if not os.path.isdir(fb):
            os.mkdir(fb)
        np.savetxt(fb+'ckg_%s_%d_%d_%d.txt'%(suffix,*iis),pnl)