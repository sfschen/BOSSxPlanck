import numpy as np
from mpi4py import MPI
import sys

z = float(sys.argv[1])
fid_dists = np.loadtxt('fid_dists_z_%.2f.txt'%(z))

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

# Set up the output k vector:
from compute_pell_tables import compute_pell_tables, kvec

output_shape = (2,len(kvec),19) # two multipoles and 19 types of terms


# First construct the grid
order = 4
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.01, 0.01, 0.05]

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
        p0, p2, p4 = compute_pell_tables(coord,z=z,fid_dists=fid_dists)
        
        fb = 'data/boss_z_%.2f/'%(z)
        
        np.savetxt(fb + 'boss_p0_%d_%d_%d.txt'%(iis),p0)
        np.savetxt(fb + 'boss_p2_%d_%d_%d.txt'%(iis),p2)
        np.savetxt(fb + 'boss_p4_%d_%d_%d.txt'%(iis),p4)

