import numpy as np
import sys
from mpi4py import MPI
import json

from compute_fid_dists import compute_fid_dists
from compute_xiell_tables import compute_xiell_tables
from taylor_approximation import compute_derivatives


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] + '/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Set r grid:
rmin, rmax, dr = 50, 160, 0.5

# compute fiducial distances:
fid_dists = compute_fid_dists(z,Omfid)

# Remake the data grid:
order = 4
Npoints = 2*order + 1
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s)
dxs = [0.01, 0.01, 0.05]

# Set output shape
rr = np.arange(rmin, rmax, dr)
output_shape = (len(rr),6) # this is for 1, B1, F, B1*F, B1^2, F^2

# Make parameter grid:
template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

# Compute the grid!
X0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
X2grid = np.zeros( (Npoints,)*Nparams+ output_shape)

X0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
X2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        xi0, xi2= compute_xiell_tables(coord,z=z,fid_dists=fid_dists, rmin=rmin, rmax=rmax, dr=dr)
        
        X0gridii[iis] = xi0
        X2gridii[iis] = xi2
        
comm.Allreduce(X0gridii, X0grid, op=MPI.SUM)
comm.Allreduce(X2gridii, X2grid, op=MPI.SUM)

del(X0gridii, X2gridii)
            
# Now compute the derivatives

if mpi_rank == 0:

    derivs0 = compute_derivatives(X0grid, dxs, center_ii, 5)
    derivs2 = compute_derivatives(X2grid, dxs, center_ii, 5)

    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    # Now save:
    outfile = basedir + 'emu/boss_z_%.2f_xiells.json'%(z)

    list0 = [ dd.tolist() for dd in derivs0 ]
    list2 = [ dd.tolist() for dd in derivs2 ]

    outdict = {'params': ['omegam', 'h', 'sigma8'],\
           'x0': x0s,\
           'rvec': rr.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,}

    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()
