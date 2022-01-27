# Emulator.

This directory contains files for building and calling
Taylor-series-based emulators for the power spectra.
Grids of spectra are built, spanning a fiducial model with
some offsets in each parameter directory, using e.g.

srun -n 16 -c 4 python build_grid_preal.py $PWD 0.38

to build the grids necessary for the z=0.38 real-space P(k)
emulator in a subdirectory off of the current working
directory (i.e. $PWD).

These grids are then finite-differenced (using the FinDiff
package, which is pip install-able) to form derivatives
of P(k), stored in JSON files, using e.g.

srun -n 1 -c 64 python compute_derivatives_preal.py $PWD 0.38 0.61

Such a file is finally used as the input to the emulator
(e.g. emulator_preal.py) class.  This class uses the Taylor
series to compute P(k) quickly given a parameter vector.
