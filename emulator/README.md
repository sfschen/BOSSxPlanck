# Emulator.

This directory contains files for building and calling
Taylor-series-based emulators for the power spectra.
Grids of spectra are built, spanning a fiducial model with
some offsets in each parameter directory.  These grids are
then finite-differenced (using the FinDiff package, which
is pip install-able) to form derivatives of e.g. P(k), stored
in JSON files.  For example to build the JSON files for the
Pell(k) emulators you could run:

Omfid=0.31
for zeff in 0.38 0.61 ; do
  srun -n 16 -c 4 python make_emulator_pells.py $PWD $zeff $Omfid
done

Similar "make_emulator" files exist for xiells, sigma8, etc.

The JSON files are finally used as the input to the emulator
(e.g. emulator_preal.py) class.  This class uses the Taylor
series to compute P(k) quickly given a parameter vector.
