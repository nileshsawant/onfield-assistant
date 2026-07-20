<!-- JC note: This tutorial is copied from: https://rosengroup.slite.page/p/i1TohYw0XoX2Km/Output-Files -->
<!-- I have reviewed this and agree with the info here. I have slightly modified certain text to be specific to Kestrel. -->

In addition to ASE (described below), the official [py4vasp](https://vasp.at/py4vasp/latest/) Python package can be used to automatically parse most VASP output parameters. [Pymatgen](https://rosengroup.slite.page/p/qDrLuoOoVpTUTh) also has [several utilities](https://pymatgen.org/pymatgen.io.vasp.html) for VASP output parsing.

# OUTCAR

## Overview

[OUTCAR](https://slite.com/api/files/7a2FKBHib7aIvV/outcar)

The primary output file in VASP is called the OUTCAR file. There is a ton of info in it. Here are the most important things:

- `Iteration`: This line specifies the iteration of the geometry optimization procedure _and_ the SCF loop for that given geometry. A value of `Iteration      4(   2)` would indicate "geometry step 4, SCF iteration 2."
- `energy(sigma->0)`: The value following this line is the current iteration's energy. The unit is eV.
- `magnetization (x)`: If `LORBIT` is set to >= 11, the atom-wise spin moments are printed in this table, each in units of Bohr-magnetons.

You can also use ASE to easily parse the most critical information (i.e. energy, forces, trajectory) from the `OUTCAR` as follows:

```
from ase.io import read
```

## Live Monitoring

It is often useful to monitor the progress of a calculation in real-time. This can be done using the `grad2` utility, which is available when you load one of our VASP modules.

When you want to monitor a calculation, run `grad2 OUTCAR` from the command-line in any directory containing the `OUTCAR` you want to monitor.

The output looks like the following:

```
1  Energy:  -725.474813  Log|dE|:  2.861  SCF:  12  Avg|F|:  0.146  Max|F|:  0.247  Vol.: 4326.3  Time:  0.23m
```

Do not use the energy from `grad2` for generating results and figures. The energy here is the force-consistent energy, whereas the energy used for any type of analysis is the `energy(sigma->0)` value. The `energy(sigma->0)` value is the one returned by ASE's `.get_potential_energy()` function.

# CONTCAR

The `CONTCAR` file is formatted identically as the `POSCAR` file but contains the most recent set of atomic positions and cell parameters. You can view the `CONTCAR` and `POSCAR` in programs like VESTA or ASE. When restarting a calculation, you can copy the `mv CONTCAR POSCAR` to continue the geometry optimization process.

# Volumetric Files

- `CHGCAR`: If `LCHARG = .TRUE.`, the charge density will be written out in this file.
- `WAVECAR`: If `LWAVE = .TRUE.`, the wavefunction will be written out in this file. This is particularly useful when restarting calculations, as they will automatically read this file.

# Other Files

- `DOSCAR`: Data about the density of states.
- `EIGENVAL`: Data about the Kohn-Sham eigenvalues.
- `OSZICAR`: Contains data for each SCF iteration. Useful for identifying SCF convergence issues. This is the energy data displayed via the `grad2` utility.

