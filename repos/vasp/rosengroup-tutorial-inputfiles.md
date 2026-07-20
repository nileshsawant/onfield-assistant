<!-- JC note: This tutorial is copied from: https://rosengroup.slite.page/p/0CKuG1eLlFb3Lc/Input-Files -->
<!-- I have reviewed this and agree with the info here. I have slightly modified certain text to be specific to Kestrel. -->

# Overview

The [VASP manual](https://www.vasp.at/wiki/index.php/The_VASP_Manual) is the "holy bible" of VASP. Read it with care when you have questions. The [official VASP tutorials](https://www.vasp.at/tutorials/latest/) and the [official VASP YouTube channel](https://www.youtube.com/@vasp8588/videos) are valuable "getting started" resources.

There are four key input files needed to run a VASP calculation:

1. The `INCAR`: this contains most of the input flags.
2. The `POSCAR`: this contains the input structure.
3. The `KPOINTS`: this contains details about the k-point grid.
4. The `POTCAR`: this contains the pseudopotential details.

## INCAR

[INCAR](https://slite.com/api/files/JO5ZDoG1YxJhG1/incar)

There are a [wide assortment](https://www.vasp.at/wiki/index.php/Category:INCAR_tag) of input arguments that VASP can take in the `INCAR` file, and there are many subtle rules for when to say which flags to what value. The group's software infrastructure will automatically make sure that your `INCAR` settings are compatible for the most part.

Please do not try to memorize all the INCAR flags. This is merely for ease of reference. That said, after some time using VASP, you will end up knowing more about these flags than you would ever wish.

🟢 = "Very important; must learn for running your first calculations."

🟨 = "Reasonably important but not needed right way; learn after you've got the basics down."

♦️ = "Fairly specialized. Come back to this later."

### Level of Theory

- 🟢 `ENCUT`: this sets the plane-wave kinetic energy cutoff. The higher the value, the more precise your calculations will be at the expense of higher computational cost. At the top of each element's `POTCAR` file is a parameter named `ENMAX`. The value of `ENCUT` , if not explicitly set, is the largest `ENMAX` value across the elements in your system. In practice, you should set `ENCUT` to at least 1.3 times this maximum `ENMAX` value for any structure relaxation involving changes to the cell volume in order to minimize artificial stresses.

- Rule of thumb: Try at least 520 eV to start. In practice, you should ensure this is converged.

- 🟢 `GGA`: this sets the exchange-correlation functional (i.e. the "level of theory"). This has the largest impact on the results of the calculation.

- Rule of thumb: Try `GGA = PE` first (i.e. the PBE functional), particularly if you are just learning how to use VASP.
- If using ASE, the `xc` keyword can be used to set the functional as described in the [documentation](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#exchange-correlation-functionals). It is often more convenient since it will auto-set all relevant flags for that functional (e.g. `xc = "pbe"` will auto-set `GGA = PE`).

- 🟨 `IVDW`: this sets any van der Waals correction scheme. Grimme's DFT-D methods are common in molecular systems where vdW corrections are particularly important.

- Rule of thumb: For typical bulk solids, don't need to set it unless you have reason to. For systems with notable vdW interactions, try `IVDW = 12` first.

- ♦️ `LDAU`/ `LDAUJ` / `LDAUL` / `LDAUPRINT` / `LDAUTYPE` / `LDAUU`: these arguments specify details related to the DFT+U method, which corrects so-called "self-interaction errors" of GGA functionals. If you are new to DFT, do not worry about these parameters yet.
- ♦️ `LHFCALC` / `HFSCREEN`: these methods specify details related to the use of hybrid functionals. Like DFT+U, this can also correct common "self-interaction errors" of GGA functionals.
- ♦️ `METAGGA`: this parameter defines a particular meta-GGA functional, which is generally more accurate than a GGA functional but less accurate than a hybrid functional.

- If using ASE, the `xc` keyword will take care of this flag.

### SCF

- 🟢 `PREC`: specifies the precision used.

- Rule of thumb: Generally set this to `Accurate`.

- 🟢 `EDIFF`: this sets the SCF convergence tolerance (in eV). The lower the value, the more precise your energies and forces will be.

- Rule of thumb: Try `1e-5` to start.

- 🟨 `ALGO`: this sets the SCF convergence algorithm.

- Rule of thumb: Try `Fast` to start. If you have problems, switch to `All` .

- 🟨 `ISMEAR`: this sets how the partial occupancies are treated (i.e. the smearing method) to smooth out discontinuities near the Fermi level. There are a lot of subtle rules about what value to set for what material as laid out in the VASP manual, which our software stack will try and set for you where appropriate.

- Rule of thumb: When in doubt, `ISMEAR = 0` is a good bet with a small value of `SIGMA` (0.01 - 0.05).

- 🟨 `LASPH`: this defines whether non-spherical contributions are included in the energy and forces. Do not mix calculations with `True` and `False` for `LASPH`.

- Rule of thumb: Always set this to `True` to make your life easier.

- 🟨 `LREAL`: setting this value to `Auto` can speed up calculations for large systems but can reduce the quality of the energy and should always be set back to `False` for computed properties of interest.

- Rule of thumb: Always set this to `False` (i.e. the default) to make your life easier, even when VASP complains.

- 🟨 `SIGMA`: specifies the smearing parameter (e.g. width). Like `ISMEAR`, this is very material-dependent.

- Rule of thumb: When in doubt, set this to a small value (0.01 - 0.05) along with `ISMEAR = 0`.

### Geometry Optimization

- 🟢 `EDIFFG`: this sets the maximum net force (in eV/A) for a structure to be considered converged during a geometry optimization. The lower the value, the closer to the true local minimum your structure will be. Note that a negative sign should be used by convention (e.g. `EDIFFG = -0.03`).

- Rule of thumb: Use a value of `-0.03` or tighter (e.g. `-0.01`).

- 🟢 `ISIF`: this defines which degrees of freedom should be modified during the structure relaxation. `ISIF = 2` indicates that the positions are relaxed at fixed cell volume. `ISIF = 3` indicates that the positions and cell shape/volume are simultaneously relaxed.

- Rule of thumb: Use `ISIF = 3` for cell relaxations and `ISIF = 2` otherwise.

- 🟢 `NSW`: specifies the maximum number of geometry optimization steps to consider.

- Rule of thumb: Set to an arbitrarily large value, e.g. 200.

- 🟨 `IBRION`: this is the geometry optimization algorithm.

- Rule of thumb: Try `IBRION = 2` (conjugate gradient) to start.

- 🟨 `ISYM`: this defines whether symmetry constraints should be accounted for.

- Rule of thumb: Set to `0` for geometry optimizations to ensure the symmetry is allowed to change, unless you are intentionally looking to model a given symmetry.

### Magnetization

- 🟢 `ISPIN`: this sets whether the calculation should be run without or with spin-polarization. If the system has unpaired electrons, it should be run with spin-polarization.

- Rule of thumb: When in doubt, start with `ISPIN = 2` to see if there are favorable magnetic states. If there aren't, the default value of `ISPIN = 1` is fine.
- If using ASE, setting `atoms.set_initial_magnetic_moments()` will automatically toggle `ISPIN` accordingly. Don't directly specify `ISPIN` in this case.

- 🟢 `LORBIT`: this parameter specifies how much information related to magnetization is written out to the `OUTCAR` file.

- Rule of thumb: Always set this to `LORBIT = 11` to make your life easier.

- 🟢 `MAGMOM`: this parameter specifies details related to the initial guess for the magnetic moments.

- Rule of thumb: Initialize each transition metal with a high-spin magnetic configuration as a first attempt. Then go back and try other magnetic configurations, if applicable.
- If using ASE, setting `atoms.set_initial_magnetic_moments()` will automatically toggle `MAGMOM` accordingly. Don't directly specify `MAGMOM` in this case.

- ♦️ `NUPDOWN`: forces a particular net number of unpaired electrons.

### File I/O

- 🟨 `ISTART`: this defines if the VASP calculation should restart from a pre-existing wavefunction. If a `WAVECAR` is present in the same directory, VASP will automatically try to restart from it.

- Rule of thumb: No need to change the default value in most cases.

- 🟨 `LWAVE`: this parameter specifies whether the wavefunction is written out to the filesystem. This is especially useful for restarting a calculation but is quite large.
- 🟨 `LCHARG`: whether to write out the charge density to the filesystem. This is useful for post-processing but is quite large.
- ♦️ `LAECHG`: this specifies whether `AECCAR` files are written out, which are used by certain post-processing codes. Never use `LAECHG` if `NSW` is not set to 0.
- ♦️ `NEDOS`: describes how many points are written out in the density of states output file (`DOSCAR`).

### Miscellaneous

- 🟨 `EFERMI`: VASP recommends generally just setting this to `MIDGAP`. That will yield slightly more consistent Fermi energies (and, as a result, band gaps) but otherwise changes nothing.

- Rule of thumb: Always set this to `MIDGAP` to make your life easier if you are using VASP 6.4+.

- 🟨 `NCORE`: this parameter can be tuned for a specific machine and system to speed up calculations by parallelizing over orbitals. The only way to know which value is best is to try them out! It should generally be used in place of `NPAR`.
- 🟨 `KPAR`: similarly, this parameter tells VASP how to parallelize the k-points. It only makes sense to use if there are many k-points, as a result.

## POSCAR

[POSCAR](https://slite.com/api/files/ep9Dl-AEQqL247/poscar)

The `POSCAR` file contains all information needed to define the input structure. The easiest way to generate a `POSCAR` file is using a program like [ASE](https://rosengroup.slite.page/p/BGJeKFLJdSPpJK), [Pymatgen](https://rosengroup.slite.page/p/qDrLuoOoVpTUTh), or a visualization program like VESTA. A code-snippet to convert a crystal structure in `.cif` format to a `POSCAR` file is shown below:

```
from ase.io import read, write
```

Alternatively, with Pymatgen:

```
from pymatgen.core import Structure
```

## KPOINTS

#### KSPACING

VASP now recommends using the parameter `KSPACING` in the `INCAR` file in place of a dedicated `KPOINTS` file. The `KSPACING` parameter is such that a smaller value corresponds to a higher _k_-point density (i.e. greater precision). Nonetheless, it is still useful to know what the `KPOINTS` file is since you may see many instances of it in the wild. For more information about the `KSPACING` parameter, refer to Section III of [this document](https://drive.google.com/file/d/1fUUx0wrrtMRcSss5yv3NiQuC7J5IiEKL/view). A convenience function can be found [in quacc](https://github.com/Quantum-Accelerators/quacc/blob/f1b4ce80df5051e956c025d93b3751400853b2b5/src/quacc/utils/kpts.py#L106-L109).

#### The `KPOINTS` File

[KPOINTS](https://slite.com/api/files/4Upvb-Th4PDR2V/kpoints)

The `KPOINTS` file encodes the density of the k-point grid. A greater number of k-points typically indicates a more precise calculation. The number of required k-points to reach convergence is typically smaller as a given lattice dimension increases (i.e. large unit cells require fewer total k-points, on average). If you are using ASE, the `KPOINTS` file will be automatically written out based on the value of the `kpts` keyword argument.

There are many schemes in the [Pymatgen](https://rosengroup.slite.page/p/qDrLuoOoVpTUTh) code for automatically setting appropriate _k_-point values. One such example is shown below:

```
from pymatgen.io.vasp.inputs import Kpoints
```

If you are using a 1x1x1 k-point grid, this is referred to as a gamma-point only calculation, and you can use the `vasp_gam` executable instead of the `vasp_std` executable for a significant speedup.

## POTCAR

VASP ships with several different variations of pseudopotentials to use. Except in unusual circumstances, you typically want to use the most up-to-date version of the `potpaw_PBE` pseudopotentials. Even if you are not using the PBE functional, these are generally still the pseudopotentials you should use. On our machines, the pseudopotentials can be found in the same parent folder that contains the VASP source code (e.g. `/home/ROSENGROUP/software/vasp/vasp_potcars` on Tiger). At the time of writing, the most up to date pseudopotential set is the .64 set.

VASP has a [recommended set](https://www.vasp.at/wiki/index.php/Available_PAW_potentials#Recommended_potentials_for_DFT_calculations) of pseudopotentials to use. The recommendations, for the most part, are reasonable. If using ASE, the pseudopotentials will be automatically selected on the basis of the `setups` [keyword argument](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#setups).

The VASP POTCAR files are proprietary and should never be made publicly available (e.g. on GitHub, Zenodo, or anywhere else). Also, under no circumstances should you edit the POTCAR files.

