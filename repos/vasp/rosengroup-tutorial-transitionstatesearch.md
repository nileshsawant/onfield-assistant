<!-- JC note: This tutorial is copied from: https://rosengroup.slite.page/p/DMdg_HLKGHX1_b/Transition-State-Searches -->
<!-- I have reviewed this and agree with the info here. I have slightly modified certain text to be specific to Kestrel. -->

# VTST Scripts

The Henkleman group has [VTST Scripts](https://theory.cm.utexas.edu/vtsttools/scripts.html) for interfacing with transition state calculation in VASP. For the machines where we have compiled our own custom VASP module, when you load the VASP module you will also have these scripts added to your `PATH` automatically.

# VTST Tools

VASP has its own internal routines for [NEB](https://www.vasp.at/wiki/index.php/Nudged_elastic_bands) and [dimer](https://www.vasp.at/wiki/index.php/Improved_dimer_method) calculations. The instructions and parameters below are **not** applicable to the internal VASP routines. It does not matter which set of routines you use, although VTST Tools has some nice additional features at the time of writing.

The Henkelman group has a patch for VASP called [VTST Tools](https://theory.cm.utexas.edu/vtsttools/index.html) that provides several nice features related to transition state searches. On Kestrel, look for the `+tpc` suffix (i.e., "third-party codes") in the module name (e.g. `module load vasp/XXXX+tpc`) for a VASP build with VTST Tools enabled. 

## Nudged Elastic Band Calculation

In general, to find transition states, your best bet is to start with a nudged elastic band (NEB) calculation as described below:

1. Relax the initial and final structures as normal.
2. Interpolate between the two via `nebmake.pl POSCAR1 POSCAR2 N` where `N` is the number of images you want, excluding the initial and final structures. Starting with 6 images is fairly reasonable. The `00` and final image directories correspond to the initial and final structures and will not be updated during the course of the NEB calculation.
    1. Before proceeding, do a sanity check and make sure that the images that are generated look reasonable.
    2. Note that you are running `N` structure relaxations concurrently, so you may need to increase the number of nodes accordingly.
3. For the initial and final images, put the `OUTCAR` alongside the `POSCAR` in their corresponding directories.
4. Run an [Nudged Elastic Band (NEB) calculation](https://theory.cm.utexas.edu/vtsttools/neb.html) with `ICHAIN = 0`, `IMAGES=N`, `IBRION=3`, `POTIM=0`, and `IOPT=1`. If you would like to run a Climbing Image NEB (CI-NEB) calculation, you can also set `LCLIMB = .TRUE.` in your INCAR file. Sometimes you can run a CI-NEB calculation right away, but in general it helps to do a regular NEB calculation first (perhaps with a relatively coarse value of `EDIFFG`, such as -0.1). Your other parameters should be those as usual for a structure optimization.
    1. There is no need to use super high-quality settings yet. Decrease _k_-point grids and other settings as needed if you need to move quickly. You will be refining the calculation later.
    2. If your NEB is struggling to converge, consider switching to `IOPT=7` or increase the number of images.
5. Over the course of the calculation, check that the NEB results look reasonable with `nebef.pl` . This will show the residual forces, the absolute energy, and the relative energy of each image. You can also view the trajectory of your NEB via `ase gui 0*/CONTCAR`.
6. When your NEB run has completed or reached the walltime, clean it up with `vfin.pl neb_run1`. This will store the compressed outputs in a new folder called `neb_run1` and prepare the parent directory for a new NEB calculation by copying the `CONTCAR` files to `POSCAR` files.
    1. If you have already done this process and have an existing `neb_run1` folder, make sure to give the directory argument a new name (e.g. `neb_run2`) to not overwrite your prior run.
7. In the `neb_run1` folder that you made, run `nebresults.pl` to generate some summaries of your completed NEB run.
    1. There is an `.eps` file that you can view in a program like Adobe Illustrator, which contains the minimum energy pathway.

That's it! From the parent directory (i.e. the directory above `neb_run1`), you can now rerun your NEB calculation (e.g. with higher quality settings), run a CI-NEB calculation, or use it as staging for a dimer calculation (described below). In general, the regular NEB calculation will only give you an estimate the transition state. A CI-NEB or dimer calculation is needed to find the true transition state structure.

## Dimer Calculation

To run a Dimer calculation, first start with an NEB calculation described above. Once you have completed each step in the above process, do the following:

1. Copy the `exts.dat` file in the `neb_run1` folder to the parent directory (i.e. the directory above `neb_run1`).
2. In the same parent directory, run `neb2dim.pl` to set up a dimer calculation. This will make a folder named `dim` where the input files for the dimer calculation are prepared.
3. In the generated `dim` folder, run a [dimer calculation](https://theory.cm.utexas.edu/vtsttools/dimer.html) to finalize the transition state search starting from your reasonable transition state guess obtained from the NEB method. Use `ICHAIN=2`, `IBRION=3`, `POTIM=0`, and `IOPT=2`. Delete the `IMAGES = N` flag if it's still there. Other parameters should match that of a typical structure relaxation. Do not retain the NEB input arguments.
    1. To monitor and evaluate the success of a dimer calculation, you should look at the `DIMCAR` file, as described in the [VTST documentation](https://theory.cm.utexas.edu/vtsttools/dimer.html#dimer-output). Make sure the dimer calculation has converged, which will be indicated by a `---` in the `DIMCAR`.
    2. To continue from a previous dimer calculation, run `mv CONTCAR POSCAR` and `mv NEWMODECAR MODECAR` and re-submit the job.
    3. If you struggle to reach convergence, consider switching to `IOPT=7` or use a better guess for the transition state. Use your high-quality settings here once you are close to the transition state.

The above process is oftentimes faster than running a full CI-NEB calculation, especially if you can get a reasonable guess for the NEB minimum energy pathway quickly. However, if you have trouble converging with the dimer method, the CI-NEB method is a good fallback.

# Intrinsic Reaction Coordinate Calculations

An intrinsic reaction coordinate (IRC) calculation will tell you if the transition state you found connects your proposed reactants and products and that there are no other transition states in between that you missed. It will also more thoroughly map out the true minimum energy pathway than would otherwise be possible with the NEB method.

The IRC calculation takes the transition state as the input and will run two calculations — one where the transition state is nudged backward and the other forward along the imaginary mode, until they each reach the reactants and products, respectively.

The best way to run an IRC calculation is to use the [method that is natively within VASP](https://www.vasp.at/wiki/index.php/Intrinsic-reaction-coordinate_calculations). It requires that you set `IBRION = 40` along with the typical parameters for a structure relaxation (the IRC method only supports `ISIF = 2` at this time). You will need to run two calculations, one in the backward and one in the forward directions, which are specified via `IRC_DIRECTION = -1` and `IRC_DIRECTION = 1`. From the two trajectories, you can reconstruct the IRC.

You should use the `CONTCAR` from the transition state of a dimer calculation as the `POSCAR` for the IRC calculation. You must also append to the bottom of your `POSCAR` the direction associated with the instability, which you can obtain from the `NEWMODECAR` of a dimer run if you are using the VTST-based dimer method. Note that VASP recommends that you start from a well-converged calculation with a very low `EDIFFG` (as low as -0.005, although this is likely extremely difficult to achieve in practice).

Especially near the minima (i.e. the reactant and product) where the potential energy surface is shallow, it can be nearly impossible to get an IRC calculation to fully converge to the endpoints. In practice, if you run the IRC calculation for enough steps, and it's clear that it is essentially guaranteed to converge to the proposed reactant and product, then you can call it a day. Optimize the endpoints of the IRC run to confirm. When you plot the IRC path though, be sure to only include the true IRC points and the endpoints (i.e. not the results from any geometry optimization).

Visualize the pathway using this script once the IRC has run (on a tiger visualization node):

```
from ase.visualize import view
```

If you are struggling with the IRC calculation, a quick but far less precise approach is to first run a dimer calculation, clean up the result with `vfin.pl dimer1` from VTST Scripts, and then do `dimmins.pl POSCAR MODECAR <displacement>` on the resulting files where `<displacement>` is the displacement in angstroms (e.g. 0.1 or -0.1). This will not yield a true IRC path, but a structure relaxation from the displaced structures should likely still go to the desired reactant and product. An IRC calculation would still be best.

