<!-- JC note: This tutorial is copied from: https://rosengroup.slite.page/p/KYrelPp5j3h_qe/SeeK-Path-Band-Structures -->
<!-- I have reviewed this and agree with the info here. I have slightly modified certain text to be specific to Kestrel. -->

# Purpose

In this document, I go over how to calculate the band structure of a material using a convenient code called [SeeK-Path](https://github.com/giovannipizzi/seekpath). In short, to calculate the band structure of a material, you need to generate a _k_-path that goes through high symmetry points in the reciprocal lattice. It is not trivial to do so except in very simple cases. SeeK-Path helps generate these high symmetry points for you.

# Guide

1. If you haven't done so already, carry out a geometry optimization on your material to make sure it is at a local minimum in the potential energy surface.
2. Calculate the `WAVECAR` and `CHGCAR` files via `LWAVE = .True.` and `LCHARG = .True.`, respectively. This is a self-consistent field (SCF) calculation.
3. Using your DFT-optimized structure (e.g. the `CONTCAR`), upload the crystal structure to the [SeeK-path](https://www.materialscloud.org/work/tools/seekpath) GUI and click "Calculate this structure."
4. SeeK-path will generate two important pieces of information: the standard primitive cell for the band structure calculation and its corresponding _k_-path. This structure may be a different representation than the one from your geometry optimization, and you must make sure you use the one that is consistent with the generated _k_-path.
    1. Using the data on SeeK-path page, update your structure to have the lattice vectors and coordinates shown. To do this, I often take the `CONTCAR` from the prior calculation and copy/paste the new coordinates and lattice vectors over the original. Always view the resulting structure (e.g. in VESTA) to make sure it looks right.
    2. For the _k_-path, click the "VASP KPOINTS input for LDA/GGA" drop-down and copy the text to your clipboard. This will be your new `KPOINTS` file. The only thing you need to change is the `<...>` in the second line, which should be replaced by an integer containing the number of points you would like between each high-symmetry point. A value of 20 is typically fairly reasonable to try.
5. With your updated structure and your _k_-path, run a non-self-consistent field (NSCF) calculation in VASP. This calculation is nearly identical to a standard single-point calculation except that you need to set `ICHARG = 11` to start from the previously converged `CHGCAR`. You will also need to use the new `KPOINTS` file you made and disable any `kspacing` flags you may have been using.


