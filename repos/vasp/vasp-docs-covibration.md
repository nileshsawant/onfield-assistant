# VASP · Covibration

(source: vasp-docs-covibration.html)

{{Template:At_and_mol - Tutorial}}

== Task ==

Calculation of the vibrational frequencies of a CO molecule.

== Input ==

=== POSCAR ===
 CO molecule in a box
  1.0          ! universal scaling parameters
  8.0 0.0 0.0  ! lattice vector  a(1)
  0.0 8.0 0.0  ! lattice vector  a(2)
  0.0 0.0 8.0  ! lattice vector  a(3)
 1 1           ! number of atoms for each species
 sel           ! selective degrees of freedom are changed
 cart          ! positions in cartesian coordinates
  0 0 0       F F T  ! first atom
  0 0 1.143   F F T  ! second atom

Alternatively, try to fix one of the atoms completely.

=== INCAR ===
 SYSTEM = CO molecule in a box
 ISMEAR = 0   ! Gaussian smearing
 IBRION = 5   ! calculate second derivatives, Hessian matrix, and phonon frequencies
              ! from finite differences
 NFREE = 2    ! central differences
 POTIM = 0.02 ! 0.02 A stepwidth
 NSW = 1      ! ionic steps > 0

=== KPOINTS ===
 Gamma-point only
  0
 Monkhorst Pack
  1 1 1
  0 0 0

== Calculation ==

*The selected degrees of freedom are displaced once in the direction \hat{x} and once in -\hat{x} by 0.02 \AA (POTIM).

*In the present case this makes 4 displacements plus the equilibrium positions (i.e. a total of five ionic configurations).

=== OUTCAR ===

At the end of the OUTCAR file the following output should be obtained:

 SECOND DERIVATIVES (NOT SYMMETRIZED)
 ------------------------------------
               1Z          2Z
  1Z  -114.737304  114.737304
  2Z   114.458316 -114.458316


 Eigenvectors and eigenvalues of the dynamical matrix
 ----------------------------------------------------


   1 f  =   63.887522 THz   401.417139 2PiTHz 2131.058277 cm-1   264.217647 meV
             X         Y         Z           dx          dy          dz
      0.000000  0.000000  0.000000            0           0   -0.655280
      0.000000  0.000000  1.143000            0           0    0.755386

   2 f/i=    0.038494 THz     0.241864 2PiTHz    1.284016 cm-1     0.159198 meV
             X         Y         Z           dx          dy          dz
      0.000000  0.000000  0.000000            0           0   -0.755386
      0.000000  0.000000  1.143000            0           0   -0.655280

== Download ==
 COvib.tgz

{{Template:At_and_mol}}

Back to the main page.

Category:Examples
