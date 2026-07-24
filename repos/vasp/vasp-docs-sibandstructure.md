# VASP · Sibandstructure

(source: vasp-docs-sibandstructure.html)

{{Template:Hybrid_functionals - Tutorial}}

== Task ==

Calculation of the bandstructure for Si within DFT+HF.

The bandstructure in VASP can be obtained following three different procedures. The standard procedure (procedure 1),

applicable at PBE level, is also described in Fcc Si bandstructure example.

Within Hybrid functional theory it is possible to plot bandstructure using procedure 2 or 3.

== Procedure 1: Standard procedure (DFT)==
Only possible within DFT. Described in Fcc Si bandstructure example:

=== Standard self-consistent (SC) run ===

*POSCAR

```

system Si
5.430
0.5 0.5 0.0
0.0 0.5 0.5
0.5 0.0 0.5
2
cart
0.00 0.00 0.00
0.25 0.25 0.25

```

*INCAR
 System = fcc Si
 ISMEAR = 0; SIGMA = 0.1;

*KPOINTS

```

4x4x4
 0
G
 4 4 4
 0 0 0

```

=== Non-SC calculation (ICHARG=11) ===

Use preconverged CHGCAR file and a suitable KPOINTS file (KPOINTS_PBE_bands)

*INCAR
 System = fcc Si
 ICHARG = 11 #charge read file
 ISMEAR = 0; SIGMA = 0.1;
 LORBIT = 11

*KPOINTS
 k-points for bandstructure L-G-X-U K-G
  10
 line
 reciprocal
   0.50000  0.50000  0.50000    1
   0.00000  0.00000  0.00000    1

   0.00000  0.00000  0.00000    1
   0.00000  0.50000  0.50000    1

   0.00000  0.50000  0.50000    1
   0.25000  0.62500  0.62500    1

   0.37500  0.7500   0.37500    1
   0.00000  0.00000  0.00000    1

=== Plot using p4v ===

P4VASP: [http://www.p4vasp.at p4v]

== Procedure 2: 0-weight (Fake) SC procedure (PBE & Hybrids) ==

This procedure can be applied to compute bandstructure at Hybrid functionals and DFT level.

=== Standard DFT run ===
*INCAR
 ## Default
 ISMEAR =  0
 SIGMA  =  0.01
 GGA    = PE

 ## HSE
 #LHFCALC = .TRUE. ; HFSCREEN = 0.2 ; AEXX = 0.25
 #ALGO = D ; TIME = 0.4 ; LDIAG = .TRUE.

*KPOINTS

```

Automatically generated mesh
 0
G
 4 4 4
 0 0 0

```

=== Hybrid calculation using a suitably modified KPOINTS file ===

*INCAR
 ## Default
 ISMEAR =  0
 SIGMA  =  0.01
 GGA    = PE

 ## HSE
 LHFCALC = .TRUE. ; HFSCREEN = 0.2 ; AEXX = 0.25
 ALGO = D ; TIME = 0.4 ; LDIAG = .TRUE.

*KPOINTS_HSE_bands (see README.txt)

```

Explicit k-points list
      18
Reciprocal lattice
    0.00000000000000    0.00000000000000    0.00000000000000             1
    0.25000000000000    0.00000000000000    0.00000000000000             8
    0.50000000000000    0.00000000000000    0.00000000000000             4
    0.25000000000000    0.25000000000000    0.00000000000000             6
    0.50000000000000    0.25000000000000    0.00000000000000            24
   -0.25000000000000    0.25000000000000    0.00000000000000            12
    0.50000000000000    0.50000000000000    0.00000000000000             3
   -0.25000000000000    0.50000000000000    0.25000000000000             6
0.00000000 0.00000000 0.00000000 0.000
0.00000000 0.05555556 0.05555556 0.000
0.00000000 0.11111111 0.11111111 0.000
0.00000000 0.16666667 0.16666667 0.000
0.00000000 0.22222222 0.22222222 0.000
0.00000000 0.27777778 0.27777778 0.000
0.00000000 0.33333333 0.33333333 0.000
0.00000000 0.38888889 0.38888889 0.000
0.00000000 0.44444444 0.44444444 0.000
0.00000000 0.50000000 0.50000000 0.000

```

=== Plot using p4v ===

P4VASP: [http://www.p4vasp.at p4v]

'''Mind''': Remove from the bandstructure plot the eigenvalues corresponding to the the regular k-points mesh.

== Procedure 3: VASP2WANNIER90 (GW, Hybrids, PBE)==
Wannier function interpolation using the VASP2WANNIER90 interface.
Applicable in all cases (here applied for hybrids; for GW see Bandstructure_of_Si_in_GW_(VASP2WANNIER90) example).

=== Standard DFT run ===
*INCAR
 ## Default
 ISMEAR =  0
 SIGMA  =  0.01
 GGA    = PE

 ## HSE
 #LHFCALC = .TRUE. ; HFSCREEN = 0.2 ; AEXX = 0.25
 #ALGO = D ; TIME = 0.4 ; LDIAG = .TRUE.

 ##VASP2WANNIER
 #LWANNIER90=.TRUE.

*KPOINTS

```

Automatically generated mesh
 0
G
 4 4 4
 0 0 0

```

=== HSE + LWANNIER90 run ===

*INCAR
 ## Default
 ISMEAR =  0
 SIGMA  =  0.01
 GGA    = PE

 ## HSE
 LHFCALC = .TRUE. ; HFSCREEN = 0.2 ; AEXX = 0.25
 ALGO = D ; TIME = 0.4 ; LDIAG = .TRUE.

 ##VASP2WANNIER
 LWANNIER90=.TRUE.

Use the wannier90.win file given below which contains all instructions needed to generate the necessary input files
for the WANNIER90 runs (wannier90.amn, wannier90.mmn, wannier90.eig).

'''Mind''': If the wannier90.win file does not exist VASP will create a default wannier90.win compatible with
the POSCAR and INCAR files, which needs to be suitably modified by including the proper instruction required
to generate the maximally localized wannier functions (refer to the [http://www.wannier.org/doc/user_guide.pdf WANNIER90 manual]).

*wannier90.win

```

num_wann=8
num_bands=8

Begin Projections
Si:sp3
End Projections

dis_froz_max=9
dis_num_iter=1000

guiding_centres=true

#restart         =  plot
#bands_plot      =  true
#begin kpoint_path
#L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
#G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
#X 0.50000  0.00000 0.5000 K 0.37500 -0.37500 0.0000
#K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
#end kpoint_path
#bands_num_points 40
#bands_plot_format gnuplot xmgrace

begin unit_cell_cart
     2.7150000     2.7150000     0.0000000
     0.0000000     2.7150000     2.7150000
     2.7150000     0.0000000     2.7150000
end unit_cell_cart

begin atoms_cart
Si       0.0000000     0.0000000     0.0000000
Si       1.3575000     1.3575000     1.3575000
end atoms_cart

mp_grid =     4     4     4

begin kpoints
     0.0000000     0.0000000     0.0000000
     0.2500000     0.0000000     0.0000000
     0.5000000     0.0000000     0.0000000
     0.2500000     0.2500000     0.0000000
     0.5000000     0.2500000     0.0000000
    -0.2500000     0.2500000     0.0000000
     0.5000000     0.5000000     0.0000000
    -0.2500000     0.5000000     0.2500000
     0.0000000     0.2500000     0.0000000
     0.0000000     0.0000000     0.2500000
    -0.2500000    -0.2500000    -0.2500000
    -0.2500000     0.0000000     0.0000000
     0.0000000    -0.2500000     0.0000000
     0.0000000     0.0000000    -0.2500000
     0.2500000     0.2500000     0.2500000
     0.0000000     0.5000000     0.0000000
     0.0000000     0.0000000     0.5000000
    -0.5000000    -0.5000000    -0.5000000
     0.0000000     0.2500000     0.2500000
     0.2500000     0.0000000     0.2500000
    -0.2500000    -0.2500000     0.0000000
    -0.2500000     0.0000000    -0.2500000
     0.0000000    -0.2500000    -0.2500000
     0.0000000     0.5000000     0.2500000
     0.2500000     0.0000000     0.5000000
    -0.2500000    -0.2500000     0.2500000
    -0.5000000    -0.2500000    -0.5000000
     0.2500000     0.5000000     0.0000000
     0.2500000    -0.2500000    -0.2500000
    -0.5000000    -0.5000000    -0.2500000
     0.0000000     0.2500000     0.5000000
    -0.2500000     0.2500000    -0.2500000
    -0.2500000    -0.5000000    -0.5000000
     0.5000000     0.0000000     0.2500000
    -0.5000000    -0.2500000     0.0000000
     0.0000000    -0.5000000    -0.2500000
    -0.2500000     0.0000000    -0.5000000
     0.2500000     0.2500000    -0.2500000
     0.5000000     0.2500000     0.5000000
    -0.2500000    -0.5000000     0.0000000
    -0.2500000     0.2500000     0.2500000
     0.5000000     0.5000000     0.2500000
     0.0000000    -0.2500000    -0.5000000
     0.2500000    -0.2500000     0.2500000
     0.2500000     0.5000000     0.5000000
    -0.5000000     0.0000000    -0.2500000
     0.0000000    -0.2500000     0.2500000
     0.2500000     0.0000000    -0.2500000
    -0.2500000    -0.2500000    -0.5000000
     0.2500000     0.5000000     0.2500000
     0.2500000    -0.2500000     0.0000000
    -0.5000000    -0.2500000    -0.2500000
     0.2500000     0.2500000     0.5000000
     0.0000000     0.2500000    -0.2500000
    -0.2500000    -0.5000000    -0.2500000
     0.5000000     0.2500000     0.2500000
    -0.2500000     0.0000000     0.2500000
     0.0000000     0.5000000     0.5000000
     0.5000000     0.0000000     0.5000000
     0.2500000    -0.2500000     0.5000000
     0.5000000     0.2500000    -0.2500000
    -0.5000000    -0.2500000    -0.7500000
     0.2500000    -0.5000000    -0.2500000
    -0.2500000     0.2500000    -0.5000000
end kpoints

```

=== Compute Wannier functions ===

run wannier90:

wannier90.x wannier90

This run generates the wannier90 standard output (wannier90.wout)
and the file wannier90.chk needed for the wannier interpolation (next step)

=== Obtain bandstructure (Wannier interpolation) and plot using XMGRACE or GNUPLOT===

Uncomment the bandstructure plot flags in wannier90.win and rerun (restart) wannier90:

wannier90.x wannier90

This run generates the following bandstructure files which can be visualized using xmgrace or gnuplot:

wannier90_band.agr

wannier90_band.dat

wannier90_band.gnu

*README.txt

```

Bandstructure plot in VASP (Three different ways)

1) Standard way: PBE (Fcc Si bandstructure example).
   1.1  Standard self-consistent (SC) run
   1.2  non-SC calculation (ICHARG=11) using preconverged CHGCAR file and KPOINTS_PBE_bands
   1.3  Plot using p4v

2) Fake SC procedure: PBE & HSE
   2.1 Standard self-consistent (SC) run
   2.2 Additional SC-run using KPOINTS_HSE_bands
   2.3 Plot using p4v

----
The file KPOINTS_HSE_bands is constructed by copying the IBZKPT file from run 2.1 to the KPOINTS file:

IBZKPT
Automatically generated mesh
       8
Reciprocal lattice
    0.00000000000000    0.00000000000000    0.00000000000000             1
    0.25000000000000    0.00000000000000    0.00000000000000             8
    0.50000000000000    0.00000000000000    0.00000000000000             4
    0.25000000000000    0.25000000000000    0.00000000000000             6
    0.50000000000000    0.25000000000000    0.00000000000000            24
   -0.25000000000000    0.25000000000000    0.00000000000000            12
    0.50000000000000    0.50000000000000    0.00000000000000             3
   -0.25000000000000    0.50000000000000    0.25000000000000             6

Then add the desired additional k-points with zero weight and change the total number of k-points

Explicit k-points list
      18 <--- CHANGE TOTAL NUMBER OF K-POINTS !!
Reciprocal lattice
    0.00000000000000    0.00000000000000    0.00000000000000             1
    0.25000000000000    0.00000000000000    0.00000000000000             8
    0.50000000000000    0.00000000000000    0.00000000000000             4
    0.25000000000000    0.25000000000000    0.00000000000000             6
    0.50000000000000    0.25000000000000    0.00000000000000            24
   -0.25000000000000    0.25000000000000    0.00000000000000            12
    0.50000000000000    0.50000000000000    0.00000000000000             3
   -0.25000000000000    0.50000000000000    0.25000000000000             6
0.00000000 0.00000000 0.00000000 0.000 <--- ZERO WEIGHT !!
0.00000000 0.05555556 0.05555556 0.000
0.00000000 0.11111111 0.11111111 0.000
0.00000000 0.16666667 0.16666667 0.000
0.00000000 0.22222222 0.22222222 0.000
0.00000000 0.27777778 0.27777778 0.000
0.00000000 0.33333333 0.33333333 0.000
0.00000000 0.38888889 0.38888889 0.000
0.00000000 0.44444444 0.44444444 0.000
0.00000000 0.50000000 0.50000000 0.000
----

3) VASP2WANNIER90: PBE, HSE & GW
   3.1 Standard SC run using the existing wannier.win file
   3.2 run wannier90 (wannier90.x wannier90) to generate MLWFs
   3.3 uncomment bandstructure plot flags in wannier90.win and restart wannier90

----
If the wannier90.win file does not exist VASP will create a default wannier90.win compatible with
the POSCAR and INCAR, which need to be suitably modify by including the proper instruction required
to generate the MLWFs (refer to the wannier90 manual):

default wannier90.win
 num_wann =     8  ! set to NBANDS by VASP

use_bloch_phases = .T.

begin unit_cell_cart
     2.7150000     2.7150000     0.0000000
     0.0000000     2.7150000     2.7150000
     2.7150000     0.0000000     2.7150000
end unit_cell_cart

begin atoms_cart
Si       0.0000000     0.0000000     0.0000000
Si       1.3575000     1.3575000     1.3575000
end atoms_cart

mp_grid =     4     4     4

begin kpoints
     0.0000000     0.0000000     0.0000000
     0.2500000     0.0000000     0.0000000
     0.5000000     0.0000000     0.0000000
     0.2500000     0.2500000     0.0000000
     0.5000000     0.2500000     0.0000000
    -0.2500000     0.2500000     0.0000000
     0.5000000     0.5000000     0.0000000
    -0.2500000     0.5000000     0.2500000
     0.0000000     0.2500000     0.0000000
     0.0000000     0.0000000     0.2500000
    -0.2500000    -0.2500000    -0.2500000
    -0.2500000     0.0000000     0.0000000
     0.0000000    -0.2500000     0.0000000
     0.0000000     0.0000000    -0.2500000
     0.2500000     0.2500000     0.2500000
     0.0000000     0.5000000     0.0000000
     0.0000000     0.0000000     0.5000000
    -0.5000000    -0.5000000    -0.5000000
     0.0000000     0.2500000     0.2500000
     0.2500000     0.0000000     0.2500000
    -0.2500000    -0.2500000     0.0000000
    -0.2500000     0.0000000    -0.2500000
     0.0000000    -0.2500000    -0.2500000
     0.0000000     0.5000000     0.2500000
     0.2500000     0.0000000     0.5000000
    -0.2500000    -0.2500000     0.2500000
    -0.5000000    -0.2500000    -0.5000000
     0.2500000     0.5000000     0.0000000
     0.2500000    -0.2500000    -0.2500000
    -0.5000000    -0.5000000    -0.2500000
     0.0000000     0.2500000     0.5000000
    -0.2500000     0.2500000    -0.2500000
    -0.2500000    -0.5000000    -0.5000000
     0.5000000     0.0000000     0.2500000
    -0.5000000    -0.2500000     0.0000000
     0.0000000    -0.5000000    -0.2500000
    -0.2500000     0.0000000    -0.5000000
     0.2500000     0.2500000    -0.2500000
     0.5000000     0.2500000     0.5000000
    -0.2500000    -0.5000000     0.0000000
    -0.2500000     0.2500000     0.2500000
     0.5000000     0.5000000     0.2500000
     0.0000000    -0.2500000    -0.5000000
     0.2500000    -0.2500000     0.2500000
     0.2500000     0.5000000     0.5000000
    -0.5000000     0.0000000    -0.2500000
     0.0000000    -0.2500000     0.2500000
     0.2500000     0.0000000    -0.2500000
    -0.2500000    -0.2500000    -0.5000000
     0.2500000     0.5000000     0.2500000
     0.2500000    -0.2500000     0.0000000
    -0.5000000    -0.2500000    -0.2500000
     0.2500000     0.2500000     0.5000000
     0.0000000     0.2500000    -0.2500000
    -0.2500000    -0.5000000    -0.2500000
     0.5000000     0.2500000     0.2500000
    -0.2500000     0.0000000     0.2500000
     0.0000000     0.5000000     0.5000000
     0.5000000     0.0000000     0.5000000
     0.2500000    -0.2500000     0.5000000
     0.5000000     0.2500000    -0.2500000
    -0.5000000    -0.2500000    -0.7500000
     0.2500000    -0.5000000    -0.2500000
    -0.2500000     0.2500000    -0.5000000
end kpoints
----

```

Wannier90 Manual: [http://www.wannier.org/doc/user_guide.pdf WANNIER90 manual]

LWANNIER90 in the VASP Manual: LWANNIER90.

== Download ==
 5_4_Si_bandstructure.tgz

{{Template:Hybrid_functionals}}

Category:Examples
