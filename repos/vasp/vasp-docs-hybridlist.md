# VASP · Hybridlist

(source: vasp-docs-hybridlist.html)

A certain number of unscreened and screened hybrid functionals are available in VASP, and furthermore if VASP is compiled with the library of exchange-correlation functionals Libxc, then most of the existing hybrid functionals can be used. Examples of INCAR files are shown below. Since VASP.6.4.0 it is possible to use hybrid functionals that mix meta-GGA and Hartree-Fock exchange. Note that it is in general recommended to use the PBE POTCAR files for hybrid functionals.

=== Range-separated hybrid functionals ===

*HSE06
 LHFCALC = .TRUE.
 GGA = PE
 HFSCREEN = 0.2

:with the default values AEXX=0.25, AGGAX=1-AEXX=0.75, AGGAC=1, and ALDAC=1.

*HSE03
 LHFCALC = .TRUE.
 GGA = PE
 HFSCREEN = 0.3

:with the default values AEXX=0.25, AGGAX=1-AEXX=0.75, AGGAC=1, and ALDAC=1.

*HSEsol
 LHFCALC = .TRUE.
 GGA = PS
 HFSCREEN = 0.2

:with the default values AEXX=0.25, AGGAX=1-AEXX=0.75, AGGAC=1, and ALDAC=1.

*Dielectric-dependent hybrid (DDH) RS-DDH
 LHFCALC = .TRUE.
 LMODELHF = .TRUE.
 AEXX = \varepsilon_{\infty}^{-1}
 BEXX = 0.25
 HFSCREEN = \mu
 GGA = PE

:where \varepsilon_{\infty}^{-1} is the inverse dielectric constant and \mu is the range-separation parameter. See a detailed description of the DDH functionals in the documentation for the LMODELHF tag as well as here.

*Dielectric-dependent hybrid (DDH) DD-RSH-CAM,DSH
 LHFCALC = .TRUE.
 LMODELHF = .TRUE.
 AEXX = \varepsilon_{\infty}^{-1}
 HFSCREEN = \mu
 GGA = PE

:with the default value BEXX=1 and where \varepsilon_{\infty}^{-1} is the inverse dielectric constant and \mu is the range-separation parameter. See a detailed description of the DDH functionals in the documentation for the LMODELHF tag as well as here.

*RSHXLDA
 LHFCALC = .TRUE.
 LRHFCALC = .TRUE.
 GGA = CA (or PZ)
 HFSCREEN = 0.75 # Optimal value for solids
 ALDAC = 1.0     # Necessary since correlation is by default not included when AEXX=1

:with the default value AEXX=1.

*RSHXPBE
 LHFCALC = .TRUE.
 LRHFCALC = .TRUE.
 GGA = PE
 HFSCREEN = 0.91 # Optimal value for the enthalpies of formation of molecules
 ALDAC = 1.0     # Necessary since correlation is by default not included when AEXX=1
 AGGAC = 1.0     # Necessary since correlation is by default not included when AEXX=1

:with the default values AEXX=1.

*sX-LDA
 LHFCALC = .TRUE.
 LTHOMAS = .TRUE.
 GGA = CA (or PZ)
 HFSCREEN = k_{\rm TF}
 ALDAC = 1.0     # Necessary since correlation is by default not included when AEXX=1
 AGGAC = 1.0     # Necessary since correlation is by default not included when AEXX=1

:with the default value AEXX=1 and where k_{\rm TF} is the Thomas-Fermi screening. More details can be found at LTHOMAS as well as here.

=== Unscreened hybrid functionals ===

*PBE0 (PBEh)
 LHFCALC = .TRUE.
 GGA = PE

:with the default values AEXX=0.25, AGGAX=1-AEXX=0.75, AGGAC=1, and ALDAC=1.

*B3LYP with VWN3 (or VWN5) for LDA correlation
 LHFCALC = .TRUE.
 GGA     = B3 (or B5)
 AEXX    = 0.2
 AGGAX   = 0.72
 AGGAC   = 0.81
 ALDAC   = 0.19

:with the default value ALDAX=1-AEXX=0.8.

*B3PW91 (using Libxc, see the tag LIBXC1)
 LHFCALC = .TRUE.
 GGA = LIBXC
 LIBXC1 = HYB_GGA_XC_B3PW91 # or 401
 AEXX = 0.2

*B1-WC (using Libxc, see the tag LIBXC1)
 LHFCALC = .TRUE.
 GGA = LIBXC
 LIBXC1 = HYB_GGA_XC_B1WC # or 412
 AEXX = 0.16

*SCAN0
 LHFCALC = .TRUE.
 METAGGA = SCAN

:with the default values AEXX=0.25, AMGGAX=1-AEXX=0.75, and AMGGAC=1.

*Hartree-Fock (no correlation)
 LHFCALC = .TRUE.
 AEXX    = 1

:with the default values AGGAX=1-AEXX=0, ALDAC=0, and AGGAC=0.

Note the default values when LHFCALC{{=.TRUE.:
*ALDAX, AGGAX and AMGGAX are set to 1-AEXX.
*ALDAC, AGGAC and AMGGAC are set to 0 if AEXX{{=}}1 or to 1 if AEXX\neq1.}}

== Related tags and articles ==
GGA,
METAGGA,
LIBXC1,
LIBXC2,
AEXX,
BEXX,
ALDAX,
ALDAC,
AGGAX,
AGGAC,
AMGGAX,
AMGGAC,
LHFCALC,
HFSCREEN,
LMODELHF,
LTHOMAS,
LRHFCALC,
Hybrid functionals: formalism

== References ==

----
Category:Exchange-correlation functionalsCategory:Hybrid_functionalsCategory:Howto
