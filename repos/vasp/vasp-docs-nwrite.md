# VASP · Nwrite

(source: vasp-docs-nwrite.html)

0 {{! 1 {{!}} 2 {{!}} 3 {{!}} 4|2}}

Description: This tag determines how much will be written to the file OUTCAR ('verbosity tag').
----

The options for NWRITE are given in detail as

::{| cellpadding="5" cellspacing="5" style="width: 100%; border-spacing: 5px;"
| style="text-align:center; background-color:#DEC4EB;"| Feature || style="text-align:center; background-color:#DEC4EB;"| NWRITE = 0 || style="text-align:center; background-color:#DEC4EB;"| NWRITE = 1 || style="text-align:center; background-color:#DEC4EB;"| NWRITE = 2 || style="text-align:center; background-color:#DEC4EB;"| NWRITE = 3
|-
|style="background-color:#BBCCF5;"| Contributions to electronic energy at each electronic iteration || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| e || style="background-color:#D9F8F5;"| e
|-
|style="background-color:#BBCCF5;"| Convergence information || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| e || style="background-color:#D9F8F5;"| e
|-
|style="background-color:#BBCCF5;"| Eigenvalues || style="background-color:#D9F8F5;"| f+l || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| e
|-
|style="background-color:#BBCCF5;"| DOS + charge density || style="background-color:#D9F8F5;"| f+l || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| e
|-
|style="background-color:#BBCCF5;"| Total energy and electronic contributions || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i
|-
|style="background-color:#BBCCF5;"| Stress || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i
|-
|style="background-color:#BBCCF5;"| Basis vectors || style="background-color:#D9F8F5;"| f+l || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i
|-
|style="background-color:#BBCCF5;"| Forces || style="background-color:#D9F8F5;"| f+l || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i || style="background-color:#D9F8F5;"| i
|-
|style="background-color:#BBCCF5;"| Lattice and space group information for ISYM>0 || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| f || style="background-color:#D9F8F5;"| f
|-
|style="background-color:#BBCCF5;"| Symmetry operations for ISYM>0 || style="background-color:#D9F8F5;"|  || style="background-color:#D9F8F5;"|  || style="background-color:#D9F8F5;"|  || style="background-color:#D9F8F5;"| f
|-
|style="background-color:#BBCCF5;"| Timing information || style="background-color:#D9F8F5;"|  || style="background-color:#D9F8F5;"|  || style="background-color:#D9F8F5;"| X || style="background-color:#D9F8F5;"| X
|}

where the following abbreviations have been used

::{| cellpadding="5" cellspacing="5" style="width: 100%; border-spacing: 5px;"
| style="text-align:center; background-color:#DEC4EB;"| Code || style="text-align:center; background-color:#DEC4EB;"| Meaning
|-
|style="background-color:#BBCCF5;"| f+l || style="background-color:#D9F8F5;"| first and last ionic step
|-
|style="background-color:#BBCCF5;"| f || style="background-color:#D9F8F5;"| first ionic step
|-
|style="background-color:#BBCCF5;"| i || style="background-color:#D9F8F5;"| each ionic step
|-
|style="background-color:#BBCCF5;"| e || style="background-color:#D9F8F5;"| each electronic step
|-
|style="background-color:#BBCCF5;"| X || style="background-color:#D9F8F5;"| when applicable
|}

For long molecular-dynamics runs, use 0 or 1. For short runs use 2. 3 might give information if something goes wrong.
4 is for debugging only.
== Related tags and articles ==

OUTCAR, IALGO, IBRION, MDALGO, ISIF, ISYM, EDIFF, EDIFFG, Troubleshooting electronic convergence

{{sc|NWRITE|Examples|Examples that use this tag}}

Category:INCAR tagCategory:SymmetryCategory:ForcesCategory:Ionic minimizationCategory:Electronic minimizationCategory:Performance
