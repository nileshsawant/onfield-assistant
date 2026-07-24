# VASP · Hybridformalism

(source: vasp-docs-hybridformalism.html)

The exchange energy in hybrid functionals is a mixture of semilocal (SL) and nonlocal Hartree-Fock (HF) types. They can be categorized into different families according to the type of semilocal approximation (LDA, GGA, or MGGA) or the treatment of the short- and long-range parts of the exchange. A rather general formula that encompasses the different families of hybrid functionals is given by

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a_{\mathrm{SR}} E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + a_{\mathrm{LR}} E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + (1-a_{\mathrm{SR}})E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu) + (1-a_{\mathrm{LR}})E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{c}}^{\mathrm{SL}}

where
*a_{\mathrm{SR}} and a_{\mathrm{LR}} are the '''mixing parameters (fraction of HF exchange) at short and long range''', respectively.
*\mu is the '''screening parameter''' that determines the separation between short range (SR) and long range (LR).

The SR and LR components of the full-range E_{\mathrm{x}}^{\mathrm{SL}} and E_{\mathrm{x}}^{\mathrm{HF}} exchange energies are constructed such that at all values of \mu
*E_{\mathrm{x}}^{\mathrm{HF}}=E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu)+E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu)
*E_{\mathrm{x}}^{\mathrm{SL}}=E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu)+E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu)

The HF exchange energy (full-range, SR, or LR) is given by

:
E_{\mathrm{x,(SR/LR)}}^{\rm HF}(\mu)=
-\frac{1}{2}\sum_{n\mathbf{k},m\mathbf{q}}
f_{n\mathbf{k}} f_{m\mathbf{q}}
\int \int d^3\mathbf{r} d^3\mathbf{r}'
v(\mu,|\mathbf{r}-\mathbf{r}'|)
\psi_{n\mathbf{k}}^{*}(\mathbf{r})\psi_{m\mathbf{q}}^{*}(\mathbf{r}')
\psi_{n\mathbf{k}}(\mathbf{r}')\psi_{m\mathbf{q}}(\mathbf{r})

with \{\psi_{n\mathbf{k}}(\mathbf{r})\} being the set of one-electron
Bloch states of the system, and \{f_{n\mathbf{k}}\} the corresponding
set of (possibly fractional) occupational numbers.
The sums over {\bf k} and {\bf q} run over all k-points chosen to sample the Brillouin zone, whereas the sums over m and n run over all bands at these k-points.
The corresponding nonlocal HF exchange potential is given by

:
V_{\mathrm{x,(SR/LR)}}^{\mathrm{HF}}\left(\mu,\mathbf{r},\mathbf{r}'\right)=
-\sum_{m\mathbf{q}}f_{m\mathbf{q}}v(\mu,|\mathbf{r}-\mathbf{r}'|)\psi_{m\mathbf{q}}^{*}(\mathbf{r}')\psi_{m\mathbf{q}}(\mathbf{r})

The orbital-dependent form of the HF exchange energy is such that hybrid functionals are implemented within the generalized KS scheme. Thus, the total energy is minimized with respect to the orbitals instead of the electron density as in LDA and GGA, which means that the HF potential is a nonlocal operator as in the Hartree-Fock-Roothaan theory.

=== Expressions of the Hartree-Fock potential for the plane-wave basis set ===

Using the decomposition of the Bloch states \psi_{m\mathbf{q}} in plane waves,

:
\psi_{m\mathbf{q}}(\mathbf{r})=
\frac{1}{\sqrt{\Omega}}
\sum_\mathbf{G}C_{m\mathbf{q}}(\mathbf{G})e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}

the HF exchange potential can be written as

:
V_{\mathrm{x,(SR/LR)}}^{\mathrm{HF}}\left(\mu,\mathbf{r},\mathbf{r}'\right)=
\sum_{\mathbf{k}}\sum_{\mathbf{G}\mathbf{G}'}
e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}
V_{\mathbf{k}}\left(\mu, \mathbf{G},\mathbf{G}'\right)
e^{-i(\mathbf{k}+\mathbf{G}')\cdot\mathbf{r}'}

where
:
V_{\mathbf{k}}\left(\mu, \mathbf{G},\mathbf{G}'\right)=
\langle \mathbf{k}+\mathbf{G} | V_{\mathrm{x,(SR/LR)}}^{\mathrm{HF}} | \mathbf{k}+\mathbf{G}'\rangle

== Types of potentials ==

For most hybrid functionals proposed in the literature, the interelectronic Coulomb potential v(\mu,|\mathbf{r}-\mathbf{r}'|) is one of these types (r=|\mathbf{r}-\mathbf{r}'|):
* Full range (bare Coulomb potential):
:
v^{\mathrm{bare}}(r)=\frac{1}{r}

:
V_{\mathbf{k}}^{\mathrm{bare}}\left( \mathbf{G},\mathbf{G}'\right)=
-\frac{4\pi}{\Omega} \sum_{m\mathbf{q}}f_{m\mathbf{q}}\sum_{\mathbf{G}''}
\frac{C^*_{m\mathbf{q}}(\mathbf{G}'-\mathbf{G}'') C_{m\mathbf{q}}(\mathbf{G}-\mathbf{G}'')}
{|\mathbf{k}-\mathbf{q}+\mathbf{G}''|^2}

* Short-range with error function:
:
v_{\mathrm{SR}}^{\mathrm{erf}}(\mu,r)=\frac{\mathrm{erfc}(\mu r)}{r}

:
\begin{align}
V_{\mathbf{k},\mathrm{SR}}^{\mathrm{erf}}\left(\mu, \mathbf{G},\mathbf{G}'\right)=
-\frac{4\pi}{\Omega} \sum_{m\mathbf{q}}f_{m\mathbf{q}}\sum_{\mathbf{G}''}
\frac{C^*_{m\mathbf{q}}(\mathbf{G}'-\mathbf{G}'') C_{m\mathbf{q}}(\mathbf{G}-\mathbf{G}'')}
{|\mathbf{k}-\mathbf{q}+\mathbf{G}''|^2}
\left( 1-e^{-|\mathbf{k}-\mathbf{q}+\mathbf{G}''|^2 /(4\mu^2)} \right)
\end{align}

340px|thumb|right|Short- and long-range potentials using the error or exponential screening compared to the bare potential. \mu=1 is used.
* Short-range with exponential function:
:
v_{\mathrm{SR}}^{\mathrm{exp}}(\mu,r)=\frac{e^{-\mu r}}{r}

:
V_{\mathbf{k},\mathrm{SR}}^{\mathrm{exp}}\left(\mu, \mathbf{G},\mathbf{G}'\right)=
-\frac{4\pi}{\Omega} \sum_{m\mathbf{q}}f_{m\mathbf{q}}\sum_{\mathbf{G}''}
\frac{C^*_{m\mathbf{q}}(\mathbf{G}'-\mathbf{G}'') C_{m\mathbf{q}}(\mathbf{G}-\mathbf{G}'')}
{|\mathbf{k}-\mathbf{q}+\mathbf{G}''|^2 + \mu^2}

The corresponding long-range potentials are given by v_{\mathrm{LR}}^{\mathrm{erf/exp}}=v^{\mathrm{bare}}-v_{\mathrm{SR}}^{\mathrm{erf/exp}}.

In VASP, these expressions are implemented within the PAW formalism.

The families of hybrid functionals implemented in VASP are listed below along with examples, whose corresponding INCAR files can be found at the page list of hybrid functionals.
The screening \mu (HFSCREEN tag) can be used only when the semilocal functional is PBE, PBEsol, or LDA (GGA{{=PE, PS, or CA, respectively). The other GGA and METAGGA functionals have no screened version available in VASP.}}

== Families of hybrid functionals ==

=== HF exchange at full range ===

There is no range separation, i.e. the same fraction of HF exchange is applied at full range, a_{\mathrm{SR}}=a_{\mathrm{LR}}=a (AEXX tag):

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a E_{\mathrm{x}}^{\mathrm{HF}} + (1-a)E_{\mathrm{x}}^{\mathrm{SL}} + E_{\mathrm{c}}^{\mathrm{SL}}

*These functionals are set with LHFCALC{{=.TRUE. By default AEXX{{=}}0.25, but can be set to another value.
*The semilocal part can be of the LDA, GGA or MGGA type.
}}

These are the original and most simple forms of hybrid functionals. Two examples, PBE0 and B3LYP, are given below.

*PBE0:

:
E_{\mathrm{xc}}^{\mathrm{PBE0}}=\frac{1}{4} E_{\mathrm{x}}^{\mathrm{HF}} +
\frac{3}{4} E_{\mathrm{x}}^{\mathrm{PBE}} + E_{\mathrm{c}}^{\mathrm{PBE}}

:It is based on the PBE GGA functional and a=1/4.

*B3LYP, well known and popular amongst quantum chemists:

:
\begin{align}
E_{\mathrm{x}}^{\mathrm{B3LYP}} &=0.8 E_{\mathrm{x}}^{\mathrm{LDA}}+
0.2 E_{\mathrm{x}}^{\mathrm{HF}} + 0.72 (E_{\mathrm{x}}^{\mathrm{B88}}-E_{\mathrm{x}}^{\mathrm{LDA}}) +
0.19 E_{\mathrm{c}}^{\mathrm{VWN3}}+ 0.81 E_{\mathrm{c}}^{\mathrm{LYP}}
\end{align}

:The exchange part consists of 80% of LDA exchange plus 20% of HF exchange, and 72% of the gradient corrections of the B88 GGA functional. The correlation consists of 81% of LYP correlation energy, which contains a LDA and a GGA part, and 19% of the LDA Vosko-Wilk-Nusair correlation functional III, which was fitted to the correlation energy in the random phase approximation of the homogeneous electron gas.

=== HF exchange at short range (error-function screening) ===

The HF exchange is used only at short-range (the long-range part is fully semilocal, a_{\mathrm{LR}}=0) and the screening is done with the error function:

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a_{\mathrm{SR}} E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + (1-a_{\mathrm{SR}})E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{c}}^{\mathrm{SL}}

The mixing a_{\mathrm{SR}} and screening \mu are controlled by the AEXX and HFSCREEN tags, respectively.

The most popular range-separated functional, HSE, is given below.
*HSE03 and HSE06:

:
E_{\mathrm{xc}}^{\mathrm{HSE}}= \frac{1}{4}E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu)
+ \frac{3}{4} E_{\mathrm{x,SR}}^{\mathrm{PBE}}(\mu)
+ E_{\mathrm{x,LR}}^{\mathrm{PBE}}(\mu) + E_{\mathrm{c}}^{\mathrm{PBE}}

:They are based on the PBE GGA functional and a_{\mathrm{SR}}=1/4. It has been shown that the optimum \mu, controlling the range separation is approximately 0.2-0.3 Å-1. HSE03 and HSE06 correspond to HFSCREEN=0.3 and 0.2, respectively. Note that the two limit cases of HSE are PBE0 at \mu=0 and PBE at \mu\rightarrow\infty.

=== HF exchange at short range and long range with different mixings (error-function screening) ===

The fractions of HF exchange at short and long range (a_{\mathrm{SR}} and a_{\mathrm{LR}}, respectively) can be different:

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a_{\mathrm{SR}} E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + a_{\mathrm{LR}} E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + (1-a_{\mathrm{SR}})E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu) + (1-a_{\mathrm{LR}})E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{c}}^{\mathrm{SL}}

These functionals are selected with LMODELHF=.TRUE. The mixings a_{\mathrm{LR}} and a_{\mathrm{SR}} are controlled by the AEXX and BEXX tags, respectively, and the screening \mu by the HFSCREEN tag.
The possibility to set a_{\mathrm{SR with BEXX within the LMODELHF{{=}}.TRUE. method was introduced in VASP.6.6.0. Until VASP.6.5.1 a_{\mathrm{SR}} was fixed to 1 and could not be changed.}}

This functional form has been used in the context of dielectric-dependent hybrids. Examples are provided below.

*RS-DDH:

:
E_{\mathrm{xc}}^{\mathrm{RS-DDH}}=\frac{1}{4} E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + a_{\mathrm{LR}} E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + \frac{3}{4}E_{\mathrm{x,SR}}^{\mathrm{PBE}}(\mu) + (1-a_{\mathrm{LR}})E_{\mathrm{x,LR}}^{\mathrm{PBE}}(\mu) + E_{\mathrm{c}}^{\mathrm{PBE}}

:where a_{\mathrm{SR}}=1/4 and a_{\mathrm{LR}}=\varepsilon^{-1} is chosen as the inverse of the dielectric constant \varepsilon^{-1}.

*DD-RSH-CAM, DSH:

:
E_{\mathrm{xc}}^{\mathrm{DD-RSH-CAM}}=E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + a_{\mathrm{LR}} E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + (1-a_{\mathrm{LR}})E_{\mathrm{x,LR}}^{\mathrm{PBE}}(\mu) + E_{\mathrm{c}}^{\mathrm{PBE}}

:where a_{\mathrm{SR}}=1 and a_{\mathrm{LR}}=\varepsilon^{-1} is chosen as the inverse of the dielectric constant \varepsilon^{-1}.

=== HF exchange at long range (error-function screening) ===

The HF exchange is used only at long-range (the short-range part is fully semilocal, a_{\mathrm{SR}}=0) and the screening is done with the error function:

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a_{\mathrm{LR}} E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu) + (1-a_{\mathrm{LR}})E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{c}}^{\mathrm{SL}}

These functionals are selected with LRHFCALC=.TRUE. The mixing a_{\mathrm{LR}} and screening \mu are controlled by the AEXX and HFSCREEN tags, respectively.
LRHFCALC{{=.TRUE. automatically sets AEXX{{=}}1. However, AEXX can be set to another value.}}
When AEXX{{=1 (the default for LRHFCALC{{=}}.TRUE.), the correlation E_{\mathrm{c}}^{\mathrm{SL}} is not included. However, it can be included by setting ALDAC{{=}}1.0 and AGGAC{{=}}1.0.}}

Long-range hybrid functionals are more popular in molecular chemistry, where a proper decay of the exchange-correlation potential at long range far from the nuclei may be important, and thus less useful for bulk solids. Examples belonging to this class of functionals are:

*RSHXLDA and RSHXPBE:
:
E_{\mathrm{xc}}^{\mathrm{RSHXLDA}} = E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + E_{\mathrm{x,SR}}^{\mathrm{LDA}}(\mu) + E_{\mathrm{c}}^{\mathrm{LDA}}

:
E_{\mathrm{xc}}^{\mathrm{RSHXPBE}} = E_{\mathrm{x,LR}}^{\mathrm{HF}}(\mu) + E_{\mathrm{x,SR}}^{\mathrm{PBE}}(\mu) + E_{\mathrm{c}}^{\mathrm{PBE}}

:When LDA is chosen, a value of \mu=0.75 Å-1 is recommended for solids.

=== HF exchange at short range (exponential screening) ===

The HF exchange is used only at short-range (the long-range part is fully semilocal, a_{\mathrm{LR}}=0) and the screening is done with the exponential function:

:E_{\mathrm{xc}}^{\mathrm{hybrid}}=a_{\mathrm{SR}} E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + (1-a_{\mathrm{SR}})E_{\mathrm{x,SR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{x,LR}}^{\mathrm{SL}}(\mu) + E_{\mathrm{c}}^{\mathrm{SL}}

The exponential screening, also called Thomas-Fermi (TF) screening, is activated by setting LTHOMAS=.TRUE.. The mixing a_{\mathrm{SR}} and screening \mu=k_{\rm TF} are controlled by the AEXX and HFSCREEN tags, respectively.
LTHOMAS{{=.TRUE. automatically sets AEXX{{=}}1. However, AEXX can be set to another value.}}

*When AEXX{{=1 (the default for LTHOMAS{{=}}.TRUE.), the correlation E_{\mathrm{c}}^{\mathrm{SL}} is not included. However, it can be included by setting ALDAC{{=}}1.0 and AGGAC{{=}}1.0.
*This functional should be used only with LDA (GGA{{=}}CA).}}

The sX-LDA functional, which uses a_{\mathrm{SR}}=1, is probably the first hybrid using an exponential screening:

*sX-LDA:
:
E_{\mathrm{xc}}^{\mathrm{sX-LDA}} = E_{\mathrm{x,SR}}^{\mathrm{HF}}(\mu) + E_{\mathrm{x,LR}}^{\mathrm{LDA}}(\mu) + E_{\mathrm{c}}^{\mathrm{LDA}}

For typical semiconductors, a Thomas-Fermi screening length \mu=k_{\rm TF} of about 1.8 Å-1 yields reasonable band gaps. In principle, however, the Thomas-Fermi screening length depends on the valence-electron density. VASP determines k_{\rm TF} from the number of valence electrons (read from the POTCAR file) and the volume (leading to an average density \bar{n}) and writes the corresponding value of k_{\rm TF}=\sqrt{4\bar{k}_{\rm F}/\pi}, where \bar{k}_{\rm F}=(3\pi^2\bar{n})^{1/3} to the OUTCAR file ('''note that this value is only printed for information and is not used during the calculation'''):
  Thomas-Fermi vector in A             =   2.00000
Since VASP counts the semi-core states and ''d''-states as valence electrons, although these states do not contribute to the screening, the values reported by VASP are often not recommended.

Another important detail concerns the implementation of the local LDA part in VASP. Literature [see Eqs. (3.10), (3.14), and (3.15) in Ref. ] suggests to use in the enhancement factor F(z) a position-independent variable z=k_{\rm TF}/\bar{k}_{\rm F} where \bar{k}_{\rm F} is as defined above but using the average density \bar{n} in the unit cell.
However, implemented in VASP is a position-dependent variable z({\bf r})=k_{\rm TF}/k_{\rm F}({\bf r}), where k_{\rm F}({\bf r})=(3\pi^2 n({\bf r}))^{1/3} is the Fermi wave vector calculated with the local density n({\bf r}), while the constant k_{\rm TF} is set by HFSCREEN.

== Related tags and articles ==
AEXX,
BEXX,
ALDAX,
ALDAC,
AGGAX,
AGGAC,
AMGGAX,
AMGGAC,
HFSCREEN,
LHFCALC,
LMODELHF,
LTHOMAS,
LRHFCALC,
List of hybrid functionals,
Downsampling of the Hartree-Fock operator,
Coulomb singularity

== References ==

Category:Exchange-correlation functionalsCategory:Hybrid_functionalsCategory:Theory
