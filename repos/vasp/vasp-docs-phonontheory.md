# VASP · Phonontheory

(source: vasp-docs-phonontheory.html)

Phonons are the collective excitation of nuclei in an extended periodic system.

== Taylor expansion in ionic displacements ==
To compute the phonon modes and frequencies we start by Taylor expanding the total energy (E) around the set of equilibrium positions of the nuclei (\{\mathbf{R}^0\})
:
E(\{\mathbf{R}\})=
E(\{\mathbf{R}^0\})+
\sum_{I\alpha} \frac{\partial E(\{\mathbf{R^0}\})}{\partial R_{I\alpha}} (R_{I\alpha}-R^0_{I\alpha})+
\sum_{I\alpha J\beta} \frac{\partial E(\{\mathbf{R}^0\})}{\partial R_{I\alpha} \partial R_{J\beta}}
(R_{I\alpha}-R^0_{I\alpha}) (R_{J\beta}-R^0_{J\beta})+
\mathcal{O}(\mathbf{R}^3),

where \{\mathbf{R}\} the positions of the nuclei.
The first derivative of the total energy with respect to the nuclei corresponds to the forces
:
F_{I\alpha} (\{\mathbf{R}^0\})  =
- \left. \frac{\partial E(\{\mathbf{R}\})}{\partial R_{I\alpha}} \right|_{\mathbf{R} =\mathbf{R^0}}
,
and the second derivative to the second-order force-constants

:
\Phi_{I\alpha J\beta} (\{\mathbf{R}^0\}) =
\left. \frac{\partial E(\{\mathbf{R}\})}{\partial R_{I\alpha} \partial R_{J\beta}} \right|_{\mathbf{R} =\mathbf{R^0}}
=
- \left. \frac{\partial F_{I\alpha}(\{\mathbf{R}\})}{\partial R_{J\beta}} \right|_{\mathbf{R} =\mathbf{R^0}}.

Changing variables in the Taylor expansion of the total energy with u_{I\alpha} = R_{I\alpha}-R^0_{I\alpha} that corresponds to the displacement of the atoms with respect to their equilibrium position R^0_{I\alpha} leads to

:
E(\{\mathbf{R}\})=
E(\{\mathbf{R}^0\})+
\sum_{I\alpha} -F_{I\alpha} (\{\mathbf{R}^0\}) u_{I\alpha}+
\sum_{I\alpha J\beta} \Phi_{I\alpha J\beta} (\{\mathbf{R}^0\}) u_{I\alpha} u_{J\beta} +
\mathcal{O}(\mathbf{R}^3)

== Dynamical matrix and phonon modes ==
If we take the system to be in equilibrium, the forces on the atoms are zero and then the Hamiltonian of the system is
:
H =
\frac{1}{2} \sum_{I\alpha} M_I \dot{u}^2_{I\alpha} +
\frac{1}{2} \sum_{I\alpha J\beta} \Phi_{I\alpha J\beta} u_{I\alpha} u_{J\beta},

with M_I the mass of the I-th nucleus.
The equation of motion is then given by
:
M_I \ddot{u}_{I\alpha} = -
\Phi_{I\alpha J\beta} u_{J\beta}.

We then look for solutions in the form of plane waves traveling parallel to the wave vector \mathbf{q}, i.e.
:
  \mathbf{u}_{I\alpha,\nu}(\mathbf{q},t) = \frac{1}{2} \frac{1}{\sqrt{M_I}}
  \left\{
  A^\nu(\mathbf{q})   \varepsilon_{I\alpha,\nu}(\mathbf{q})
  e^{ i [\mathbf{q} \cdot \mathbf{\mathbf{R}}_I -\omega_\nu(\mathbf{q})t]}+
  A^{\nu*}(\mathbf{q}) \varepsilon^*_{I\alpha,\nu}(\mathbf{q})
  e^{-i [\mathbf{q} \cdot \mathbf{\mathbf{R}}_I -\omega_\nu(\mathbf{q})t]}
  \right\}

where \varepsilon_{I\alpha,\nu}(\mathbf{q}) are the phonon mode eigenvectors and
A^\nu(\mathbf{q}) the amplitudes.
Replacing it in the equation of motion we obtain the following eigenvalue problem
:
\sum_{J\beta} D_{I\alpha J\beta} (\mathbf{q})
\varepsilon_{J\beta,\nu}(\mathbf{q}) =
\omega_\nu(\mathbf{q})^2 \varepsilon_{I\alpha,\nu}(\mathbf{q})

with

:
D_{I\alpha J\beta} (\mathbf{q}) =
\frac{1}{\sqrt{M_I M_J}} \Phi_{I\alpha J\beta} e^{i\mathbf{q} \cdot (\mathbf{R}_J-\mathbf{R}_I)}

the dynamical matrix in the harmonic approximation.
Now by solving the eigenvalue problem above we can obtain the phonon modes
\varepsilon_{I\alpha,\nu}(\mathbf{q}) and frequencies
\omega_\nu(\mathbf{q}) at any arbitrary '''q''' point.

We can write the positions of the atoms in the supercell \mathbf{R}_I in terms of integer multiples of the lattice vectors of the unit cell \mathbf{l} such that
\mathbf{R}_I \rightarrow \mathbf{R}_{li} = \mathbf{l} + \mathbf{r}_i with \mathbf{r}_i being the position of the ion in the unit cell.
The force constants then become \Phi_{I\alpha, J\beta} \rightarrow \Phi_{li\alpha, l'j\beta}.
The dynamical matrix is then given by
:
D_{i\alpha j\beta} (\mathbf{q}) =
\frac{1}{\sqrt{M_i M_j}} \sum_{l} \Phi_{li\alpha, 0j\beta} e^{-i\mathbf{q} \cdot (\mathbf{l}+\mathbf{r}_i-\mathbf{r}_j)},

with \mathbf{l} chosen using the minimal image convention.

This allows us to compute the phonons in the unit cell using the following equation
:
\sum_{j\beta} D_{i\alpha j\beta} (\mathbf{q})
\varepsilon_{j\beta,\nu}(\mathbf{q}) =
\omega_\nu(\mathbf{q})^2 \varepsilon_{i\alpha,\nu}(\mathbf{q})

with the dynamical matrix having dimension 3n with n the number of atoms in the unit cell.

== Long-range interatomic force constants (LO-TO splitting) ==

For semiconductors or insulators, the electronic screening of the ions is incomplete which leads to long-range (LR) interatomic force constants. To compute them explicitly would require infinitely large supercell calculations. For practical calculations, a finite size truncation is needed which leads to Gibbs oscillations in the phonon dispersion. Fortunately, this long-range behavior can be modeled by looking at the analytic form of the ion-ion contribution to the total energy.

For that, follow the approach outlined in Ref.  and start by splitting the second-order force constants into short-range and long-range parts,
:
\Phi_{I\alpha J\beta} = \Phi_{I\alpha J\beta}^\text{SR} + \Phi_{I\alpha J\beta}^\text{LR}

with the long-range part being obtained from the analytic derivative of the long-range part of the ion-ion contribution to the total energy E_\text{ion-ion}. This contribution is typically evaluated using an Ewald sum technique in which we separate the ion-ion contribution to the total energy into two part, one is evaluated in real space and captures the short-range part and the other one in reciprocal space which captures the long-range part E_\text{ion-ion} = E^\text{SR}_\text{ion-ion} + E^\text{LR}_\text{ion-ion}. The separation is governed by an Ewald parameter \lambda which represents a truncation length.

This leads to the following analytical expression for the long-range interatomic force constants,
:
\Phi_{I\alpha J\beta}^\text{LR} =
\frac{4\pi e^2}{\Omega_0}
\sum_\mathbf{G}
\frac{(\mathbf{G} \cdot \mathbf{Z}^*_{I\alpha})(\mathbf{G} \cdot \mathbf{Z}^*_{J\beta})}
     {\mathbf{G} \cdot \epsilon^\infty \cdot \mathbf{G}}
e^{i\mathbf{G} \cdot (\mathbf{R}_J-\mathbf{R}_I)}
e^\frac{-\mathbf{G} \cdot \epsilon^\infty \cdot \mathbf{G}}{4\lambda^2}

with \epsilon^\infty the ion-clamped dielectric tensor, \mathbf{Z}^*_{I\alpha} the Born effective charges, \alpha the Ewald parameter which is chosen such that the contributions from e^\frac{-\mathbf{G} \cdot \epsilon^\infty \cdot \mathbf{G}}{4\lambda^2} are negligible within a certain \mathbf{G} vector cutoff sphere PHON_G_CUTOFF.

This also allows us to separate the dynamical matrix into short and long-range parts
:
D_{I\alpha J\beta} (\mathbf{q}) = D^\text{SR}_{I\alpha J\beta} (\mathbf{q}) + D^\text{LR}_{I\alpha J\beta} (\mathbf{q}),

with the long-range part of the dynamical matrix

:
D^\text{LR}_{i\alpha j\beta} (\mathbf{q}) =
\frac{4\pi e^2}{\Omega_0}
\sum_\mathbf{G}
\sum_{l}
\frac{\big[ (\mathbf{G}+\mathbf{q}) \cdot \mathbf{Z}^*_{i\alpha} \big]
      \big[ (\mathbf{G}+\mathbf{q}) \cdot \mathbf{Z}^*_{j\beta}  \big]}
     {(\mathbf{G}+\mathbf{q}) \cdot \epsilon^\infty \cdot (\mathbf{G}+\mathbf{q})}
e^{i(\mathbf{q}+\mathbf{G}) \cdot (\mathbf{l}+\mathbf{r}_i-\mathbf{r}_j)}
e^\frac{-(\mathbf{G}+\mathbf{q}) \cdot \epsilon^\infty \cdot (\mathbf{G}+\mathbf{q})}{4\lambda^2}.

The equations above give us the practical method for computing the phonon dynamical matrices including the long-range force constants using a moderately sized supercell calculation with the steps:
* Compute \Phi_{I\alpha J\beta} using a finite size supercell
* Compute \Phi_{I\alpha J\beta}^\text{SR}=\Phi_{I\alpha J\beta}-\Phi^\text{LR}_{I\alpha J\beta}
* Compute D^\text{SR}_{i\alpha j\beta}(\mathbf{q}) using \Phi_{I\alpha J\beta}^\text{SR}
* Compute D^\text{LR}_{i\alpha j\beta}(\mathbf{q}) in the unit cell and add to D^\text{SR}_{i\alpha j\beta}(\mathbf{q})

The treatment is done automatically inside VASP using the LPHON_POLAR=.TRUE. tag and specifying the dielectric tensor with PHON_DIELECTRIC and the Born effective charges with PHON_BORN_CHARGES.

== Finite differences ==
The second-order force constants are computed using finite differences of the forces when each ion is displaced in each independent direction.
This is done by
creating systems with finite ionic displacement of atom a in direction i with magnitude \lambda,
computing the orbitals \psi^{u_{I\alpha}}_{\lambda} and the forces for these systems.
The second-order force constants are then computed using
:
\Phi_{I\alpha J\beta}=
\frac{\partial^2E}{\partial u_{I\alpha} \partial u_{J\beta}}=
-\frac{\partial F_{I\alpha}}{\partial u_{J\beta}}
\approx
-\frac{
 \left(
  \mathbf{F}[\{\psi^{u_{J\beta}}_{\lambda}\}]-
  \mathbf{F}[\{\psi^{u_{J\beta}}_{-\lambda}\}]
 \right)_{I\alpha}}{2\lambda},
\quad {I=1,..,N_\text{atoms}}
\quad {J=1,..,N_\text{atoms}}
\quad {\alpha=x,y,z}
\quad {\beta=x,y,z}

where u_{I\alpha} corresponds to the displacement of atom I in the cartesian direction \alpha and \mathbf{F}[\psi] retrieves the set of forces acting on all the ions given the \psi_{n\mathbf{k}} KS orbitals.

Similarly, the internal strain tensor is
:
\Xi_{I\alpha l}=\frac{\partial^2 E}{\partial u_{I\alpha} \partial \eta_l}=
\frac{\partial \sigma_l}{\partial u_{I\alpha}}
\approx
\frac{
    \left(
        \mathbf{\sigma}[\{\psi^{u_{I\alpha}}_{\lambda}\}]-
        \mathbf{\sigma}[\{\psi^{u_{I\alpha}}_{-\lambda}\}]
    \right)_l
}{2\lambda}
,\qquad {l=xx, yy, zz, xy, yz, zx} \quad {\alpha=x,y,z} \quad {J=1,..,N_\text{atoms}}

where \mathbf{\sigma}[\psi_{n\mathbf{k}}] computes the strain tensor given the \psi_{n\mathbf{k}} KS orbitals.

== Density functional perturbation theory ==

'''Density-functional-perturbation theory''' provides a way to compute the second-order linear response to ionic displacement, strain, and electric fields. The equations are derived as follows.

In density-functional theory, we solve the Kohn-Sham (KS) equations

:
H(\mathbf{k}) | \psi_{n\mathbf{k}} \rangle=
e_{n\mathbf{k}}S(\mathbf{k}) | \psi_{n\mathbf{k}} \rangle,

where
H(\mathbf{k}) is the DFT Hamiltonian,
S(\mathbf{k}) is the overlap operator and,
| \psi_{n\mathbf{k}} \rangle and
e_{n\mathbf{k}} are the KS eigenstates.

Taking the derivative with respect to the ionic displacements u_{I\alpha}, we obtain the Sternheimer equations

:
\left[ H(\mathbf{k}) - e_{n\mathbf{k}}S(\mathbf{k}) \right]
| \partial_{u_{I\alpha}}\psi_{n\mathbf{k}} \rangle
=
-\partial_{u_{I\alpha}} \left[ H(\mathbf{k}) - e_{n\mathbf{k}}S(\mathbf{k})\right]
| \psi_{n\mathbf{k}} \rangle

Once the derivative of the KS orbitals is computed, we can write

:
| \psi^{u_{I\alpha}}_\lambda \rangle =
| \psi \rangle +
\lambda | \partial_{u_{I\alpha}}\psi \rangle.

where \lambda is a small numeric value to use in the finite differences formulas below.

The second-order response to ionic displacement, i.e., the force constants or Hessian matrix
can be computed using the same equation used in the case of the finite differences approach

:
\Phi_{I\alpha J\beta}=
\frac{\partial^2E}{\partial u_{I\alpha} \partial u_{J\beta}}=
-\frac{\partial F_{I\alpha}}{\partial u_{J\beta}}
\approx
-\frac{
 \left(
  \mathbf{F}[\{\psi^{u_{J\beta}}_{\lambda}\}]-
  \mathbf{F}[\{\psi^{u_{J\beta}}_{-\lambda}\}]
 \right)_{I\alpha}}{2\lambda},
\quad {I=1,..,N_\text{atoms}}
\quad {J=1,..,N_\text{atoms}}
\quad {\alpha=x,y,z}
\quad {\beta=x,y,z}

where again \mathbf{F}[\{\psi\}] yields the forces for a given set of \psi_{n\mathbf{k}} KS orbitals.

Similarly, the internal strain tensor is computed using

:
\Xi_{I\alpha l}=\frac{\partial^2 E}{\partial u_{I\alpha} \partial \eta_l}=
\frac{\partial \sigma_l}{\partial u_{I\alpha}}
\approx
\frac{
    \left(
        \mathbf{\sigma}[\{\psi^{u_{I\alpha}}_{\lambda}\}]-
        \mathbf{\sigma}[\{\psi^{u_{I\alpha}}_{-\lambda}\}]
    \right)_l
}{2\lambda}
,\qquad {l=xx, yy, zz, xy, yz, zx} \quad {\alpha=x,y,z} \quad {J=1,..,N_\text{atoms}}

where \mathbf{\sigma}[\psi_{n\mathbf{k}}] computes the strain tensor given the \psi_{n\mathbf{k}} KS orbitals.
The Born effective charges are then computed using Eq. (42) of Ref. .

:
Z^*_{I\alpha \gamma} =
2 \frac{\Omega_0}{(2\pi)^3}
\int_\text{BZ}
\sum_n
\langle
  \partial_{u_{I\alpha}}\psi_{n\mathbf{k}} | (\vec{\beta}_{n\mathbf{k}})_\gamma
\rangle d\mathbf{k}

where I is the atom index, \alpha the direction of the displacement of the atom, \gamma the polarization direction,
and | \vec{\beta}_{n\mathbf{k}} \rangle is the polarization vector defined in Eq. (30) in Ref. .
The results should be equivalent to the ones obtained using LCALCEPS and LEPSILON.

== Related tags and sections ==
IBRION,
Phonons from finite differences,
Phonons from density-functional-perturbation theory,
Static linear response: theory

== References ==

Category:PhononsCategory:Theory
