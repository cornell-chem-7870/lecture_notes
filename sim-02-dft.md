---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Density Functional Theory

## Learning Objectives

- Explain why density functional theory (DFT) uses electron density rather than the full many-electron wavefunction.
- State the Hohenberg-Kohn theorems and describe what they imply for ground-state electronic structure.
- Derive the Kohn-Sham equations and interpret the role of the exchange-correlation functional.
- Understand the self-consistent field (SCF) loop used to solve practical DFT calculations.
- Recognize common classes of exchange-correlation approximations and their tradeoffs.
- Identify typical strengths and known failure modes of standard DFT.

## The Problem: Electronic Structure Beyond Classical Potentials

In molecular simulation, we often model nuclei as classical particles and evaluate energies with an empirical force field.
This is computationally efficient and often accurate for near-equilibrium dynamics.
However, this approach is limited when electronic structure changes significantly, such as during bond breaking, charge transfer, or chemical reaction.

To model those processes from first principles, we need a quantum description of electrons.
For fixed nuclei (Born-Oppenheimer approximation), we solve an electronic Schrodinger equation of the form

$$
\hat{H}_e \Psi = E \Psi
$$

where $\Psi$ is the many-electron wavefunction.
The key difficulty is dimensionality: for $N$ electrons, $\Psi$ depends on $3N$ spatial coordinates (and spin labels).
This is one reason direct wavefunction methods become expensive quickly as system size grows.

Density functional theory addresses this by shifting focus from $\Psi$ to the electron density $n(\mathbf{r})$, which depends only on three spatial coordinates.

## The Hohenberg-Kohn Theorems

The formal foundation of DFT is given by two Hohenberg-Kohn (HK) theorems for the ground state.

### First Hohenberg-Kohn Theorem

The first theorem states that the ground-state density $n(\mathbf{r})$ uniquely determines the external potential $v_{ext}(\mathbf{r})$ (up to an additive constant).
Consequently, the Hamiltonian is determined, and therefore all ground-state observables are functionals of the density.

This is a remarkable statement:
instead of treating the wavefunction as the primary object, we can in principle recover all ground-state information from density alone.

To see why this is true, we use a proof by contradiction.
Assume two different external potentials, $v_{ext}(\mathbf{r})$ and $v_{ext}'(\mathbf{r})$, differ by more than a constant but produce the same ground-state density $n_0(\mathbf{r})$.
Let their Hamiltonians be $\hat{H}$ and $\hat{H}'$, with ground states $\Psi$ and $\Psi'$ and ground-state energies $E_0$ and $E_0'$.

Because $\Psi'$ is not the ground state of $\hat{H}$, the variational principle gives

$$
E_0 < \langle \Psi' | \hat{H} | \Psi' \rangle.
$$

Write $\hat{H} = \hat{H}' + (\hat{V}_{ext} - \hat{V}_{ext}')$:

$$
\begin{aligned}
E_0 &< \langle \Psi' | \hat{H}' | \Psi' \rangle + \langle \Psi' | \hat{V}_{ext} - \hat{V}_{ext}' | \Psi' \rangle \\
&= E_0' + \int n_0(\mathbf{r})\left[v_{ext}(\mathbf{r}) - v_{ext}'(\mathbf{r})\right] d\mathbf{r}.
\end{aligned}
$$

Similarly, because $\Psi$ is not the ground state of $\hat{H}'$,

$$
\begin{aligned}
E_0' &< \langle \Psi | \hat{H}' | \Psi \rangle \\
&= E_0 + \int n_0(\mathbf{r})\left[v_{ext}'(\mathbf{r}) - v_{ext}(\mathbf{r})\right] d\mathbf{r}.
\end{aligned}
$$

Adding the two strict inequalities gives

$$
E_0 + E_0' < E_0 + E_0',
$$

which is impossible.
Therefore, two different external potentials (up to more than an additive constant) cannot produce the same ground-state density.
So there is a one-to-one mapping between $n_0(\mathbf{r})$ and $v_{ext}(\mathbf{r})$ (up to a constant).

### Second Hohenberg-Kohn Theorem

The second theorem gives a variational principle in terms of density.
There exists an energy functional $E[n]$ such that

$$
E_0 = \min_n E[n]
$$

subject to the constraints

$$
n(\mathbf{r}) \ge 0, \qquad \int n(\mathbf{r}) d\mathbf{r} = N.
$$

So if we had the exact functional $E[n]$, ground-state calculations would reduce to constrained minimization over admissible densities.
The catch is that part of this exact functional is unknown in closed form.

## Kohn-Sham Reformulation

The Kohn-Sham approach provides a practical route to use the HK theorems.
Instead of minimizing a fully unknown functional directly, we map the interacting electron problem to an auxiliary noninteracting system that reproduces the same ground-state density.

### Kohn-Sham Orbitals

The central unknowns in the Kohn-Sham construction are one-electron orbitals $\phi_i(\mathbf{r})$.
These are not the true many-electron wavefunction; they are auxiliary orbitals of a noninteracting reference system chosen so that they reproduce the exact ground-state density (in exact DFT).

They satisfy orthonormality:

$$
\int \phi_i^*(\mathbf{r})\phi_j(\mathbf{r})\,d\mathbf{r} = \delta_{ij}.
$$

Given orbital occupations $f_i$, the density is

$$
n(\mathbf{r}) = \sum_i f_i |\phi_i(\mathbf{r})|^2.
$$

For a closed-shell molecule in a spin-restricted picture, occupied spatial orbitals usually have $f_i=2$ and unoccupied orbitals have $f_i=0$.
In metallic systems or finite-temperature smearing schemes, fractional occupations can occur.

Conceptually, each $|\phi_i(\mathbf{r})|^2$ contributes a piece of the total electron density, while the full many-body effects are represented through the effective potential and the exchange-correlation functional.

### Energy Decomposition

Kohn-Sham DFT writes the energy as

$$
E[n] = T_s[n] + \int v_{ext}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r} + E_H[n] + E_{xc}[n]
$$

where:

- $T_s[n]$ is the kinetic energy of a noninteracting reference system.
- $\int v_{ext} n$ is the electron-nuclear interaction term.
- $E_H[n]$ is the classical Hartree electron-electron repulsion.
- $E_{xc}[n]$ is the exchange-correlation functional.

In terms of the Kohn-Sham orbitals, the noninteracting kinetic energy is

$$
T_s[n] = -\frac{1}{2}\sum_i f_i \int \phi_i^*(\mathbf{r})\nabla^2\phi_i(\mathbf{r})\,d\mathbf{r}.
$$

The classical electron-electron repulsion (Hartree term) is

$$
E_H[n] = \frac{1}{2}\iint \frac{n(\mathbf{r})n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,d\mathbf{r}\,d\mathbf{r}'.
$$

For comparison, the exact interacting electron-electron repulsion is

$$
V_{ee}[\Psi] = \frac{1}{2}\sum_{i\neq j}\left\langle \Psi \middle| \frac{1}{|\mathbf{r}_i-\mathbf{r}_j|} \middle| \Psi \right\rangle,
$$

so $E_{xc}[n]$ contains the difference between this many-body interaction and the Hartree approximation, along with kinetic corrections beyond $T_s[n]$.

The exchange-correlation term collects everything missing from the first three contributions, including exchange and correlation effects and the correction from true interacting kinetic energy to $T_s$.

### Effective Potential and Kohn-Sham Equations

Minimizing the energy under the particle-number constraint leads to one-electron equations:

$$
\left[-\frac{1}{2}\nabla^2 + v_{eff}(\mathbf{r})\right]\phi_i(\mathbf{r}) = \epsilon_i\phi_i(\mathbf{r})
$$

with

$$
v_{eff}(\mathbf{r}) = v_{ext}(\mathbf{r}) + v_H(\mathbf{r}) + v_{xc}(\mathbf{r}),
$$

and

$$
v_{xc}(\mathbf{r}) = \frac{\delta E_{xc}[n]}{\delta n(\mathbf{r})}.
$$

The density is reconstructed from occupied Kohn-Sham orbitals:

$$
n(\mathbf{r}) = \sum_{i \in occ} f_i |\phi_i(\mathbf{r})|^2
$$

where $f_i$ are orbital occupation numbers.

At this point we have a closed-looking system, but it is nonlinear because $v_{eff}$ depends on $n$, while $n$ depends on orbitals obtained from $v_{eff}$.
So we solve it iteratively.

## The SCF Loop in DFT

A practical DFT calculation follows a self-consistent field loop:

1. Start from an initial guess density $n^{(0)}(\mathbf{r})$.
2. Build $v_{eff}^{(k)}(\mathbf{r})$ from the current density.
3. Solve the Kohn-Sham equations for $\phi_i^{(k)}$ and $\epsilon_i^{(k)}$.
4. Build an updated density $\tilde{n}^{(k+1)}(\mathbf{r})$.
5. Mix old and new densities, then test convergence.
6. Repeat until changes in density and energy are below tolerance.

Convergence can be slow or oscillatory.
Common stabilization strategies include density mixing, damping, and DIIS/Pulay acceleration.
For metallic systems, occupation smearing is often important.

## Exchange-Correlation Functionals: The Central Approximation

The exact $E_{xc}[n]$ is unknown, so practical DFT depends on approximation families.
A common hierarchy is:

- LDA (local density approximation): depends only on local density $n(\mathbf{r})$.
- GGA (generalized gradient approximation): uses $n$ and $\nabla n$.
- meta-GGA: includes additional semilocal ingredients such as kinetic-energy density.
- Hybrids: mix semilocal exchange with a fraction of exact Hartree-Fock exchange.

No single functional is universally best.
In practice, one chooses based on target properties and system type, then validates against reference data where possible.

```{note}
In real research workflows, functional choice and numerical settings (basis, cutoff, k-point mesh, pseudopotential) should be convergence-tested together.
A good functional with unconverged numerics still gives unreliable results.
```

## Numerical Ingredients in Real Calculations

Beyond formal equations, implementation details strongly affect quality and cost.

### Basis and Representation

Three common choices are:

- Gaussian basis sets (frequent in molecular quantum chemistry).
- Plane-wave basis sets (frequent for periodic solids).
- Real-space grids/wavelets (alternative discretizations with different scaling and boundary behavior).

### Pseudopotentials and Core Electrons

Treating all electrons explicitly is expensive, especially for heavier atoms.
Pseudopotentials or frozen-core approximations reduce cost by replacing chemically inert core electrons with an effective potential.
This usually gives major speedups, but transferability must be checked.

### Periodic Systems and k-Point Sampling

For periodic crystals, quantities are averaged over the Brillouin zone.
So total energies and forces depend on k-point sampling quality.
Insufficient sampling can lead to significant errors, especially for metals.

## What DFT Gets Right and Where It Fails

Standard DFT often performs well for:

- Ground-state geometries and structural trends.
- Vibrational frequencies (with caveats).
- Relative energies at moderate computational cost.

Common difficulties include:

- Strongly correlated electronic states.
- Delocalization and self-interaction errors.
- Weak van der Waals interactions unless a dispersion correction is included.
- Underestimated band gaps with many semilocal functionals.

These limitations motivate extensions such as DFT+U, hybrid and range-separated functionals, many-body perturbation methods (for example GW), and time-dependent DFT for excitations.

## A Compact Example Workflow

Suppose we want the equilibrium bond length of a diatomic molecule.
A standard DFT workflow is:

1. Choose functional, basis/pseudopotential, and convergence thresholds.
2. Compute total energies on a grid of bond lengths.
3. Ensure each geometry is SCF-converged and numerical settings are converged.
4. Fit the resulting energy curve and locate its minimum.
5. Compare with experiment or a higher-level reference method.

This process illustrates a core point: DFT is not only a theory but also a numerical procedure, and both aspects must be controlled.

## Summary

Density functional theory provides a practical first-principles framework for electronic structure by expressing the ground-state problem in terms of density rather than the full many-electron wavefunction.
The Hohenberg-Kohn theorems guarantee exactness in principle, while the Kohn-Sham construction makes calculations tractable in practice.
The exchange-correlation functional is the central approximation, and careful numerical convergence is essential for reliable predictions.