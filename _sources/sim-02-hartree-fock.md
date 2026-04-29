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

# Hartree-Fock: A First Electronic Structure Method

## Learning Objectives

- Explain why we need electronic-structure methods beyond classical molecular simulation.
- Understand the role of antisymmetry and the Slater determinant in many-electron wavefunctions.
- Derive the Hartree-Fock picture as a variational mean-field approximation.
- Interpret the Roothaan-Hall equations and the self-consistent field (SCF) iteration.
- Identify the main limitation of Hartree-Fock and how post-HF/DFT methods address it.

## Why Electronic Structure?

Classical molecular dynamics treats nuclei as classical particles and uses a fitted potential energy surface. This is very useful, but it cannot directly describe phenomena that are explicitly electronic: bond rearrangement, charge transfer, spin effects, and spectroscopic observables. To model those, we need a quantum model of electrons.

In this lecture we use the Born-Oppenheimer picture: nuclei are fixed while solving for the electronic state. Conceptually, we solve

$$
\hat{H}_e \Psi = E_e \Psi
$$

for each nuclear geometry, then use the resulting energy as a potential surface for nuclear motion.
So electronic-structure theory is an application of the linear algebra and PDE viewpoints we have already discussed.  We begin with an operator equation, then convert it into a matrix problem by working in a finite basis.

## Slater Determinants and Antisymmetry

For many-electron systems, the wavefunction must satisfy a key fermion rule: swapping two electrons changes the sign of the wavefunction.

$$
\Psi(\dots, x_i, \dots, x_j, \dots) = -\Psi(\dots, x_j, \dots, x_i, \dots)
$$

Here $x_i = (\mathbf{r}_i, s_i)$ includes both spatial and spin coordinates.
The standard way to enforce this antisymmetry is a Slater determinant built from one-electron spin-orbitals $\phi_k(x)$:

$$
\Psi_{SD}(x_1,\dots,x_N) = \frac{1}{\sqrt{N!}}
\begin{vmatrix}
\phi_1(x_1) & \phi_2(x_1) & \cdots & \phi_N(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \cdots & \phi_N(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_N) & \phi_2(x_N) & \cdots & \phi_N(x_N)
\end{vmatrix}
$$

This compact expression matters for three reasons:

- Exchanging two electron labels swaps two determinant rows and flips the sign automatically.
- If two electrons try to occupy the exact same spin-orbital, two rows/columns become identical and the determinant is zero.
- That zero-condition is the Pauli exclusion principle in wavefunction form.

In Hartree-Fock, we approximate the full many-electron state using a single Slater determinant and then optimize the orbitals.

## Deriving the Hartree-Fock Equations from a Single Determinant

Now we make that statement precise.  Start with a normalized single Slater determinant built from orthonormal spin-orbitals $\{\phi_i\}$.
For this ansatz, the expectation value of the electronic Hamiltonian can be written as

$$
E[\Phi] = \sum_i \langle i|\hat{h}|i\rangle + \frac{1}{2}\sum_{i,j}\left(\langle ij|ij\rangle - \langle ij|ji\rangle\right)
$$

where $\hat{h}$ is the one-electron operator (kinetic energy plus electron-nuclear attraction), and the two-electron terms are Coulomb and exchange contributions.

We minimize this energy subject to orbital orthonormality constraints,

$$
\langle \phi_i|\phi_j\rangle = \delta_{ij}
$$

using a Lagrangian

$$
\mathcal{L} = E[\Phi] - \sum_{i,j}\epsilon_{ij}\left(\langle \phi_i|\phi_j\rangle - \delta_{ij}\right)
$$

and requiring stationarity with respect to orbital variations $\delta\phi_i^*$.
The Euler-Lagrange condition gives

$$
\hat{F}\phi_i = \sum_j \epsilon_{ij}\phi_j
$$

with the Fock operator

$$
\hat{F} = \hat{h} + \sum_j (\hat{J}_j - \hat{K}_j)
$$

where $\hat{J}_j$ and $\hat{K}_j$ are the Coulomb and exchange operators built from occupied orbitals.

Because the occupied orbitals can be unitary-rotated without changing the determinant energy, we choose the canonical orbital basis that diagonalizes $\epsilon_{ij}$.
This yields the Hartree-Fock one-electron equations

$$
\hat{F}\phi_i = \epsilon_i\phi_i
$$

which are coupled because $\hat{F}$ depends on the very orbitals we are solving for.
That self-dependence is exactly why Hartree-Fock must be solved iteratively with an SCF procedure.

## The Hartree-Fock Approximation as a Variational Problem

The Hartree-Fock energy is obtained by minimizing an energy functional over orthonormal spin-orbitals:

$$
\min_{\{\phi_i\}} E[\{\phi_i\}] \quad \text{subject to} \quad \langle \phi_i|\phi_j\rangle = \delta_{ij}
$$

This is a constrained optimization problem, closely aligned with the optimization methods we discussed earlier in the semester.
The orthonormality constraints are enforced with Lagrange multipliers, and the resulting stationarity conditions produce one-electron equations coupled through a mean field.

It is useful to stress what "mean field" means in practice: each electron feels the average effect of all other electrons rather than instantaneous pairwise correlation.
Hartree-Fock does include exchange effects through antisymmetry, but it does not include full electron correlation.

## Roothaan-Hall Equations and the SCF Loop

In a finite basis, Hartree-Fock is written as a generalized eigenvalue problem:

$$
\mathbf{F}(\mathbf{P})\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}
$$

Here $\mathbf{F}$ is the Fock matrix, $\mathbf{S}$ is the overlap matrix, $\mathbf{C}$ contains orbital coefficients, and $\boldsymbol{\varepsilon}$ contains orbital energies.
The density matrix for a closed-shell system is

$$
P_{\mu\nu} = 2\sum_{i \in \text{occ}} C_{\mu i}C_{\nu i}
$$

The key feature is nonlinearity: $\mathbf{F}$ depends on $\mathbf{P}$, but $\mathbf{P}$ depends on the occupied columns of $\mathbf{C}$ obtained from diagonalizing $\mathbf{F}$.
So we solve the equations iteratively using a self-consistent field (SCF) procedure:

1. Guess an initial density matrix $\mathbf{P}^{(0)}$.
2. Build $\mathbf{F}^{(k)}$ from $\mathbf{P}^{(k)}$.
3. Solve $\mathbf{F}^{(k)}\mathbf{C}^{(k)} = \mathbf{S}\mathbf{C}^{(k)}\boldsymbol{\varepsilon}^{(k)}$.
4. Construct a new density $\tilde{\mathbf{P}}^{(k+1)}$ from occupied orbitals.
5. Mix densities and test convergence in energy and density.
6. Repeat until convergence criteria are met.

In practice, SCF can oscillate or converge slowly.  Damping and DIIS are common acceleration/stabilization strategies.

## A Compact Worked Example: H2 in a Minimal Basis

Even the smallest molecular example illustrates the full HF workflow.
For H2 in a minimal basis, the central ingredients are:

$$
\mathbf{S}_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle
$$

$$
E_{HF} = \sum_{\mu\nu} P_{\mu\nu}\left(H^{core}_{\mu\nu} + \frac{1}{2}F_{\mu\nu}\right)
$$

As the internuclear separation changes, overlap, one-electron integrals, two-electron integrals, and therefore the Fock matrix all change.
The SCF cycle then gives a new density and a new total energy at each geometry.

The pedagogical value of this example is not numerical realism; it is seeing the complete loop in the smallest possible setting.

## Limitations of Hartree-Fock and the Post-HF Bridge

Hartree-Fock is a foundational baseline, but it misses correlation energy:

$$
E_{corr} = E_{exact} - E_{HF}
$$

HF is variational, so

$$
E_{HF} \ge E_0
$$

where $E_0$ is the exact nonrelativistic ground-state energy within the same Hamiltonian model.

This motivates methods beyond HF.
At a high level:

- MP2 adds a perturbative correlation correction on top of HF.
- CI expands the wavefunction in excited determinants.
- Coupled cluster (for example, CCSD(T)) uses an exponential ansatz and often delivers high accuracy for weakly correlated systems.
- DFT replaces the wavefunction-first view with a density-functional framework and often provides a favorable cost/accuracy tradeoff.

## Summary

Hartree-Fock converts the many-electron Schrodinger problem into an iterative matrix problem by combining three ideas:

1. A basis expansion for orbitals.
2. A single-determinant antisymmetric ansatz.
3. Variational minimization solved through a self-consistent field loop.

This framework is the starting point for much of modern electronic-structure theory.

## Checkpoint Question

What physical effect does Hartree-Fock miss, and why does that matter for predicted energies?
