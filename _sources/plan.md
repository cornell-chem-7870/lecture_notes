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

# Density Functional Theory: From Many-Electron Wavefunctions to Practical Electronic Structure

## Learning Objectives

- Explain why density functional theory (DFT) is formulated in terms of electron density rather than the many-electron wavefunction.
- State the Hohenberg-Kohn theorems and describe their implications for ground-state electronic structure.
- Derive the Kohn-Sham equations as a practical mean-field-like reformulation of interacting electrons.
- Interpret the exchange-correlation functional and compare common approximation families (LDA, GGA, meta-GGA, hybrids).
- Describe the self-consistent field (SCF) loop in DFT and common numerical choices (basis sets, grids, pseudopotentials).
- Identify typical DFT strengths, common failure modes, and when to use beyond-DFT methods.

## 1. Why DFT?

- Motivation: Hartree-Fock and post-HF accuracy/cost tradeoff; need a practical method for larger systems.
- Computational scaling perspective and why wavefunction methods become expensive.
- Core idea: use electron density $n(\mathbf{r})$ as the central variable.
- Compare unknown dimensionality:
  - Wavefunction: depends on $3N$ spatial coordinates (plus spin).
  - Density: depends on 3 spatial coordinates.
- Transition statement: DFT replaces a difficult wavefunction problem with a density-functional minimization problem.

## 2. The Hohenberg-Kohn Theorems

### 2.1 First Hohenberg-Kohn Theorem

- Statement: the ground-state density uniquely determines the external potential (up to a constant).
- Consequence: all ground-state observables are functionals of the density.
- Discussion point: representability assumptions and what is meant by a valid density.

### 2.2 Second Hohenberg-Kohn Theorem

- Variational principle in density form:

$$
E_0 = \min_n \left\{ E[n] \right\}
$$

with $n(\mathbf{r}) \ge 0$ and $\int n(\mathbf{r}) d\mathbf{r} = N$.

- Emphasize: theorem is exact, but practical use depends on approximating unknown parts of $E[n]$.

## 3. Kohn-Sham Construction

### 3.1 Energy Decomposition

- Introduce Kohn-Sham partitioning:

$$
E[n] = T_s[n] + \int v_{ext}(\mathbf{r}) n(\mathbf{r}) d\mathbf{r} + E_H[n] + E_{xc}[n]
$$

- Define each term:
  - $T_s[n]$: noninteracting kinetic energy.
  - External potential energy.
  - Hartree (classical Coulomb) term.
  - Exchange-correlation term $E_{xc}[n]$ (all missing many-body physics).

### 3.2 Euler-Lagrange Equation and Effective Potential

- Constrained minimization with electron number constraint.
- Define effective potential:

$$
v_{eff}(\mathbf{r}) = v_{ext}(\mathbf{r}) + v_H(\mathbf{r}) + v_{xc}(\mathbf{r})
$$

where

$$
v_{xc}(\mathbf{r}) = \frac{\delta E_{xc}[n]}{\delta n(\mathbf{r})}
$$

### 3.3 Kohn-Sham Equations

$$
\left[-\frac{1}{2}\nabla^2 + v_{eff}(\mathbf{r})\right] \phi_i(\mathbf{r}) = \epsilon_i \phi_i(\mathbf{r})
$$

$$
n(\mathbf{r}) = \sum_{i \in occ} f_i |\phi_i(\mathbf{r})|^2
$$

- Explain analogy to Hartree-Fock SCF while stressing key conceptual difference (effective noninteracting system reproduces interacting density).

## 4. Exchange-Correlation Approximations

### 4.1 Why Approximations Are Needed

- Exact $E_{xc}[n]$ is unknown.
- Accuracy of DFT in practice is largely functional-dependent.

### 4.2 Jacob's Ladder Overview

- LDA: depends only on local density.
- GGA: adds density gradients.
- meta-GGA: includes additional semilocal ingredients (for example kinetic-energy density).
- Hybrid functionals: mix semilocal DFT exchange with exact (Hartree-Fock-like) exchange.

### 4.3 Practical Tradeoffs

- Cost vs robustness vs accuracy for molecules, solids, and reaction energetics.
- Brief mention of common issues:
  - Self-interaction error.
  - Delocalization error.
  - Missing long-range dispersion unless corrected.
  - Band-gap underestimation in many semilocal functionals.

## 5. The DFT Self-Consistent Field Loop

- Suggested algorithmic outline:

1. Choose initial density $n^{(0)}(\mathbf{r})$.
2. Build $v_{eff}^{(k)}(\mathbf{r})$ from current density.
3. Solve Kohn-Sham equations for orbitals/eigenvalues.
4. Construct new density $\tilde{n}^{(k+1)}(\mathbf{r})$.
5. Mix densities and test convergence in energy and density.
6. Repeat until converged.

- Discuss convergence aids: damping, Pulay/DIIS mixing, smearing for metals.

## 6. Numerical Ingredients in Real Calculations

### 6.1 Basis and Representation Choices

- Gaussian basis sets (common in quantum chemistry).
- Plane waves (common in periodic solid-state calculations).
- Real-space grids/wavelets (alternative discretizations).

### 6.2 Core Electrons and Pseudopotentials

- Rationale for frozen-core or pseudopotential approaches.
- Link to computational efficiency and transferability concerns.

### 6.3 Periodic Systems and k-Point Sampling

- Brief treatment of Brillouin-zone integration and k-point meshes.
- Explain why sampling quality affects energy/forces convergence.

## 7. What DFT Gets Right and Where It Fails

- Typical strengths:
  - Ground-state structures and trends.
  - Vibrational properties.
  - Reasonable total energies at moderate cost.
- Typical weaknesses:
  - Strongly correlated systems.
  - Charge-transfer excitations (in standard ground-state DFT framework).
  - Dispersion-dominated interactions without corrections.
- Bridge to next topics:
  - DFT+U, hybrid and range-separated functionals, RPA, GW, and TDDFT.

## 8. Suggested Worked Examples (for lecture or notebook)

- Example 1: Compare HF and DFT total energies for a small molecule (for example H2O) across basis set size.
- Example 2: Plot potential energy curve for H2 and compare LDA/GGA/hybrid trends.
- Example 3: Demonstrate SCF convergence behavior under different mixing parameters.
- Example 4: Show effect of dispersion correction on a weakly bound dimer.

## 9. Summary

- DFT is exact in principle via Hohenberg-Kohn, practical via Kohn-Sham, and useful because it balances cost and accuracy.
- The exchange-correlation functional is the central modeling choice.
- Numerical setup (basis, grids, pseudopotentials, convergence controls) matters as much as formal theory in real calculations.

## Checkpoint Questions

1. What key many-body effects are absorbed into $E_{xc}[n]$?
2. Why can two functionals give different predictions for the same system?
3. In what situations would you expect standard semilocal DFT to struggle?
