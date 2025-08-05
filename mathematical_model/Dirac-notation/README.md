# Dirac Notation and the Density Operator in the SKA Framework

## Overview

This document defines the mathematical foundation of the Structured Knowledge Accumulation (SKA) framework using **Dirac notation**. It formalizes how the market state is represented as a quantum-inspired state vector, how regime probabilities and structural transitions are encoded in the **density operator**, and how this structure relates to entropy-based learning.

The goal is to provide a standalone mathematical reference that other SKA modules (e.g., real-time analytics, entropy flow, or QuestDB visualizations) can refer to.


## 1. State Space Definition
Let $\large \mathcal{H}$ be a **3-dimensional complex Hilbert space** with orthonormal basis: 

$$\large \mathcal{B} = \{ \lvert 0 \rangle, \lvert 1 \rangle, \lvert 2 \rangle \}$$  

**Interpretation**: Basis vectors represent mutually exclusive, exhaustive regimes.  



## 2. Market State Vector

At any time $\large t$, the market is in a superposition of basis regimes:

$$
\large{\lvert \Psi(t) \rangle = \Psi_0(t) \lvert 0 \rangle + \Psi_1(t) \lvert 1 \rangle + \Psi_2(t) \lvert 2 \rangle}
$$

* $\large \lvert 0 \rangle, \lvert 1 \rangle, \lvert 2 \rangle$: Basis vectors for Neutral, Bull, Bear
* $\large \Psi_i(t) = A_i(t) e^{iH_i(t)} \in \mathbb{C}$: Complex amplitude encoding probability and entropy

### Normalization:

For any state \(\lvert \Psi \rangle \in \mathcal{H}\), \(\langle \Psi \lvert \Psi \rangle = 1\).

$$
\large \langle \Psi(t) | \Psi(t) \rangle = |\Psi_0|^2 + |\Psi_1|^2 + |\Psi_2|^2 = 1
$$



## 3. Density Operator

The **density operator** is the outer product of the state vector with itself:

$$
\large{\widehat{\rho}(t) = \lvert \Psi(t) \rangle \langle \Psi(t) \rvert}
$$

This object captures both **probabilities** and **structural relationships** (coherences) between regimes.

### Matrix Elements:

$$
\large \langle i \mid \widehat{\rho}(t) \mid j \rangle = \Psi_i^*(t) \Psi_j(t)
$$

This yields a 3x3 Hermitian matrix:

$$\widehat{\rho}(t) = 
\begin{pmatrix}
|\Psi_0|^2 & \Psi_0^* \Psi_1 & \Psi_0^* \Psi_2 \\
\Psi_1^* \Psi_0 & |\Psi_1|^2 & \Psi_1^* \Psi_2 \\
\Psi_2^* \Psi_0 & \Psi_2^* \Psi_1 & |\Psi_2|^2
\end{pmatrix}$$




### Interpretation:

* **Diagonal terms**: $\large \rho_{ii} = |\Psi_i|^2$ = probability of being in regime $i$
* **Off-diagonal terms**: $\large \rho_{ij} = \Psi_i^* \Psi_j$ = coherence between regimes $i$ and $j$, interpreted as transition structure


## 4. Structural Transition Operator

A transition from regime $j \to i$ is represented by:

$$
\Large{\widehat{T}_{i \leftarrow j} = \lvert i \rangle \langle j \rvert}
$$

This operator transforms basis state $\lvert j \rangle$ into $\lvert i \rangle$.

### Expectation Value:

To extract the amplitude for a structural transition from $j \to i$:

$$
\large \langle \Psi \mid \widehat{T}_{i \leftarrow j} \mid \Psi \rangle = \Psi_i^*(t) \Psi_j(t)
$$

This equals the $(i,j)$ entry of the density matrix.



## 5. Entropy Interpretation

Each complex amplitude $\Psi_i(t)$ embeds an **entropy phase**:

$$
\large \Psi_i(t) = A_i(t) e^{i H_i(t)}
$$

This phase represents the **information content** (entropy) of regime $i$. Therefore, transitions (via $\large \Psi_i^* \Psi_j$) carry entropy shifts $\large H_j - H_i$.

## 6. Transition Operator Action on the State

We consider the global transition operator (density operator):

$$
\large \widehat{T} = \lvert \Psi \rangle \langle \Psi \rvert
$$

We project this operator acting on the state back onto the basis state $\lvert i \rangle$:

$$
\large \langle i \mid \widehat{T} \mid \Psi \rangle = \langle i \mid \Psi \rangle = \sum_j \Psi_j \langle i \mid j \rangle = \sum_j \Psi_j \delta_{ij} = \Psi_i
$$

Here we used the orthonormality of the basis: $\langle i \mid j \rangle = \delta_{ij}$, the Kronecker delta.

Since $\widehat{T} \mid \Psi \rangle = \lvert \Psi \rangle$, the operator acts as a projector and leaves the state invariant. Therefore:

$$
\begin{aligned}
\langle 0 \mid \widehat{T} \mid \Psi \rangle &= \Psi_0 \\
\langle 1 \mid \widehat{T} \mid \Psi \rangle &= \Psi_1 \\
\langle 2 \mid \widehat{T} \mid \Psi \rangle &= \Psi_2
\end{aligned}
$$

The transition operator encodes the full informational identity of the market state and reflects that transitions are emergent properties of the SKA state — not discrete external events.

## 7. Summary of Core Equations

| Concept               | Formula                                                            |
| --------------------- | ------------------------------------------------------------------ |
| State vector          | $\lvert \Psi \rangle = \sum_i \Psi_i \lvert i \rangle$             |
| Density operator      | $\widehat{\rho} = \lvert \Psi \rangle \langle \Psi \rvert$         |
| Regime probability    | $\rho\_{ii} = \|\Psi_i\|^2$                                        |
| Structural transition | $\widehat{T}_{i \leftarrow j} = \lvert i \rangle \langle j \rvert$ |
| Transition amplitude  | $\langle i \mid \widehat{\rho} \mid j \rangle = \Psi_i^* \Psi_j$   |
| Entropy phase         | $\arg(\Psi_i) = H_i(t)$                                            |



## 8. Usage

This document serves as the **formal reference** for SKA modules that:

* Visualize transitions in QuestDB or Grafana
* Compute entropy-induced transition probabilities
* Apply the principle of least entropy action

Refer to this file when defining:

* Structural probability matrices
* Real-time SKA signal decoders
* Quantum-inspired simulation of regime evolution

---

## Author Note

This formalism supports a deeper view of the market as an **information-processing system**, not a sequence of discrete events. Transitions are not externally triggered but emerge from the **geometry of knowledge accumulation** encoded in the SKA state.
