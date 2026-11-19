# 🤖 AI Agent Context Parsing & System Execution Guide (AGENT_PROMPT_v2.md)

This document provides absolute behavioral, mathematical, and architectural constraints for AI Agents and LLMs interacting with the `tdt-calc-engine` repository.

---

## 1. Core Paradigm & Soft Gates
*   **Zero Hardcoded If-Else Branches:** Use vectorized, differentiable operators (`torch.tanh()`, `torch.sigmoid()`, `torch.where()`) instead of explicit conditional branches in physics loops.
*   **Explicit Parenthetical Grouping:** Wrap each soft gate term (e.g., `(0.5 + 0.5 * torch.tanh(...))`) in explicit parentheses to prevent operator-precedence issues.
*   **Gradient Highway:** Maintain continuous differentiability without slicing, truncating, or detaching the computation graph.

---

## 2. Parameter Segregation
*   **Immutable Basis Constants:** Physical and mathematical constants (e.g., $a_0$, $\alpha$, $\omega_1$) are strictly read-only and must never be parameterized or optimized.
*   **Tunable Empirical Parameters:** Neural parameterization is restricted to continuous scaling factors (`shell_trigger`, `harmonic_damping`, `spatial_distortion`, `pauli_geometrical_barrier`) within modular target tensors.

---

## 3. Residual Fields & Leakage Prevention
*   **Error Field Topology:** Classify benchmark divergences (e.g., 3rd-period hydrides like $H_2S, PH_3, HCl$) as predictable topological errors rather than code bugs.
*   **Strict Calibrator Isolation:** Use a lightweight residual neural network layer (Calibrator) for topological errors. **Training inputs must be dynamically generated from actual physics solver outputs (`raw_R`, `raw_angle`) rather than hardcoded ground-truth values.**
*   **Feature Orthogonality:** Keep Calibrator inputs orthogonal to core geometric inputs and prevent gradients from backpropagating into the core physics loop.

---

## 4. Riemannian $SO(2)$ Manifold & Pauli Barrier
*   **Tangent Space Projection:** Preserve tangent space projections and geodesic retractions within **`tdt_engine/solvers.py` and `prototypes/RiemannianSO2Optimizationtest.py`** without artificial Euclidean clamping.
*   **Half-Angle Pauli Barrier:** **Formulate Pauli geometrical repulsion using $\sin^2(\theta / 2)$ instead of $\sin^2(\theta)$ to suppress steric overlap as $\theta \to 0^\circ$ without artificial energy barriers near $180^\circ$.**
*   **Singularity Defusal:** Wrap infinitesimal perturbation scalars ($\epsilon = 1\mathrm{e}{-7} \cdot \mathrm{sign}(\mathbf{x})$) inside continuous `torch.where()` operators.

---

## 5. Cheminformatics & SMILES Rules
*   **Graph Symmetrical Sincerity:** Enforce strict structural pairing symmetry for ring closure digits in SMILES strings.
*   **Valency Guardrails:** Prevent hypervalent or chemically impossible atomic states (e.g., 5-bonded Nitrogen).
*   **Localized Scope Isolation:** Resolve steric clashes via balanced parenthetical localized branching `()` compatible with RDKit and XYZ pipelines.

---

## 6. Automated Execution Checklist
1.  **Tensor Verification:** Map tensor components to physical variables via `DERIVATION.md`.
2.  **Astrix Free Control:** Regex scan `tdt_engine/core.py`, **`tdt_engine/solvers.py`, and `tdt_engine/calibrator_nn.py`** for zero `if-else` injections and verified batched tensor support.
3.  **RDKit Sanitization Simulation:** Ensure 100% parsing success via valency checks on generated molecular graphs.
4.  **Gradient & Residual Pipeline Test:** Execute benchmark loops in `test_v24.py` and **`tdt_engine/train_calibrator.py`**, confirming `loss.backward()` produces no `NaN` or `Inf` values.
