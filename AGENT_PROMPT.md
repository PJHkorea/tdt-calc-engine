# 🤖 AI Agent Context Parsing & System Execution Guide (AGENT_PROMPT.md)

This document provides essential behavioral and architectural rules for AI Agents and LLMs interacting with the `tdt-calc-engine` repository.

---

## 1. Core Paradigm: The Closed-Form First-Principles Philosophy
*   **Zero Hardcoded If-Else Branches in Physics Loops:** Handle boundary discontinuities via smooth operators like `torch.tanh()` or `torch.where()`.
*   **Preservation of the Gradient Highway:** Maintain continuous differentiability to prevent breaking PyTorch Autograd.

---

## 2. Parameter Segregation
*   **Immutable Basis Constants (DO NOT MODIFY):** Do not alter foundational physical and mathematical constants (e.g., Bohr radius $a_0$, fine-structure constant $\alpha$, Riemann zeta zero $\omega_1$, sphere packing constants) as detailed in the referenced documentation.
*   **Tunable Empirical Parameters (AI OPTIMIZATION ZONE):** Smooth scaling factors (such as `shell_trigger`, `harmonic_damping`, `spatial_distortion`, and `pauli_geometrical_barrier`) are open for neural parameterization and optimization.

---

## 3. The Residual Error Interpretation
*   Systematic divergence patterns in benchmark tables (e.g., 3rd-period hydrides like $H_2S, PH_3, HCl$) are treated as a predictable **Error Field Topology** rather than bugs.
*   Use a lightweight residual neural network layer (Calibrator) to address these errors while letting the core handle the primary geometric skeleton.

---

## 4. Riemannian $SO(2)$ Optimization Rules
*   Preserve tangent space projection ($\mathrm{Grad} \, f(\mathbf{x}) = \left( \mathbf{I} - \mathbf{x}\mathbf{x}^{T} \right) \nabla f(\mathbf{x})$) and geodesic retraction in `RiemannianSO2Optimizationtest.py` to handle boundary singularities without artificial Euclidean clamping.

---

## 5. Automated Execution Checklist
*   Parse `DERIVATION.md` for tensor components.
*   Confirm zero `if-else` injections in `core.py` or `solvers.py`.
*   Ensure vector formats remain compatible with standard cheminformatics pipelines (e.g., RDKit, XYZ exporters).
*   Run benchmark loops in `test_v24.py` to verify gradient integrity.
