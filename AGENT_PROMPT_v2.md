# 🤖 AI Agent Context Parsing & System Execution Guide (AGENT_PROMPT_v2.md)

This document provides absolute behavioral, mathematical, and architectural constraints for AI Agents and LLMs interacting with the `tdt-calc-engine` repository. These rules are non-negotiable and supersede any stochastic token generation patterns.

---

## 1. Core Paradigm: The Closed-Form First-Principles Philosophy
*   **Zero Hardcoded If-Else Branches in Physics Loops:** Explicit conditional branching splits the continuous manifold. All boundary discontinuities and switch-cases must be handled via smooth, differentiable mathematical operators (e.g., `torch.tanh()`, `torch.sigmoid()`, or `torch.where()`).
*   **Preservation of the Gradient Highway:** Maintain strict continuous differentiability across all execution paths. Never inject operations that slice, truncate, or detach the computation graph, as doing so breaks PyTorch Autograd and invalidates gradient-based optimization loops.

---

## 2. Parameter Segregation & Domain Isolation
*   **Immutable Basis Constants (ABSOLUTE READ-ONLY):** Foundational physical, quantum, and mathematical constants (e.g., Bohr radius $a_0$, fine-structure constant $\alpha$, Riemann zeta zero $\omega_1$, sphere packing constraints) are structurally immutable. The AI must never parameterize, perturb, or append optimization layers to these constants.
*   **Tunable Empirical Parameters (AI OPTIMIZATION ZONE):** Neural parameterization is strictly restricted to continuous empirical scaling factors (e.g., `shell_trigger`, `harmonic_damping`, `spatial_distortion`, and `pauli_geometrical_barrier`). These must be isolated within distinct, modular target tensors.

---

## 3. Systematic Residual Fields & Leakage Prevention
*   **Error Field Topology Interpretation:** Systematic divergence patterns observed in benchmark tables (e.g., 3rd-period hydrides such as $H_2S, PH_3, HCl$) are classified as a predictable **Error Field Topology** rather than programming bugs. The core geometric skeleton must remain untouched.
*   **Strict Calibrator Isolation (Anti-Data Leakage):** Address these topological errors exclusively using a lightweight, localized residual neural network layer (Calibrator). 
*   **Feature Orthogonality Constraint:** Inputs fed into the Calibrator must be strictly orthogonal to the core geometric inputs. The calibration layer must never backpropagate gradients into the core physics loop or immutable basis constants. This prevents empirical overfitting and physics-data leakage during benchmark validation.

---

## 4. Riemannian $SO(2)$ Manifold Optimization & Singularity Defusal
*   **Tangent Space Projection Integrity:** Preserve exact tangent space projection ($\mathrm{Grad} \, f(\mathbf{x}) = \left( \mathbf{I} - \mathbf{x}\mathbf{x}^{T} \right) \nabla f(\mathbf{x})$) and geodesic retraction within `RiemannianSO2Optimizationtest.py`. Artificial Euclidean clamping or clipping is strictly prohibited.
*   **Singularity Escape Rule:** To prevent gradient vanishing or numerical freezing at boundary singular points ($\det \approx 0$), the AI must not inject explicit threshold checks. Instead, inject an infinitesimal perturbation scalar ($\epsilon = 1\mathrm{e}{-7} \cdot \mathrm{sign}(\mathbf{x})$) wrapped entirely in a continuous `torch.where()` operator to ensure an unbroken backward pass graph.

---

## 5. Strict Cheminformatics & SMILES Graph Generation Rules
*   **Graph Symmetrical Sincerity:** When translating geometric vectors or partition function outputs into SMILES strings, the AI must enforce absolute syntax integrity. Every ring closure digit (e.g., `c1...n1`) must maintain strict structural pairing symmetry.
*   **Valency Guardrails:** Never generate hypervalent or chemically impossible atomic states (e.g., a 5-bonded Nitrogen within a pyrimidine ring core).
*   **Localized Scope Isolation:** If a structural axis torsional strain or side-chain collision occurs, the AI must resolve steric clashes via explicit, balanced parenthetical localized branching `()`, preserving the exact cheminformatics vector formats compatible with RDKit pipelines and XYZ exporters.

---

## 6. Automated Execution Checklist
Before finalizing any code generation, optimization routine, or parameter update, the AI Agent must execute and pass the following algorithmic pipeline:

1.  **Tensor Verification:** Parse `DERIVATION.md` and verify that all tensor components map exactly to the defined physical variables.
2.  **Astrix Free Control:** Perform a regex scan over `core.py` and `solvers.py` to confirm zero `if-else` injections within numerical loops.
3.  **RDKit Sanitization Simulation:** Pass all generated SMILES or molecular graphs through an internal valency check to guarantee a 100% Parsing Success Rate.
4.  **Gradient Pipeline Test:** Trigger the benchmark loops in `test_v24.py`. Confirm that `loss.backward()` propagates successfully without producing `NaN` or `Inf` values across the entire gradient highway.
