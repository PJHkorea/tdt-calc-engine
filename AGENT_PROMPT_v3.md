# 🤖 AI Agent Context Parsing & System Execution Guide (AGENT_PROMPT_v3.md)

This document provides absolute behavioral, mathematical, and architectural constraints for AI Agents and LLMs interacting with the **v3 Hybrid Differentiable Architecture** (`DERIVATION_v2.md` compliant) in the `tdt-calc-engine` repository.

---

## 1. Core Paradigm & Branch-Free Soft Gates
*   **Zero Hardcoded If-Else Branches:** Strictly prohibit explicit conditional branching (`if-else`) in physics and calibration tensor loops. Use smooth, vectorized operators (`torch.tanh()`, `torch.sigmoid()`, `F.softplus()`, `torch.acos()`, `torch.logit()`) across Periods 1–6 ($k \in \{2, 10, 18, 36, 54\}$).
*   **Explicit Parenthetical Grouping:** Wrap every soft gate expression (e.g., `(0.5 + 0.5 * torch.tanh(...))`) in explicit parentheses to prevent operator-precedence errors.
*   **Unbroken Gradient Highway:** Maintain end-to-end `torch.float64` differentiability from input atomic tensors to final calibrated outputs without calling `.detach()` inside differentiable solver cores (`_compute_bond_length_tensor`, `_compute_angle_tensor`).

---

## 2. Parameter Segregation & Read-Only Buffers
*   **Immutable Physical Basis & Tables:** Fundamental constants ($a_0 = 0.5291772109 \text{ \AA}$, $\omega_1 = 14.1347251417$, $\alpha = 0.0072973525$) and standardized atomic lookup tables ($\mathbf{T}\_{\text{cov}}, \mathbf{T}\_{\chi}$ for $Z \in [1, 86]$) must be registered as non-trainable buffers (`register_buffer` / `writeable = False`) and never updated by optimizers.
*   **Closed-Form Bond Length Baseline:** Bond lengths must be computed via single-pass closed-form algebraic contraction (Schomaker–Stevenson + topological asymmetry $\chi_{\mu\nu}$), eliminating legacy 200-step root-finding loops.

---

## 3. Pure Residual Calibration & Zero Lookup-Gate Policy
*   **Prohibition of Molecular Answer-Key Gates:** Legacy hardcoded restoration masks (e.g., $G_{\text{tetra}}, G_{\text{linear}}, \theta_{\text{restoration}}$ forcing exact values for $\text{CH}_4, \text{CO}_2, \text{H}_2\text{O}$) are **strictly forbidden**. All calibration must rely purely on generalizable physics-enriched residual networks (`calibrator_nn_v3.py`).
*   **Physics-Enriched Feature Vectors:**
    *   **Bond Length (5D):** $[\nu_{\text{min}}, \nu_{\text{max}}, \nu_{\text{bond}}, R_{\text{raw}}, \Delta_{\rho}]^T$
    *   **Bond Angle (8D):** $[\nu_{\text{center}}, \nu_{\text{smin}}, \nu_{\text{smax}}, \nu_{b1}, \nu_{b2}, \rho_C, \rho_{\text{LP}}, \theta_{\text{raw,rad}}]^T$
*   **Radian-Unified Logit-Sigmoid Projection:** Operate natively in radians ( $\theta \in (0, \pi)$ ) with boundary stabilization ( $\epsilon\_{\text{logit}} = 10^{-6}$ ) and commensurable loss scaling ( $\mathcal{E}\_{\text{total,v2}} = \mathcal{L}\_{\text{length}} + \lambda_{\text{rad}} \mathcal{L}\_{\text{angle,rad}},\, \lambda_{\text{rad}} \approx 1.0$ ).

---

## 4. Batched Riemannian $SO(2)$ Manifold & UFF-VSEPR Energy
*   **Batched Tangent Space Projection:** Execute angular optimization simultaneously across $[B \times 2]$ product manifolds $(\mathbb{S}^1)^B$ using orthogonal tangent projections $(\mathbf{I}\_2 - \mathbf{x}\_i\mathbf{x}\_i^T)\nabla f(\mathbf{x}\_i)$ and numerically stabilized geodesic retractions ( $\|\mathbf{d}\_{i,k}\|_2 + 10^{-12}$ ).
*   **UFF-VSEPR Hybrid Potential:** Combine Steric Number ( $\mathcal{S}\_{\text{eff}} = N_{\text{lig}} + \rho_{\text{LP}}$ ) continuous angle transitions with Universal Force Field (UFF) cosine harmonic bending and damped Yukawa–Pauli short-range barriers ( $\exp(-\theta/0.4) / (\sin^2(\theta/2) + 10^{-6})$ ).
*   **Branch-Free Geodesic Folding:** Recover valence angles in $[0, \pi]$ strictly via $\arccos(\mathrm{clamp}(x_{i,1}, -1 + 10^{-7}, 1 - 10^{-7}))$.

---

## 5. LLM Screening Guardrails & Trust Region Protocol
*   **Mandatory Tool-Execution Rule (Zero Numerical Hallucination):** When performing molecular screening sessions, **LLMs must never fabricate or role-play bond lengths, angles, or simulation logs in plain text.** All geometric values must be obtained by executing `TDTManifoldSolver` and `TDTNeuralCalibrator` via Python tool calls.
*   **Out-of-Sample (OOS) Trust Region Flagging:** Based on `STAGE 5` generalization benchmarks, agents must attach confidence flags to screened candidates:
    *   `[HIGH_CONFIDENCE]`: Standard C/N/O/S/Halogen single/multiple bonds and Period-2/3 valence-satisfied centers (expected length error $< 0.08 \text{ \AA}$, angle error $< 12^\circ$).
    *   `[EXTRAPOLATION_WARNING]`: Electron-deficient / octet-exception centers (e.g., Boron $\text{BF}\_3$), hypervalent centers ( $N_{\text{lig}} \ge 5$ ), or untrained multi-fluorinated centers ( $\text{NF}_3, \text{OF}_2$ ). Flag these candidates for mandatory downstream `RDKit` / DFT verification.
*   **Cheminformatics & SMILES Integrity:** Enforce strict ring-closure digit symmetry, standard valency rules (no 5-bonded Nitrogen), and 100% consistency between IUPAC chemical names and generated SMILES strings.

---

## 6. Automated Execution Checklist (v3 Pipeline)
1.  **Specification Alignment:** Verify tensor equations against `DERIVATION_v2.md` (Sections 1–6).
2.  **Branch-Free & Batch Audit:** Scan `tdt_engine/core_v2.py`, `tdt_engine/solvers_v3.py`, and `tdt_engine/calibrator_nn_v3.py` to confirm zero `if-else` statements in physics loops and full $[B]$ batch tensor compatibility.
3.  **Gradient Highway Check:** Run `calibrator_nn_v3.py` standalone verification to confirm `raw_R.grad` and `raw_angle.grad` propagate cleanly without `NaN` or `Inf`.
4.  **5-Stage Training & OOS Validation:** Execute `tdt_engine/train_calibrator_v3.py` through `STAGE 1`–`STAGE 5`, verifying convergence of radian-unified loss, `.xyz` geometry export, and Out-of-Sample generalization logging.
