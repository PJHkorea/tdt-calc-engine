# Computational Mathematization for Differentiable Programming (v2)

## Mathematical Derivation of the Hybrid Differentiable Molecular Geometry Engine (tdt_engine_v2)

This document provides the formal mathematical derivation of the **v2 Hybrid Physics-Informed Differentiable Framework** utilized in **tdt_engine**. The architecture integrates invariant topological phase-transition operators with standardized empirical force-field approximations (Covalent Radii and UFF potentials) inside a strictly smooth, end-to-end differentiable tensor manifold.

## 1. Fundamental Arithmetic Invariants and Extended Topological Shell Space

We define the read-only invariant background space using the foundational physical and number-theoretic constants normalized to atomic length scales ($\text{\AA}$):

$$
\mathcal{S} = \{ a_0, \omega_1, \alpha \}
$$

Where:
- $a_0 = 0.5291772109 \text{ \AA}$ (Bohr Radius in Angstroms)
- $\omega_1 = 14.1347251417$ (First Non-Trivial Zero of the Riemann Zeta Function, $\zeta(s) = 0$)
- $\alpha = 0.0072973525$ (Baryon Fine-Structure Constant)

To support heavy halogen and pharmaceutical elements (e.g., $\text{Br}, \text{I}$) without discontinuous conditional branching (`if-else`), the continuous phase-transition boundary for atomic shells $N(\nu)$ as a function of atomic number $\nu \in \mathbb{R}^+$ is extended across periods 1 through 6, modulated by the topological permittivity $\delta$:

$$
\delta = \alpha \cdot \omega_1 \approx 0.103147
$$

$$
N(\nu) = 1.0 + \sum_{k \in \{2, 10, 18, 36, 54\}} \left[ \frac{1}{2} + \frac{1}{2}\tanh \left( \frac{\nu - (k + 0.5)}{\delta} \right) \right]
$$

---

## 2. Continuous Topological Core Density and Standardized Covalent Baseline

To eliminate discontinuous piecewise conditional branching (`if-else`) at the hydrogen boundary ($\nu \to 1$), the Riemann localization density $\zeta_{\rho}(\nu)$ is reformulated into a strictly smooth, infinitely differentiable mapping using a continuous soft-floor operator:

$$
\zeta_{\rho}(\nu) = 10^{-5} + \ln\left( 1.0 + \frac{\omega_1 \cdot \mathrm{softplus}(\nu - 1.0)}{2\pi} \right), \qquad \mathrm{softplus}(x) = \ln(1 + e^x)
$$

To prevent unbounded radial expansion in heavy periods ($N(\nu) \ge 3$) while preserving first-principles topological modulation, the effective atomic core radius $a_{\text{C}}(\nu)$ anchors a read-only standardized covalent radius lookup baseline $r_{\text{cov}}(\nu)$ (Pyykkö/Cordero scale) modulated by the continuous shell transition $\Theta_{\text{shell}}$ and zeta compression field:

$$
\Theta_{\text{shell}}(\nu) = \frac{1}{2} + \frac{1}{2}\tanh\left(\frac{N(\nu) - 1.5}{0.1}\right)
$$

$$
a_{\text{C}}(\nu) = r_{\text{cov}}(\nu) \cdot \left[ 1.0 + \alpha \cdot \zeta_{\rho}(\nu) \cdot \Theta_{\text{shell}}(\nu) \right] \cdot \tanh\left(\frac{\omega_1 \nu}{\alpha N(\nu)^2 \pi^2}\right)
$$

Applying Kepler's optimal hypersphere packing invariant in 4-dimensional Euclidean space ($\kappa_4 = \frac{\pi}{\sqrt{18}}$), the normalized topological core density $\rho_{\text{C}}(\nu)$ is bounded cleanly without quartic divergence:

$$
\rho_{\text{C}}(\nu) = \tanh\left[ \left(\frac{a_{\text{C}}(\nu)}{a_0}\right)^2 \cdot \frac{\pi}{\sqrt{18}} \right] \cdot \left(1 + \alpha \cdot \zeta_{\rho}(\nu)\right)
$$

---

* **Branch-Free Read-Only Tensor Buffer Lookup ($\nu \in \{1, \dots, 86\}$):**
To retrieve standardized empirical baselines ( $r_{\text{cov}}(\nu)$ and $\chi_{\text{pauling}}(\nu)$ ) across batched atomic tensors without conditional branching or dictionary lookups, constants are registered as immutable read-only lookup buffers ($\mathbf{T}\_{\text{cov}}, \mathbf{T}\_{\chi} \in \mathbb{R}^{87}$):

$$
r_{\text{cov}}(\nu) = \mathbf{T}_{\text{cov}}[\lfloor \nu \rfloor], \qquad \chi_{\text{pauling}}(\nu) = \mathbf{T}_{\chi}[\lfloor \nu \rfloor]
$$

---


## 3. Closed-Form Hybrid Bond Length Convergence and Metric Tensor Modifier

To eliminate iterative root-finding bottlenecks (200-step gradient loops) and prevent halving artifacts in period-3+ halides ($\nu \ge 17$), the equilibrium bond distance $R_{\text{bond}}$ is reformulated into a single-pass, closed-form algebraic expression combining standardized electronegativity contraction with continuous topological tensors.

For a bonded pair of nuclei $A$ and $B$ with atomic numbers $\nu_A, \nu_B$ and Pauling electronegativities $\chi_A, \chi_B$, the smooth topological asymmetry $\eta_{\text{asym}}$ and smoothed density contrast $\Delta_{\rho}$ are defined without non-differentiable absolute-value cusps ($\epsilon = 10^{-8}$):

$$
\eta_{\text{asym}} = \left( \frac{\nu_A - \nu_B}{\nu_A + \nu_B} \right)^2, \qquad \Delta_{\rho} = \frac{\sqrt{(\rho_A - \rho_B)^2 + \epsilon}}{\rho_A + \rho_B + \epsilon}
$$

The continuous ionic field modifier $\Phi_{\text{ionic}}$ and universal dielectric scaling $\Omega_{\text{universal}}$ regulate charge-transfer contraction smoothly across covalent-to-ionic regimes:

$$
\Phi_{\text{ionic}} = 1.0 - \alpha\sqrt{\omega_1} \cdot \Delta_{\rho} \cdot \tanh(\rho_A\rho_B)
$$

$$
\mathcal{L}_{\text{ionic}} = 1.0 + \alpha \cdot \left( \frac{\omega_1}{\pi} \right), \qquad \Omega_{\text{universal}} = 1.0 + (\mathcal{L}_{\text{ionic}} - 1.0) \cdot \tanh \left( \frac{\Delta_{\rho}}{\alpha} \right)
$$

Replacing the discontinuous $0.5\times$ heavy-element truncation, the metric tensor modifier $\chi_{\mu\nu}$ preserves radial scale integrity across all periods:

$$
\chi_{\mu\nu} = \left[ 1.0 - \alpha \cdot \eta_{\text{asym}} \cdot \omega_1 \cdot \frac{\pi}{2} \right] \cdot \Phi_{\text{ionic}} \cdot \Omega_{\text{universal}}
$$

To account for multiple-bond shortening ($\nu_{\text{bond}} \in [1.0, 3.0]$) and hydride charge-inversion expansion without `if-else` branches, we define the continuous bond-order damper $\mathcal{B}\_\{\text{scale}}$ and soft hydride gate $\mathcal{H}\_\{\text{gate}}$:

$$
\mathcal{B}_{\text{scale}}(\nu_{\text{bond}}) = \frac{1.0}{1.0 + \alpha\omega_1(\nu_{\text{bond}} - 1.0)} \cdot \left[ 1.0 - \left( \frac{\alpha\omega_1}{\pi} \right) \cdot \left( \frac{1}{2} + \frac{1}{2}\tanh\left( \frac{\nu_{\text{bond}} - 1.5}{0.05} \right) \right) \right]
$$

$$
\nu_{\text{min}} = \min(\nu_A, \nu_B), \qquad \nu_{\text{max}} = \max(\nu_A, \nu_B)
$$

$$
\mathcal{H}_{\text{gate}} = \left[ \frac{1}{2} + \frac{1}{2}\tanh\left( \frac{1.5 - \nu_{\text{min}}}{0.05} \right) \right] \cdot \left[ \frac{1}{2} + \frac{1}{2}\tanh\left( \frac{\nu_{\text{max}} - 1.5}{0.05} \right) \right]
$$

$$
\mathcal{H}_{\text{expansion}} = 1.0 + \mathcal{H}_{\text{gate}} \cdot \eta_{\text{asym}} \cdot (1.0 - \tanh(\rho_A\rho_B)) \cdot \left( \frac{\omega_1}{N(\nu_{\text{max}})\pi} \right)
$$

Finally, anchoring the modified core radii $a_{A,\text{mod}}, a_{B,\text{mod}}$ with the standardized Schomaker–Stevenson electronegativity contraction term ($c_{\text{SS}} = 0.08 \text{ \AA}$) yields the direct, closed-form equilibrium bond length $R_{\text{bond}}$ in a single vectorized tensor pass:

$$
a_{A,\text{mod}} = a_{\text{C}}(\nu_A) \cdot \mathcal{B}_{\text{scale}} \cdot \mathcal{H}_{\text{expansion}} \cdot \chi_{\mu\nu}, \qquad a_{B,\text{mod}} = a_{\text{C}}(\nu_B) \cdot \mathcal{B}_{\text{scale}} \cdot \mathcal{H}_{\text{expansion}} \cdot \chi_{\mu\nu}
$$

$$
R_{\text{bond}} = \left( a_{A,\text{mod}} + a_{B,\text{mod}} \right) - c_{\text{SS}} \cdot \sqrt{(\chi_A - \chi_B)^2 + \epsilon} \cdot (1.0 - \mathcal{H}_{\text{gate}})
$$

---

## 4. Hybrid UFF-VSEPR Knot Energy and Continuous Angular Alignment

To eliminate piecewise conditional branching (`if-else`) in valence electron counting and resolve multi-ligand steric miscalculations 
(e.g., $\text{CH}\_\4$, $\text{NH}\_\3$), the effective valence electron count $\nu_{\text{val}}$ of the central atom $\nu_C$ is derived smoothly via continuous shell gates:

$$
\Theta_{2}(\nu_C) = \frac{1}{2} + \frac{1}{2}\tanh\left(\frac{N(\nu_C) - 1.5}{0.1}\right), \qquad \Theta_{3}(\nu_C) = \frac{1}{2} + \frac{1}{2}\tanh\left(\frac{N(\nu_C) - 2.5}{0.1}\right)
$$

$$
\nu_{\text{val}} = \nu_C - 2.0\,\Theta_{2}(\nu_C) - 8.0\,\Theta_{3}(\nu_C)
$$

Given the total coordination bond-order sum $\Sigma\_\{\text{bond}} = \sum_{k} \nu\_\{bk}$ around the central atom (rather than only two arbitrary branches), the smooth lone-pair density operator $\rho\_\{\text{LP}}$ is formulated without hard zero-clamping discontinuities ($\epsilon = 10^{-8}$):

$$
\bar{\nu}_{\text{LP}} = \frac{\nu\_\{\text{val}} - \Sigma\_\{\text{bond}}}{2.0}, \qquad \rho\_\{\text{LP}} = \mathrm{softplus}\_\{\beta}(\bar{\nu}\_\{\text{LP}}) = \frac{1}{\beta}\ln\left(1 + e^{\beta \bar{\nu}\_\{\text{LP}}}\right) \quad (\beta = 50.0)
$$

Let $N_{\text{lig}}$ denote the number of bonded ligands attached to the central atom. The effective Steric Number $\mathcal{S}\_\{\text{eff}} = N\_\{\text{lig}} + \rho\_\{\text{LP}}$ governs the baseline VSEPR reference geometry $\theta\_\{\text{VSEPR}}$ smoothly across linear ($180^\circ$), trigonal planar ($120^\circ$), and tetrahedral ($109.4712^\circ$) regimes:

$$
\theta_{\text{VSEPR}} = \pi - \left(\pi - \frac{2\pi}{3}\right)\left[\frac{1}{2} + \frac{1}{2}\tanh\left(\frac{\mathcal{S}_{\text{eff}} - 2.5}{0.15}\right)\right] - \left(\frac{2\pi}{3} - \theta_{\text{tet}}\right)\left[\frac{1}{2} + \frac{1}{2}\tanh\left(\frac{\mathcal{S}_{\text{eff}} - 3.5}{0.15}\right)\right]
$$

Where $\theta\_\{\text{tet}} = 109.4712^\circ \times \frac{\pi}{180^\circ}$. The smooth 3-body electronegativity asymmetry $\mathcal{A}\_\{\text{neg}}$ and lone-pair angular compression $\theta\_\{\text{shrunk}}$ refine the target equilibrium angle $\theta\_\{\text{target}}$:

$$
\mathcal{A}_{\text{neg}} = \frac{\sqrt{\left(\nu_C - 0.5(\nu_{S1} + \nu_{S2})\right)^2 + \epsilon}}{\nu_C + 0.5(\nu_{S1} + \nu_{S2})}
$$

$$
\theta_{\text{shrunk}} = \alpha \left(\frac{\omega_1}{\pi}\right) (1.0 + \alpha\mathcal{A}_{\text{neg}}) \cdot \frac{\sqrt{\rho_{\text{LP}} + \epsilon} \cdot \omega_1}{\Phi_{\text{side,mod}}^{\gamma_{\text{scaler}}}}, \qquad \gamma_{\text{scaler}} = 1.0 - \alpha\omega_1
$$

$$
\theta_{\text{target}} = \theta_{\text{VSEPR}} - \theta_{\text{shrunk}}
$$

To achieve full compatibility with the Riemannian $SO(2)$ circular manifold while preventing artificial over-contraction in linear molecules ($\text{CO}\_\2$), the angular bending energy $\Delta\_\{\text{UFF}}(\theta)$ adopts the standardized **Universal Force Field (UFF)** cosine harmonic expansion coupled with the self-consistent topological tension $\mathcal{C}\_\{\text{self}}$:

$$
\Delta_{\text{UFF}}(\theta) = \frac{1 - \cos(\theta - \theta_{\text{target}})}{\sin^2(\theta_{\text{target}}) + \epsilon_{\text{lin}}}, \qquad \epsilon_{\text{lin}} = 0.05
$$

$$
\mathcal{T}_{\text{background}} = \exp \left( -\frac{\Phi_{\text{side,mod}}}{\gamma_{\text{scaler}}} \right), \qquad \mathcal{C}_{\text{self}} = \frac{1}{\alpha} \cdot \rho_C^{(1.0 - \gamma_{\text{scaler}})} \cdot \frac{\omega_1}{4\pi} \cdot \gamma_{\text{scaler}}^2 \cdot \mathcal{T}_{\text{background}}
$$

Finally, the total multi-dimensional knot energy $\mathbb{E}\_\{\text{knot}}(\theta)$ combines the smooth spatial residual $\Delta\_\{\text{space}}(\theta)$, the UFF-coupled angular potential, and the symmetric damped Yukawa-Pauli short-range steric barrier $\mathcal{P}\_\{\text{Pauli, v2}}(\theta)$:

$$
\Delta_{\text{space}}(\theta) = \frac{\sqrt{\left(R_{\text{AB,opt}}(\theta) - R_{\text{AB,geo}}(\theta)\right)^2 + \epsilon}}{R_A + R_B}
$$

$$
\mathcal{P}_{\text{Pauli, v2}}(\theta) = (a_A + a_B)\alpha\omega_1 \cdot \frac{\exp(-\theta / 0.4)}{\sin^2(\theta / 2) + 10^{-6}}
$$

$$
\mathbb{E}_{\text{knot}}(\theta) = \Delta_{\text{space}}(\theta) + \Delta_{\text{UFF}}(\theta) \cdot \mathcal{C}_{\text{self}} + \mathcal{P}_{\text{Pauli, v2}}(\theta)
$$

---

## 5. Batched Riemannian $SO(2)$ Manifold Projection and Geodesic Retraction Solver

To eliminate numerical singularities and artificial boundary clamping discontinuities ($\theta \to 0$ or $\theta \to \pi$) while scaling to multi-body polyatomic molecules, all $B$ angular degrees of freedom in a molecular graph are vectorized simultaneously onto the compact 1-sphere product manifold $\mathcal{M}^B = (\mathbb{S}^1)^B \subset \mathbb{R}^{B \times 2}$:

$$
\mathbf{X} = \begin{bmatrix} \cos\boldsymbol{\theta} & \sin\boldsymbol{\theta} \end{bmatrix} \in \mathbb{R}^{B \times 2}, \quad \text{subject to } \|\mathbf{x}_i\|_2 = 1 \quad (\forall i \in \{1, \dots, B\})
$$

### 5.1 Batched Orthogonal Riemannian Gradient Derivation

For each angular coordinate $\mathbf{x}\_i \in \mathbb{R}^2$, the unconstrained Euclidean gradient $\nabla\_\{\mathbf{x}\_i} f(\mathbf{X})$ contains normal components that violate the unit-circle constraint. To confine the optimization trajectory strictly to the tangent bundle $T\_\{\mathbf{X}}\mathcal{M}^B$, we apply the vectorized orthogonal projection operator $\mathcal{P}\_\{\mathbf{x}\_\i}$:

$$
\mathrm{Grad} \, f(\mathbf{x}_i) = \mathcal{P}_{\mathbf{x}_i}\left(\nabla_{\mathbf{x}_i} f(\mathbf{X})\right) = \left( \mathbf{I}_2 - \mathbf{x}_i \mathbf{x}_i^{T} \right) \nabla_{\mathbf{x}_i} f(\mathbf{X})
$$

Explicitly expanding this contraction across the batch yields:

$$
\mathrm{Grad} \, f(\mathbf{x}_i) = \begin{bmatrix} 1 - x_{i,1}^2 & -x_{i,1} x_{i,2} \\ -x_{i,1} x_{i,2} & 1 - x_{i,2}^2 \end{bmatrix} \begin{bmatrix} \frac{\partial f}{\partial x_{i,1}} \\ \frac{\partial f}{\partial x_{i,2}} \end{bmatrix}
$$

This tangential projection naturally modulates the metric tensor $g\_\{\mu\nu}$, automatically compressing step sizes near the steep Yukawa-Pauli steric wall ($\theta \to 0$) without coordinate explosions.

### 5.2 Numerically Stabilized Geodesic Retraction and Branch-Free Folding

To advance all $B$ angular vectors concurrently along the exact circular geodesics without drifting off $\|\mathbf{x}\_i\|\_2 = 1$, let $\mathbf{d}\_\{i,k} = -\mathrm{Grad} \, f(\mathbf{x}\_\{i,k})$ be the tangential descent direction and $\eta\_k$ be the adaptive step size. Introducing a numerical stabilizer $\epsilon = 10^{-12}$ prevents division-by-zero singularities at stationary points ($\|\mathbf{d}\_\{i,k}\|\_2 \to 0$):

$$
\mathbf{x}_{i,k+1} = \mathcal{R}_{\mathbf{x}_{i,k}}(\eta_k \mathbf{d}_{i,k}) = \mathbf{x}_{i,k} \cos\left(\eta_k \|\mathbf{d}_{i,k}\|_2\right) + \frac{\mathbf{d}_{i,k}}{\|\mathbf{d}_{i,k}\|_2 + \epsilon} \sin\left(\eta_k \|\mathbf{d}_{i,k}\|_2\right)
$$

Upon vectorized convergence ($\max_i \|\mathrm{Grad} \, f(\mathbf{x}\_i)\|\_2 < 10^{-7}$), rather than relying on piecewise conditional branches (`if-else`) for lower-semicircle trajectories, we exploit the intrinsic cosine symmetry of the unit circle ($\cos(2\pi - \theta) = \cos\theta = x\_\{i,1}$) to recover the folded valence angle $\theta\_\{i,\text{final}} \in [0, \pi]$ in a single closed-form differentiable operation ($\epsilon\_\{\text{acos}} = 10^{-7}$):

$$
\theta_{i,\text{final}} = \arccos\left( \mathrm{clamp}\left(x_{i,1},\; -1.0 + \epsilon_{\text{acos}},\; 1.0 - \epsilon_{\text{acos}}\right) \right)
$$

This branch-free geodesic folding guarantees $C^\infty$ smoothness across the interior valence domain $(0, \pi)$ and enables simultaneous batch optimization across entire polyatomic molecular graphs.


---

## 6. Pure Residual Neural Calibration Architecture (`calibrator_nn_v3.py`)

With the hybrid UFF-VSEPR and standardized covalent baseline eliminating severe algebraic divergences in Stages 2–5, all hardcoded molecular lookup gates ($G_{\text{tetra}}$, $G_{\text{linear}}$, $\theta_{\text{restoration}}$) are completely removed. The calibration module is refactored into a **pure, generalizable Residual Physics Learning layer** that captures subtle many-body electronic and steric deviations across arbitrary polyatomic molecules.

### 6.1 Generalizable Bond Length Residual Flux

Given the permutation-invariant bond feature tensor $\mathbf{v}\_\{\text{len}} = [\nu_{\text{min}}, \nu_{\text{max}}, \nu_{\text{bond}}, R_{\text{raw}}, \Delta_{\rho}]^{T}$, a lightweight Multi-Layer Perceptron $\mathcal{R}\_\{\text{len}}$ predicts the continuous residual offset against reference quantum/DFT targets. Network weights are initialized within $\pm 10^{-4}$ to guarantee identity preservation at initialization:

$$
R_{\text{calibrated}} = R_{\text{raw}} + \mathcal{R}_{\text{len}}(\mathbf{v}_{\text{len}})
$$

### 6.2 Physics-Enriched Angular Residual Field on $SO(2)$

To enable the angular residual network $\mathcal{R}\_\{\text{ang}}$ to generalize across periods and lone-pair regimes without hardcoded element switches, the input feature vector $\mathbf{v}\_\{\text{ang}}$ is enriched with the central topological core density $\rho_C$ and lone-pair density $\rho_{\text{LP}}$ derived in Sections 2 and 4:

$$
\mathbf{v}_{\text{ang}} = \left[\nu_{\text{center}},\; \nu_{\text{smin}},\; \nu_{\text{smax}},\; \nu_{b1},\; \nu_{b2},\; \rho_C,\; \rho_{\text{LP}},\; \theta_{\text{raw}}\right]^{T}
$$

Working natively in radians ( $\theta \in (0, \pi)$ ), the calibrated equilibrium angle is obtained via smooth residual addition followed by a differentiable range-preserving projection:

$$
\theta_{\text{unbounded}} = \theta_{\text{raw}} + \mathcal{R}_{\text{ang}}(\mathbf{v}_{\text{ang}})
$$

$$
\theta_{\text{calibrated}} = \pi \cdot \sigma\left( \mathrm{logit}\left(\frac{\theta_{\text{raw}}}{\pi}\right) + \mathcal{R}_{\text{ang}}(\mathbf{v}_{\text{ang}}) \right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

*(Note: Operating in logit-sigmoid space guarantees $\theta\_\{\text{calibrated}} \in (0, \pi)$ 

strictly without hard clamping discontinuities, while reducing smoothly to $\theta\_\{\text{raw}}$ when $\mathcal{R}\_\{\text{ang}} \to 0$.)*

---

* **Numerical Boundary Stabilization ($\epsilon_{\text{logit}} = 10^{-6}$):**
To prevent logarithmic singularities ( $\mathrm{logit}(0) \to -\infty$, $\mathrm{logit}(1) \to +\infty$) when $\theta_{\text{raw}} $ approaches $0$ or $\pi$ in linear triatomic systems (e.g., $\text{CO}_2$ , $\text{HCN}$) , the normalized angular ratio is smoothly bounded prior to logit transformation:

$$
p_{\text{raw}} = \mathrm{clamp}\left( \frac{\theta_{\text{raw}}}{\pi},\; \epsilon_{\text{logit}},\; 1.0 - \epsilon_{\text{logit}} \right), \qquad \theta_{\text{calibrated}} = \pi \cdot \sigma\left( \mathrm{logit}(p_{\text{raw}}) + \mathcal{R}_{\text{ang}}(\mathbf{v}_{\text{ang}}) \right)
$$

---

### 6.3 Scale-Unified Multi-Flux Objective and Pseudo-Force Guardrail

By formulating all angular residuals in radians rather than degrees, the dimensional variance of bond lengths ($\text{\AA}^2$) and bond angles ($\text{rad}^2$) becomes naturally commensurable. This eliminates the need for extreme gradient-suppressing factors ($\lambda_{\text{ang}} = 10^{-4}$) that previously stalled angular network convergence:

$$
\mathcal{E}_{\text{total, v2}} = \frac{1}{M}\sum_{i=1}^{M} \left(R_{\text{calibrated}}^{(i)} - R_{\text{ref}}^{(i)}\right)^2 + \lambda_{\text{rad}} \cdot \frac{1}{N}\sum_{j=1}^{N} \left(\theta_{\text{calibrated}}^{(j)} - \theta_{\text{ref}}^{(j)}\right)^2, \qquad \lambda_{\text{rad}} \approx 1.0
$$

The resulting analytical pseudo-force field $\mathbf{F}\_\{\text{pseudo}} = -\nabla\_\{\mathbf{X}} \mathcal{E}\_\{\text{total, v2}}$ provides an ultra-fast, $C^\infty$-smooth physical guardrail signal for downstream molecular generative models and screening pipelines.
