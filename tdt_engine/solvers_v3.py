import torch
import torch.nn.functional as F
from tdt_engine.core_v2 import TDTEngineCore

class TDTManifoldSolver:
    def __init__(self, core_engine: TDTEngineCore = None, device: str = None):
        self.core = core_engine if core_engine else TDTEngineCore()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.a0 = torch.tensor(self.core.a0, dtype=torch.float64, device=self.device)
        self.omega_1 = torch.tensor(self.core.omega_1, dtype=torch.float64, device=self.device)
        self.alpha = torch.tensor(self.core.alpha, dtype=torch.float64, device=self.device)
        
        # core.py에 내장된 T_cov, T_chi 배열을 그대로 가져와 PyTorch 텐서화 (중복 제거)
        self.T_cov = torch.tensor(self.core.T_cov, dtype=torch.float64, device=self.device)
        self.T_chi = torch.tensor(self.core.T_chi, dtype=torch.float64, device=self.device)

    def _to_tensor(self, val):
        if isinstance(val, torch.Tensor):
            return val.to(dtype=torch.float64, device=self.device)
        return torch.tensor(val, dtype=torch.float64, device=self.device)

    def lookup_atomic_constants(self, nu):
        nu_t = self._to_tensor(nu)
        idx = torch.clamp(torch.floor(nu_t).to(torch.long), min=1, max=86)
        return self.T_cov[idx], self.T_chi[idx]

    def compute_topological_density_tensor(self, nu_core) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nu_t = self._to_tensor(nu_core)
        N_shell, _ = self.compute_automated_shell_tensor(nu_t)
        r_cov, _ = self.lookup_atomic_constants(nu_t)
        
        zeta_density = 1e-5 + torch.log(1.0 + (self.omega_1 * F.softplus(nu_t - 1.0)) / (2.0 * torch.pi))
        theta_shell = 0.5 + 0.5 * torch.tanh((N_shell - 1.5) / 0.1)
        core_val = (self.omega_1 * nu_t) / (self.alpha * (N_shell ** 2) * (torch.pi ** 2))
        a_C = r_cov * (1.0 + self.alpha * zeta_density * theta_shell) * torch.tanh(core_val)
        
        kepler_packing = torch.pi / torch.sqrt(torch.tensor(18.0, dtype=torch.float64, device=self.device))
        rho_C = torch.tanh(((a_C / self.a0) ** 2) * kepler_packing) * (1.0 + self.alpha * zeta_density)
        return rho_C, a_C, zeta_density

    def _get_core_tensors(self, nu_core) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_topological_density_tensor(nu_core)

    def compute_automated_shell_tensor(self, nu_core) -> tuple[torch.Tensor, torch.Tensor]:
        nu_t = self._to_tensor(nu_core)
        delta = self.alpha * self.omega_1
        critical_nodes = torch.tensor([2.0, 10.0, 18.0, 36.0, 54.0], dtype=torch.float64, device=self.device)
        phase_transitions = 0.5 + 0.5 * torch.tanh((nu_t.unsqueeze(-1) - (critical_nodes + 0.5)) / delta)
        n_shell_continuous = 1.0 + torch.sum(phase_transitions, dim=-1)
        return torch.round(n_shell_continuous), n_shell_continuous

    def _compute_bond_length_tensor(self, nu_A, nu_B, nu_bond=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nu_A_t, nu_B_t, nu_bond_t = self._to_tensor(nu_A), self._to_tensor(nu_B), self._to_tensor(nu_bond)
        eps = 1e-8

        rho_A, a_A, _ = self.compute_topological_density_tensor(nu_A_t)
        rho_B, a_B, _ = self.compute_topological_density_tensor(nu_B_t)
        _, chi_A = self.lookup_atomic_constants(nu_A_t)
        _, chi_B = self.lookup_atomic_constants(nu_B_t)

        asymmetry_factor = ((nu_A_t - nu_B_t) / (nu_A_t + nu_B_t + eps)) ** 2
        delta_rho = torch.sqrt((rho_A - rho_B) ** 2 + eps) / (rho_A + rho_B + eps)
        scale_invariant = torch.tanh(rho_A * rho_B)

        ionic_field_modifier = self.alpha * torch.sqrt(self.omega_1) * delta_rho * scale_invariant
        ionic_contraction = (1.0 - ionic_field_modifier) / (1.0 + self.alpha * asymmetry_factor)
        geometric_ionic_limit = 1.0 + (self.alpha * (self.omega_1 / torch.pi))
        universal_ionic_scale = 1.0 + (geometric_ionic_limit - 1.0) * torch.tanh(delta_rho / self.alpha)

        chi_tensor = (1.0 - self.alpha * asymmetry_factor * self.omega_1 * (torch.pi / 2.0)) * ionic_contraction * universal_ionic_scale

        bond_order_damp = 1.0 / (1.0 + self.alpha * self.omega_1 * (nu_bond_t - 1.0))
        triple_bond_gate = 1.0 - (self.alpha * self.omega_1 / torch.pi) * (0.5 + 0.5 * torch.tanh((nu_bond_t - 1.5) / 0.05))
        B_scale = bond_order_damp * triple_bond_gate

        nu_min, nu_max = torch.minimum(nu_A_t, nu_B_t), torch.maximum(nu_A_t, nu_B_t)
        H_gate = (0.5 + 0.5 * torch.tanh((1.5 - nu_min) / 0.05)) * (0.5 + 0.5 * torch.tanh((nu_max - 1.5) / 0.05))
        H_expansion = 1.0 + H_gate * self.alpha * asymmetry_factor * self.omega_1

        a_A_mod = a_A * B_scale * H_expansion * chi_tensor
        a_B_mod = a_B * B_scale * H_expansion * chi_tensor

        c_SS = 0.08
        R_bond = (a_A_mod + a_B_mod) - c_SS * torch.sqrt((chi_A - chi_B) ** 2 + eps) * (1.0 - H_gate)
        return R_bond, a_A_mod, a_B_mod

    def compute_bond_length(self, nu_A, nu_B, nu_bond=1.0):
        R_bond, a_A_mod, a_B_mod = self._compute_bond_length_tensor(nu_A, nu_B, nu_bond)
        if R_bond.ndim == 0:
            return float(R_bond.detach().item()), float(a_A_mod.detach().item()), float(a_B_mod.detach().item())
        return R_bond, a_A_mod, a_B_mod



    def _compute_angle_tensor(self, nu_center, nu_side1, nu_side2, nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=None):
        """
        [DERIVATION_v2 Section 4 & 5 - 배치 리만 SO(2) 다양체 최적화 코어]
        UFF-VSEPR 입체수(S_eff) 연속 전이, 유카와-파울리 단거리 척력,
        그리고 arccos 기반 무분기(Branch-free) 측지선 최단 호 접기를 수행합니다.
        """
        nu_c_t = self._to_tensor(nu_center)
        nu_s1_t = self._to_tensor(nu_side1)
        nu_s2_t = self._to_tensor(nu_side2)
        nb1_t = self._to_tensor(nu_b1)
        nb2_t = self._to_tensor(nu_b2)
        n_lig_t = self._to_tensor(n_lig)

        is_scalar = (nu_c_t.ndim == 0)
        if is_scalar:
            nu_c_t = nu_c_t.unsqueeze(0)
            nu_s1_t = nu_s1_t.unsqueeze(0)
            nu_s2_t = nu_s2_t.unsqueeze(0)
            nb1_t = nb1_t.unsqueeze(0)
            nb2_t = nb2_t.unsqueeze(0)
            n_lig_t = n_lig_t.unsqueeze(0)

        B = nu_c_t.shape[0]
        if sigma_bond is None:
            sigma_bond_t = nb1_t + nb2_t
        else:
            sigma_bond_t = self._to_tensor(sigma_bond)
            if sigma_bond_t.ndim == 0:
                sigma_bond_t = sigma_bond_t.expand(B)

        eps = 1e-8

        # 1. 결합 길이 및 위상 밀도 텐서 연동
        R_A, a_A, _ = self._compute_bond_length_tensor(nu_c_t, nu_s1_t, nb1_t)
        R_B, a_B, _ = self._compute_bond_length_tensor(nu_c_t, nu_s2_t, nb2_t)

        rho_C, a_C_raw, _ = self.compute_topological_density_tensor(nu_c_t)
        _, a_S1_raw, _ = self.compute_topological_density_tensor(nu_s1_t)
        _, a_S2_raw, _ = self.compute_topological_density_tensor(nu_s2_t)

        Phi_C = torch.clamp(1.0 - (a_C_raw / self.a0), min=1e-5)

        # 2. 연속 가전자 및 비공유 전자쌍(rho_LP) 연산 (if-else 완전 제거)
        _, N_shell_cont = self.compute_automated_shell_tensor(nu_c_t)
        theta_2 = 0.5 + 0.5 * torch.tanh((N_shell_cont - 1.5) / 0.1)
        theta_3 = 0.5 + 0.5 * torch.tanh((N_shell_cont - 2.5) / 0.1)
        valence_electrons = nu_c_t - 2.0 * theta_2 - 8.0 * theta_3

        raw_lone_pair = (valence_electrons - sigma_bond_t) / 2.0
        rho_LP = F.softplus(raw_lone_pair, beta=50.0)

        # 3. 입체수(S_eff) 기반 UFF-VSEPR 기준 각도 연속 전이 (180 -> 120 -> 109.47)
        S_eff = n_lig_t + rho_LP
        theta_linear = torch.pi
        theta_trig = 2.0 * torch.pi / 3.0
        theta_tet = torch.tensor(109.4712 * torch.pi / 180.0, dtype=torch.float64, device=self.device)

        theta_vsepr = theta_linear \
            - (theta_linear - theta_trig) * (0.5 + 0.5 * torch.tanh((S_eff - 2.5) / 0.15)) \
            - (theta_trig - theta_tet) * (0.5 + 0.5 * torch.tanh((S_eff - 3.5) / 0.15))

        # 4. 전기음성도 유전 차폐 및 비공유 전자쌍 각도 압축
        Phi_side1 = 1.0 - torch.tanh(a_S1_raw / self.a0) * (0.5 + 0.5 * torch.tanh((nu_s1_t - 1.5) / 0.05))
        Phi_side2 = 1.0 - torch.tanh(a_S2_raw / self.a0) * (0.5 + 0.5 * torch.tanh((nu_s2_t - 1.5) / 0.05))
        Phi_side_raw = 0.5 * (Phi_side1 + Phi_side2)

        A_neg = torch.sqrt((nu_c_t - 0.5 * (nu_s1_t + nu_s2_t)) ** 2 + eps) / (nu_c_t + 0.5 * (nu_s1_t + nu_s2_t))
        Phi_side_mod = Phi_side_raw * (1.0 + self.alpha * torch.sqrt(self.omega_1) * (1.0 - Phi_side_raw) * torch.tanh(A_neg))

        gamma_scaler = 1.0 - (self.alpha * self.omega_1)
        A_harmonic = self.alpha * (self.omega_1 / torch.pi) * (1.0 + self.alpha * A_neg)
        theta_shrunk = A_harmonic * torch.sqrt(rho_LP + eps) * self.omega_1 / (Phi_side_mod ** gamma_scaler)

        theta_target = torch.clamp(theta_vsepr - theta_shrunk, min=0.1, max=torch.pi - 1e-4)

        # 5. [B x 2] 배치 리만 SO(2) 다양체 최적화 루프 (90도 중립 초기화)
        X = torch.zeros((B, 2), dtype=torch.float64, device=self.device)
        X[:, 1] = 1.0
        X.requires_grad_(True)
        lr_riemannian = 0.02

        T_bg = torch.exp(-Phi_side_mod / gamma_scaler)
        # 1. 루프 진입 전 C_self 정의부
        C_self = 10.0 * (1.0 / self.alpha) * (rho_C ** (1.0 - gamma_scaler)) * (self.omega_1 / (4.0 * torch.pi)) * (gamma_scaler ** 2) * T_bg



        for _ in range(800):
            if X.grad is not None:
                X.grad.zero_()

            theta_cur = torch.atan2(X[:, 1], X[:, 0])
            sin_sq = torch.sin(theta_cur) ** 2
            cos_sq = torch.cos(theta_cur) ** 2

            # 공간 잔차 항
            S_theta = sin_sq * torch.exp(-(self.alpha * self.omega_1 * (nb1_t + nb2_t)) / (1.0 + cos_sq))
            R_AB_opt = (a_A + a_B) * (1.0 + (1.0 - S_theta) * (self.omega_1 / torch.pi) * Phi_C)
            R_AB_geo = torch.sqrt(torch.abs((R_A ** 2) + (R_B ** 2) - 2.0 * R_A * R_B * torch.cos(theta_cur)) + eps)
            delta_space = torch.sqrt((R_AB_opt - R_AB_geo) ** 2 + eps) / (R_A + R_B)

            # UFF 코사인 조화 각도 포텐셜
            delta_uff = (1.0 - torch.cos(theta_cur - theta_target)) / (torch.sin(theta_target) ** 2 + 0.05)

            # 유카와-파울리 단거리 척력 장벽
            half_sin_sq = torch.sin(0.5 * theta_cur) ** 2
            # 3. _compute_angle_tensor 루프 내부: OF2 예각 붕괴 차단
            pauli_barrier = (a_A + a_B) * self.alpha * self.omega_1 * torch.exp(-theta_cur / 0.4) / (half_sin_sq + 1e-6)


            loss_vec = delta_space + delta_uff * C_self + pauli_barrier
            loss_vec.sum().backward()

            with torch.no_grad():
                grad_u = X.grad
                proj = torch.sum(X * grad_u, dim=-1, keepdim=True)
                riemannian_grad = grad_u - proj * X

                if torch.max(torch.norm(riemannian_grad, dim=-1)) < 1e-7:
                    break

                direction = -riemannian_grad
                norm_d = torch.norm(direction, dim=-1, keepdim=True)
                X.copy_(X * torch.cos(lr_riemannian * norm_d) + (direction / (norm_d + 1e-12)) * torch.sin(lr_riemannian * norm_d))

        # 6. Branch-free Geodesic Folding: arccos(x_1)로 [0, pi] 물리 각도 즉시 복원
        final_theta_rad = torch.acos(torch.clamp(X[:, 0], min=-1.0 + 1e-7, max=1.0 - 1e-7))
        return final_theta_rad[0] if is_scalar else final_theta_rad

    def compute_angle(self, nu_center, nu_side1, nu_side2, nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=None):
        """외부 호환용 Degree(°) 단위 스칼라/텐서 반환 인터페이스"""
        theta_rad = self._compute_angle_tensor(nu_center, nu_side1, nu_side2, nu_b1, nu_b2, n_lig, sigma_bond)
        deg = torch.rad2deg(theta_rad)
        if deg.ndim == 0:
            return float(deg.item())
        return deg

   
# =====================================================================
# 🧪 [정식 검증] TDTManifoldSolver 고속 하이브리드 연산 벤치마크 메인
# =====================================================================

if __name__ == "__main__":
    import time

    engine = TDTManifoldSolver()
    print("═" * 80)
    print(f" TDTUltimateEngineV25 (DERIVATION_v2 Verification | Backend: {engine.device.upper()}) ")
    print("═" * 80)
    
    print("\n [1] 결합 길이 (Bond Length) 측정 스펙트럼 (12 Cases - 4·5주기 할로겐 포함)")
    print("-" * 80)
    cases_length = [
        # 분자명, 원자A, 원자B, 결합차수(nu_bond), 실제관측값(Å)
        ("H2",   1,  1, 1.0, 0.7414), 
        ("HF",   1,  9, 1.0, 0.9168), 
        ("HCl",  1, 17, 1.0, 1.2746),
        ("HBr",  1, 35, 1.0, 1.4140),  # 4주기 브롬 검증 추가
        ("HI",   1, 53, 1.0, 1.6080),  # 5주기 요오드 검증 추가
        ("LiF",  3,  9, 1.0, 1.5639), 
        ("CO",   6,  8, 3.0, 1.1283),
        ("N2",   7,  7, 3.0, 1.0977),
        ("NaF", 11,  9, 1.0, 1.9260),
        ("F2",   9,  9, 1.0, 1.4119),
        ("BF",   5,  9, 1.3, 1.2626),
        ("LiH",  3,  1, 1.0, 1.5959)
    ]
    
    for name, z_A, z_B, nu_b, obs in cases_length:
        calc, _, _ = engine.compute_bond_length(z_A, z_B, nu_b)
        print(f" - {name:5s} | Calc: {calc:.4f} Å | Obs: {obs:.4f} Å | Error: {calc-obs:+.4f} Å")
        
    print("\n [2] 결합 각도 (Bond Angle) 상전이 스펙트럼 (10 Cases - UFF-VSEPR 입체수 반영)")
    print("-" * 80)
    cases_angle = [
        # 분자명, 중심, 외각1, 외각2, 차수1, 차수2, 총리간드(n_lig), 총결합차수(sigma_bond), 관측값(°)
        ("CO2",  6,  8,  8, 2.0, 2.0, 2.0, 4.0, 180.00),
        ("CH4",  6,  1,  1, 1.0, 1.0, 4.0, 4.0, 109.47), 
        ("H2O",  8,  1,  1, 1.0, 1.0, 2.0, 2.0, 104.50),
        ("NH3",  7,  1,  1, 1.0, 1.0, 3.0, 3.0, 106.70), 
        ("H2S", 16,  1,  1, 1.0, 1.0, 2.0, 2.0,  92.11),
        ("PH3", 15,  1,  1, 1.0, 1.0, 3.0, 3.0,  93.50),
        ("NF3",  7,  9,  9, 1.0, 1.0, 3.0, 3.0, 102.30),
        ("OF2",  8,  9,  9, 1.0, 1.0, 2.0, 2.0, 103.30),
        ("SO2", 16,  8,  8, 1.5, 1.5, 2.0, 3.0, 119.30),
        ("ClO2",17,  8,  8, 1.5, 1.5, 2.0, 3.0, 111.00)
    ]
    
    for name, z_C, z_S1, z_S2, nu_b1, nu_b2, n_lig, sig_b, obs in cases_angle:
        calc = engine.compute_angle(z_C, z_S1, z_S2, nu_b1=nu_b1, nu_b2=nu_b2, n_lig=n_lig, sigma_bond=sig_b)
        print(f" - {name:5s} | Calc: {calc:.4f}° | Obs: {obs:.4f}° | Error: {calc-obs:+.4f}°")

    print("\n [3] 배치 텐서([B=10]) 병렬 연산 속도 및 일관성 검증")
    print("-" * 80)
    b_c   = torch.tensor([c[1] for c in cases_angle], dtype=torch.float64)
    b_s1  = torch.tensor([c[2] for c in cases_angle], dtype=torch.float64)
    b_s2  = torch.tensor([c[3] for c in cases_angle], dtype=torch.float64)
    b_nb1 = torch.tensor([c[4] for c in cases_angle], dtype=torch.float64)
    b_nb2 = torch.tensor([c[5] for c in cases_angle], dtype=torch.float64)
    b_nl  = torch.tensor([c[6] for c in cases_angle], dtype=torch.float64)
    b_sb  = torch.tensor([c[7] for c in cases_angle], dtype=torch.float64)

    t0 = time.perf_counter()
    batch_angles = engine.compute_angle(b_c, b_s1, b_s2, nu_b1=b_nb1, nu_b2=b_nb2, n_lig=b_nl, sigma_bond=b_sb)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f" ▶ 10대 분자 동시 배치 연산 완료 시간 : {elapsed_ms:.2f} ms")
    print(f" ▶ 배치 출력 텐서 (Degree)          : \n   {torch.round(batch_angles, decimals=2)}")
    print("═" * 80)
