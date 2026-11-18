import torch
import torch.nn as nn
import torch.nn.functional as F

class TDTUltimateEngineV25(nn.Module):
    """
    [DERIVATION_v2.md 100% 준수]
    6주기 확장 위상 껍질, Read-Only 공유결합 반지름/전기음성도 텐서 버퍼,
    폐형식 결합 길이 및 배치 리만 SO(2) 다양체 솔버를 통합한 V25 엔진입니다.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # 1. 제1원칙 수론적 불변 상수 (Read-Only Buffers)
        self.register_buffer('a0', torch.tensor(0.5291772109, dtype=torch.float64, device=self.device))
        self.register_buffer('omega_1', torch.tensor(14.1347251417, dtype=torch.float64, device=self.device))
        self.register_buffer('alpha', torch.tensor(0.0072973525, dtype=torch.float64, device=self.device))

        # 2. 표준 공유결합 반지름 r_cov (Å, Z=0~86, Pyykkö/Cordero 기준)
        r_cov_list = [
            0.00, # 0: Dummy
            0.31, 0.28, # 1-2: H, He
            1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, # 3-10: Li-Ne
            1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, # 11-18: Na-Ar
            2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, # 19-36: K-Kr
            2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, # 37-54: Rb-Xe
            2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, # 55-72: Cs-Hf
            1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50 # 73-86: Ta-Rn
        ]
        
        # 3. 표준 폴링 전기음성도 chi_pauling (Z=0~86)
        chi_list = [
            0.00, # 0: Dummy
            2.20, 0.00, # 1-2: H, He
            0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00, # 3-10: Li-Ne
            0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00, # 11-18: Na-Ar
            0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00, # 19-36: K-Kr
            0.82, 0.95, 1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.10, 2.66, 2.60, # 37-54: Rb-Xe
            0.79, 0.89, 1.10, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18, 1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.30, # 55-72: Cs-Hf
            1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02, 2.00, 2.20, 2.20 # 73-86: Ta-Rn
        ]
        
        self.register_buffer('T_cov', torch.tensor(r_cov_list, dtype=torch.float64, device=self.device))
        self.register_buffer('T_chi', torch.tensor(chi_list, dtype=torch.float64, device=self.device))

    def _to_tensor(self, val):
        if isinstance(val, torch.Tensor):
            return val.to(dtype=torch.float64, device=self.device)
        return torch.tensor(val, dtype=torch.float64, device=self.device)

    def lookup_atomic_constants(self, nu):
        """분기문 없이 Read-Only 텐서 버퍼에서 공유결합 반지름과 전기음성도를 벡터 조회"""
        nu_t = self._to_tensor(nu)
        idx = torch.clamp(torch.floor(nu_t).to(torch.long), min=1, max=86)
        return self.T_cov[idx], self.T_chi[idx]

    def compute_automated_shell(self, nu_core):
        """DERIVATION_v2 Section 1: 1~6주기 연속 위상 껍질 산출 (배치 브로드캐스팅 지원)"""
        nu_t = self._to_tensor(nu_core)
        delta = self.alpha * self.omega_1
        critical_nodes = torch.tensor([2.0, 10.0, 18.0, 36.0, 54.0], dtype=torch.float64, device=self.device)
        
        phase_transitions = 0.5 + 0.5 * torch.tanh((nu_t.unsqueeze(-1) - (critical_nodes + 0.5)) / delta)
        n_shell_continuous = 1.0 + torch.sum(phase_transitions, dim=-1)
        return torch.round(n_shell_continuous), n_shell_continuous

    def compute_topological_density(self, nu_core):
        """DERIVATION_v2 Section 2: Softplus 리만 국소 밀도 및 공유결합 앵커 코어 반지름"""
        nu_t = self._to_tensor(nu_core)
        N_shell, _ = self.compute_automated_shell(nu_t)
        r_cov, _ = self.lookup_atomic_constants(nu_t)
        
        # 수론적 리만 국소 밀도 (nu -> 1 경계 매끄러운 처리)
        zeta_density = 1e-5 + torch.log(1.0 + (self.omega_1 * F.softplus(nu_t - 1.0)) / (2.0 * torch.pi))
        
        # 공유결합 반지름 앵커링 및 위상 껍질 변조
        theta_shell = 0.5 + 0.5 * torch.tanh((N_shell - 1.5) / 0.1)
        core_val = (self.omega_1 * nu_t) / (self.alpha * (N_shell ** 2) * (torch.pi ** 2))
        a_C = r_cov * (1.0 + self.alpha * zeta_density * theta_shell) * torch.tanh(core_val)
        
        # 유계화된 4차원 케플러 패킹 코어 밀도
        kepler_packing = torch.pi / torch.sqrt(torch.tensor(18.0, dtype=torch.float64, device=self.device))
        rho_C = torch.tanh(((a_C / self.a0) ** 2) * kepler_packing) * (1.0 + self.alpha * zeta_density)
        
        return rho_C, a_C, zeta_density


    def _compute_bond_length_tensor(self, nu_A, nu_B, nu_bond=1.0):
        """
        [DERIVATION_v2 Section 3 - 내부 텐서 그래프 전용]
        200-step 경사하강법 루프와 if-else 분기를 완전히 제거한 단일 패스 폐형식(Closed-Form) 결합 길이 연산기.
        스칼라 및 1D 배치 텐서 [B]를 동시에 지원합니다.
        """
        nu_A_t = self._to_tensor(nu_A)
        nu_B_t = self._to_tensor(nu_B)
        nu_bond_t = self._to_tensor(nu_bond)
        eps = 1e-8

        rho_A, a_A, _ = self.compute_topological_density(nu_A_t)
        rho_B, a_B, _ = self.compute_topological_density(nu_B_t)
        _, chi_A = self.lookup_atomic_constants(nu_A_t)
        _, chi_B = self.lookup_atomic_constants(nu_B_t)

        # 1. 연속 위상 비대칭 텐서 및 이온성 수축 조절자 (절댓값 첨점 제거)
        asymmetry_factor = ((nu_A_t - nu_B_t) / (nu_A_t + nu_B_t + eps)) ** 2
        delta_rho = torch.sqrt((rho_A - rho_B) ** 2 + eps) / (rho_A + rho_B + eps)
        scale_invariant = torch.tanh(rho_A * rho_B)

        ionic_field_modifier = self.alpha * torch.sqrt(self.omega_1) * delta_rho * scale_invariant
        ionic_contraction = (1.0 - ionic_field_modifier) / (1.0 + self.alpha * asymmetry_factor)

        geometric_ionic_limit = 1.0 + (self.alpha * (self.omega_1 / torch.pi))
        universal_ionic_scale = 1.0 + (geometric_ionic_limit - 1.0) * torch.tanh(delta_rho / self.alpha)

        chi_tensor = (1.0 - self.alpha * asymmetry_factor * self.omega_1 * (torch.pi / 2.0)) * \
                     ionic_contraction * universal_ionic_scale

        # 2. 다중 결합 압축(B_scale) 및 수소화물 이완 게이트(H_gate) - Branchless 연산
        bond_order_damp = 1.0 / (1.0 + self.alpha * self.omega_1 * (nu_bond_t - 1.0))
        triple_bond_gate = 1.0 - (self.alpha * self.omega_1 / torch.pi) * (0.5 + 0.5 * torch.tanh((nu_bond_t - 1.5) / 0.05))
        B_scale = bond_order_damp * triple_bond_gate

        nu_min = torch.minimum(nu_A_t, nu_B_t)
        nu_max = torch.maximum(nu_A_t, nu_B_t)
        N_max, _ = self.compute_automated_shell(nu_max)

        H_gate = (0.5 + 0.5 * torch.tanh((1.5 - nu_min) / 0.05)) * \
                 (0.5 + 0.5 * torch.tanh((nu_max - 1.5) / 0.05))
        # 1. _compute_bond_length_tensor 내부: 수소화물 과팽창 억제
        H_expansion = 1.0 + H_gate * self.alpha * asymmetry_factor * self.omega_1


        # 3. 변조된 코어 반지름 및 쇼메이커-스티븐슨 전기음성도 수축 결합 (단일 패스 폐형식)
        a_A_mod = a_A * B_scale * H_expansion * chi_tensor
        a_B_mod = a_B * B_scale * H_expansion * chi_tensor

        c_SS = 0.08  # 표준 Schomaker-Stevenson 전기음성도 수축 상수 (Å)
        electronegativity_diff = torch.sqrt((chi_A - chi_B) ** 2 + eps)
        R_bond = (a_A_mod + a_B_mod) - c_SS * electronegativity_diff * (1.0 - H_gate)

        return R_bond, a_A_mod, a_B_mod

    def compute_bond_length(self, nu_A, nu_B, nu_bond=1.0):
        """외부 호환용 스칼라 튜플 반환 인터페이스"""
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

        rho_C, a_C_raw, _ = self.compute_topological_density(nu_c_t)
        _, a_S1_raw, _ = self.compute_topological_density(nu_s1_t)
        _, a_S2_raw, _ = self.compute_topological_density(nu_s2_t)
        Phi_C = torch.clamp(1.0 - (a_C_raw / self.a0), min=1e-5)

        # 2. 연속 가전자 및 비공유 전자쌍(rho_LP) 연산 (if-else 완전 제거)
        _, N_shell_cont = self.compute_automated_shell(nu_c_t)
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
        final_theta_rad = torch.acos(torch.clamp(X[:, 0].detach(), min=-1.0 + 1e-7, max=1.0 - 1e-7))
        return final_theta_rad[0] if is_scalar else final_theta_rad

    def compute_angle(self, nu_center, nu_side1, nu_side2, nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=None):
        """외부 호환용 Degree(°) 단위 스칼라/텐서 반환 인터페이스"""
        theta_rad = self._compute_angle_tensor(nu_center, nu_side1, nu_side2, nu_b1, nu_b2, n_lig, sigma_bond)
        deg = torch.rad2deg(theta_rad)
        if deg.ndim == 0:
            return float(deg.item())
        return deg



# =====================================================================
# 🧪 TDT Ultimate Engine V25 통합 검증 시스템 (스펙트럼 + 4/5주기 + 배치 가속)
# =====================================================================
if __name__ == "__main__":
    import time

    engine = TDTUltimateEngineV25()
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
