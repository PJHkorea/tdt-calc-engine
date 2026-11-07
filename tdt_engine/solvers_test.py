import torch
import numpy as np
from tdt_engine.core import TDTEngineCore

class TDTManifoldSolver:
    """
    [V24 혁신] PyTorch Acceleration Backend & SO(2) Manifold Solver
    core.py의 순수 제1원칙 수식을 기반으로 오직 수치해석적 공간 평형점만 추적합니다.
    """
    def __init__(self, core_engine: TDTEngineCore = None, device: str = None):
        # 1. 제1원칙 데이터 필드를 생산하는 core.py 엔진 의존성 주입 (DI)
        self.core = core_engine if core_engine else TDTEngineCore()
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 2. 우주 고정 상수 및 수론적 불변량을 PyTorch 장치 텐서로 고정
        self.a0 = torch.tensor(self.core.a0, dtype=torch.float64, device=self.device)
        self.omega_1 = torch.tensor(self.core.omega_1, dtype=torch.float64, device=self.device)
        self.alpha = torch.tensor(self.core.alpha, dtype=torch.float64, device=self.device)

    def _get_core_tensors(self, nu_core: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [V24_ext 중복 박멸 장치] 
        core.py가 유도한 순수 제1원칙 물리량 상수를 받아 연산 그래프 단절 없이 
        PyTorch 텐서 필드로 실시간 기하학적 미러링을 수행합니다.
        """
        # core.py 고유의 NumPy/float 연산 메커니즘을 최초 1회 그대로 신뢰하고 연동
        rho_val, a_val, zeta_val = self.core.compute_topological_density(nu_core)
        
        # 최적화 솔버 내부에서 미분이 정상 가속될 수 있도록 텐서 장치로 안전하게 수용
        rho_C = torch.tensor(rho_val, dtype=torch.float64, device=self.device)
        a_C = torch.tensor(a_val, dtype=torch.float64, device=self.device)
        zeta_density = torch.tensor(zeta_val, dtype=torch.float64, device=self.device)
        
        return rho_C, a_C, zeta_density

    def compute_automated_shell_tensor(self, nu_core: float) -> tuple[torch.Tensor, torch.Tensor]:
        """core.py 내부의 연속 껍질 함수 출력값을 텐서 연산 추적 그래프로 브릿지합니다."""
        N_round, n_cont = self.core.compute_automated_shell(nu_core)
        
        return (
            torch.tensor(N_round, dtype=torch.float64, device=self.device),
            torch.tensor(n_cont, dtype=torch.float64, device=self.device)
        )





    def compute_bond_length(self, nu_A, nu_B, nu_bond=1.0):
        rho_A, a_A, delta_zeta_A = self.compute_topological_density(nu_A)
        rho_B, a_B, delta_zeta_B = self.compute_topological_density(nu_B)
        
        # [수정] 파이썬 상수를 안전하게 텐서화
        sum_nu = nu_A + nu_B
        asymmetry_factor = torch.tensor(((nu_A - nu_B) / sum_nu) ** 2 if sum_nu > 0 else 0.0, dtype=torch.float64, device=self.device)
        chi_tensor = 1.0 - (self.alpha * asymmetry_factor * self.omega_1 * (torch.pi / 2.0))
        
        # [수정] 텐서 그래프 단절 완전 차단 (torch.where 도입)
        sum_rho = rho_A + rho_B
        density_asymmetry = torch.where(
            sum_rho > 0.0,
            torch.abs(rho_A - rho_B) / sum_rho,
            torch.tensor(0.0, dtype=torch.float64, device=self.device)
        )
        scale_invariant = torch.tanh(rho_A * rho_B)
        
        ionic_field_modifier = (self.alpha * (self.omega_1 ** 0.5) * density_asymmetry * scale_invariant)
        ionic_contraction = (1.0 - ionic_field_modifier) / (1.0 + self.alpha * asymmetry_factor)
        
        geometric_ionic_limit = 1.0 + (self.alpha * (self.omega_1 / torch.pi))
        universal_ionic_scale = 1.0 + (geometric_ionic_limit - 1.0) * torch.tanh(density_asymmetry / self.alpha)
        chi_tensor = chi_tensor * (ionic_contraction * universal_ionic_scale)
        
        Phi_A = 1.0 - (a_A / self.a0) if nu_A > 1 else torch.tensor(1.0, dtype=torch.float64, device=self.device)
        Phi_B = 1.0 - (a_B / self.a0) if nu_B > 1 else torch.tensor(1.0, dtype=torch.float64, device=self.device)
        
        bond_scale = 1.0 / (1.0 + self.alpha * self.omega_1 * (nu_bond - 1.0))
        if nu_bond > 1.5: 
            bond_scale *= (1.0 - self.alpha * (self.omega_1 / torch.pi))
        
        shielding_factor = (self.alpha * self.omega_1) / (2.0 * torch.pi)
        mod_A = (1.0 + (self.alpha * nu_bond) - (shielding_factor * (torch.tensor(nu_A, device=self.device) - 1.0) * Phi_A)) * bond_scale
        mod_B = (1.0 + (self.alpha * nu_bond) - (shielding_factor * (torch.tensor(nu_B, device=self.device) - 1.0) * Phi_B)) * bond_scale
        
        if (nu_A == 1 or nu_B == 1) and (nu_A > 1 or nu_B > 1):
            max_nu = max(nu_A, nu_B)
            N_max, _ = self.compute_automated_shell(max_nu)
            hydride_expansion = 1.0 + (asymmetry_factor * (1.0 - scale_invariant) * (self.omega_1 / (N_max * torch.pi)))
            mod_A *= hydride_expansion
            mod_B *= hydride_expansion

        if nu_A >= 17 or nu_B >= 17:
            max_nu = max(nu_A, nu_B)
            universal_d_shield = self.alpha / self.omega_1
            mass_load_shield = 0.5 + universal_d_shield * torch.tanh((torch.tensor(max_nu, device=self.device) - 17.0) / self.omega_1)
            chi_tensor *= mass_load_shield
            
        a_A_mod = a_A * mod_A * chi_tensor
        a_B_mod = a_B * mod_B * chi_tensor
        
        N_A, _ = self.compute_automated_shell(nu_A)
        N_B, _ = self.compute_automated_shell(nu_B)
        shell_trigger_A = 0.5 + 0.5 * torch.tanh((N_A - 1.5) / 0.1)
        shell_trigger_B = 0.5 + 0.5 * torch.tanh((N_B - 1.5) / 0.1)
        
        geometric_expansion_limit = 1.0 + (self.alpha * self.omega_1 * (torch.sqrt(torch.tensor(18.0, device=self.device)) / torch.pi))
        expansion_dim_A = (geometric_expansion_limit - 1.0) * shell_trigger_A
        expansion_dim_B = (geometric_expansion_limit - 1.0) * shell_trigger_B
        
        delta = self.alpha * self.omega_1
        M_log_argument = 1.0 + (delta_zeta_A + delta_zeta_B) * delta
        dimensionless_coupling_dim = self.alpha * (self.omega_1 / torch.pi) / (1.0 + expansion_dim_A + expansion_dim_B)
        
        metric_tensor_modifier = 1.0 + torch.log(M_log_argument) * dimensionless_coupling_dim
        
                # 🚀 [수정] 초기 추측값 복사 후 최적화 궤적 안전화 (lr 다운사이징 및 에폭 상향 준비)
        R_bond = (a_A_mod + a_B_mod).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([R_bond], lr=0.005) # 0.05 -> 0.005 안정화

        # 🧬 [복원 완완] 경사하강법을 통한 수리 물리적 최소 에너지 마디선 추적 루프
        for _ in range(200):
            optimizer.zero_grad()
            # R_bond 수축 방지 안전 클램프 장치
            R_clipped = torch.clamp(R_bond, min=0.1)
            left_term = torch.log(a_A_mod * a_B_mod) - metric_tensor_modifier * (2.0 * torch.log(R_clipped))
            right_term = -R_clipped / (a_A_mod + a_B_mod)
            
            # 최소화 대상인 잔차 제곱합 정의
            residual = (left_term - right_term) ** 2
            residual.backward()
            optimizer.step()
            
        # 🧬 [복원 완료] 메인 시스템 튜플 언패킹을 위한 float 데이터 정상 반환문
        return float(R_bond.detach().item()), float(a_A_mod.detach().item()), float(a_B_mod.detach().item())

    def _compute_angle_tensor(self, nu_center: float, nu_side1: float, nu_side2: float, nu_b1: float = 1.0, nu_b2: float = 1.0) -> torch.Tensor:
        """
        [내부 연산 전용 - 리만 SO(2) 다양체 최적화 코어]
        2단계의 내부 결합 길이 텐서 그래프를 그대로 직결받아 
        메모리 카피 및 그래프 단절 없이 리만 다양체 측지선 수축 최적화를 수행합니다.
        """
        # 1. 상위 텐서 연속 그래프 컨텍스트 완벽 직결 (Part 2의 텐서 출력 수용)
        R_A, a_A, _ = self._compute_bond_length_tensor(nu_center, nu_side1, nu_b1)
        R_B, a_B, _ = self._compute_bond_length_tensor(nu_center, nu_side2, nu_b2)
        
        # 1단계의 중복 박멸 인터페이스 연동
        rho_C, a_C_raw, _ = self._get_core_tensors(nu_center)
        Phi_C = torch.clamp(1.0 - (a_C_raw / self.a0), min=1e-5)
        
        theta_tet = torch.tensor(109.4712 * torch.pi / 180.0, dtype=torch.float64, device=self.device)
        theta_linear = torch.pi
        
        kappa_4 = (torch.pi ** 2) / 16.0
        bifurcation_scaler = kappa_4 * self.omega_1 * (1.0 + self.alpha * nu_center)
        bifurcation_score = torch.tanh(rho_C / bifurcation_scaler)

        target_angle = theta_tet + (theta_linear - theta_tet) * (1.0 - bifurcation_score)
        
        # 외각 원소 중복 수식 제거 후 텐서 브릿지 연동
        _, a_S1_raw, _ = self._get_core_tensors(nu_side1)
        _, a_S2_raw, _ = self._get_core_tensors(nu_side2)
        
        Phi_side1 = 1.0 - torch.tanh(a_S1_raw / self.a0) if nu_side1 > 1 else torch.tensor(1.0, dtype=torch.float64, device=self.device)
        Phi_side2 = 1.0 - torch.tanh(a_S2_raw / self.a0) if nu_side2 > 1 else torch.tensor(1.0, dtype=torch.float64, device=self.device)
        Phi_side_tensor = 0.5 * (Phi_side1 + Phi_side2)

        _, N_shell_cont = self.compute_automated_shell_tensor(nu_center)
        valence_electrons = torch.tensor(nu_center - 2.0 if N_shell_cont <= 2.05 else nu_center - 10.0, dtype=torch.float64, device=self.device)
        
        raw_lone_pair = (valence_electrons - (nu_b1 + nu_b2)) / 2.0
        smooth_lone_pair_density = raw_lone_pair * (0.5 + 0.5 * torch.tanh(raw_lone_pair / 0.01))
        if raw_lone_pair <= 1e-5:
            smooth_lone_pair_density = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        electronegativity_asymmetry = torch.tensor(abs(nu_center - 0.5 * (nu_side1 + nu_side2)) / (nu_center + 0.5 * (nu_side1 + nu_side2)), dtype=torch.float64, device=self.device)
        fluorine_polar_shield = 1.0 + (self.alpha * (self.omega_1 ** 0.5) * (1.0 - Phi_side_tensor) * torch.tanh(electronegativity_asymmetry))
        Phi_side_tensor = Phi_side_tensor * fluorine_polar_shield

        vsepr_scaler = 1.0 - (self.alpha * self.omega_1)  
        harmonic_amplitude = self.alpha * (self.omega_1 / torch.pi) * (1.0 + self.alpha * electronegativity_asymmetry)
        vsepr_shrunk = harmonic_amplitude * torch.sqrt(smooth_lone_pair_density) * self.omega_1 / (Phi_side_tensor ** vsepr_scaler)
        
        # 🧬 [V24_ext 핵심 보정]: 중원소 유전 장 폭발로 인한 180도 선형 오버슈트 진압용 대수적 지수 감쇄 장 장착
        heavy_element_damping = torch.exp(-self.alpha * torch.tensor(nu_center, dtype=torch.float64, device=self.device))
        vsepr_shrunk = vsepr_shrunk * heavy_element_damping
        
        target_angle -= vsepr_shrunk * (1.0 - bifurcation_score)
        
        N_shell, _ = self.compute_automated_shell_tensor(nu_center)
        shell_3_trigger = 0.5 + 0.5 * torch.tanh((N_shell - 2.5) / 0.1) * (0.5 + 0.5 * torch.tanh((3.5 - N_shell) / 0.1))
        
        kepler_packing = torch.pi / torch.sqrt(torch.tensor(18.0, dtype=torch.float64, device=self.device))
        
        # 🧬 [V24_ext 완충 공간 추가]: 분모 싱귤래리티를 방지하기 위한 alpha * omega_1 마디선 보강
        h2s_theoretical_target = target_angle - (self.alpha * self.omega_1 * ((2.0 * kepler_packing + self.alpha) / (Phi_side_tensor + self.alpha * self.omega_1)))
        target_angle = target_angle * (1.0 - shell_3_trigger) + h2s_theoretical_target * shell_3_trigger

        # 🚀 [리만 다양체 이식] SO(2) 원형 다양체 매핑 변수 선언
        x = torch.tensor([0.0, 1.0], dtype=torch.float64, device=self.device, requires_grad=True)
        lr_riemannian = 0.01
        
        for step in range(1200):
            theta_current = torch.atan2(x[1], x[0])
            
            if x.grad is not None:
                x.grad.zero_()
                
            restoring_term = (theta_current - target_angle) ** 2
            sin_sq = torch.sin(theta_current) ** 2
            cos_sq = torch.cos(theta_current) ** 2
            
            S_theta = sin_sq * torch.exp(-(self.alpha * self.omega_1 * (nu_b1 + nu_b2)) / (1.0 + cos_sq))
            harmonic_damping = self.alpha * (torch.cos(5.0 * theta_current) + (self.alpha * self.omega_1) * torch.cos(6.0 * theta_current))
            fused_term = 1.0 + harmonic_damping
            
            nu_b_factor = 0.5 * (nu_b1 + nu_b2)
            spatial_distortion = (1.0 - Phi_C) * (1.0 - Phi_side_tensor) * torch.tanh(((nu_b_factor - 1.0) / self.alpha))
            if nu_b_factor > 1.2:
                spatial_distortion = spatial_distortion * 0.0
            
            fused_term = fused_term * (1.0 + (self.alpha ** 1.5) * self.omega_1 * spatial_distortion * sin_sq)
            S_density = S_theta * fused_term
            
            F_dielectric_leak = (1.0 - Phi_side_tensor) * (0.5 + 0.5 * torch.tanh((torch.tensor(nu_center, device=self.device) - 5.5) / 0.1))
            S_density = S_density / (1.0 + (self.alpha ** 0.5) * (1.0 - S_density) * F_dielectric_leak)

            R_AB_opt = (a_A + a_B) * (1.0 + (1.0 - S_density) * (self.omega_1 / torch.pi) * Phi_C)
            R_AB_geo = torch.sqrt(torch.abs((R_A**2) + (R_B**2) - 2.0 * R_A * R_B * torch.cos(theta_current)))

            dimensionless_space_residual = torch.abs(R_AB_opt - R_AB_geo) / (R_A + R_B)
            dimensionless_angle_residual = (rho_C * restoring_term) / (torch.pi ** 2)
            
            background_tension = torch.log(1.0 + rho_C + Phi_side_tensor)
            self_consistent_coupling = ((1.0 / self.alpha) * (rho_C ** (1.0 - vsepr_scaler)) * 
                                        (self.omega_1 / (4.0 * torch.pi)) * (vsepr_scaler ** 2) * background_tension)
            
            pauli_geometrical_barrier = ((a_A + a_B) * self.alpha * self.omega_1) / (sin_sq + 1e-9)
            
            loss = dimensionless_space_residual + dimensionless_angle_residual * self_consistent_coupling + pauli_geometrical_barrier
            loss.backward()
            
            # 🧬 [리만 사영 & 지지선 레트랙션 수렴 엔진]
            with torch.no_grad():
                grad_u = x.grad
                proj_factor = torch.dot(x, grad_u)
                riemannian_grad = grad_u - proj_factor * x
                
                if torch.norm(riemannian_grad) < 1e-7:
                    break
                    
                direction = -riemannian_grad
                norm_d = torch.norm(direction)
                
                if norm_d > 0:
                    x.copy_(x * torch.cos(lr_riemannian * norm_d) + 
                            (direction / norm_d) * torch.sin(lr_riemannian * norm_d))
                    
        final_theta = torch.atan2(x[1], x[0])
        if final_theta < 0:
            final_theta = final_theta + 2.0 * torch.pi
            
        with torch.no_grad():
            final_theta.clamp_(min=0.001, max=torch.pi - 0.001)
            
        return final_theta

    def compute_angle(self, nu_center: float, nu_side1: float, nu_side2: float, nu_b1: float = 1.0, nu_b2: float = 1.0) -> float:
        """
        [외부 공개용 인터페이스]
        추상 규격을 충족하며 최종적으로 호환성 높은 float 단위(도, Degree) 데이터를 정상 출력합니다.
        """
        final_theta_tensor = self._compute_angle_tensor(nu_center, nu_side1, nu_side2, nu_b1, nu_b2)
        return float(torch.rad2deg(final_theta_tensor).detach().item())

