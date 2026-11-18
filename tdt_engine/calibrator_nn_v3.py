import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractCalibrator(ABC):
    @abstractmethod
    def calibrate_bond_length(self, nu_A: torch.Tensor, nu_B: torch.Tensor, raw_R: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def calibrate_bond_angle(self, nu_center: torch.Tensor, nu_side1: torch.Tensor, nu_side2: torch.Tensor, 
                             raw_angle: torch.Tensor, nu_b1: torch.Tensor, nu_b2: torch.Tensor) -> torch.Tensor:
        """코어 엔진이 도출한 raw_angle을 연속 미분 가능한 유전 장 필터로 교정합니다."""
        pass


class NeuralResidualField(nn.Module):
    """
    제1원칙 코어의 뼈대를 보존하기 위한 초경량 잔차 오차장 신경망 (Calibrator).
    입력 피처는 기저 물리 상수를 오염시키지 않도록 원자번호 스케일의 독립된 직교 공간으로 매핑됩니다.
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            nn.Tanh(),  # 그라디언트 고속도로 보존을 위한 연속 미분 가능 활성화 함수
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=torch.float64)
        )
        # 초기화 가드레일: 초기 가중치를 극소화하여 훈련 전에는 코어 엔진의 계산값을 100% 보존하도록 유도
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1e-4, 1e-4)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

import torch.nn.functional as F
from tdt_engine.core_v2 import TDTEngineCore

class TDTNeuralCalibrator(AbstractCalibrator, nn.Module):
    def __init__(self, core_engine=None, device: str = 'cpu') -> None:
        super().__init__()
        self.core = core_engine if core_engine is not None else TDTEngineCore()
        self.device = device
        
        # DERIVATION_v2 Section 6: 물리 피처가 보강된 순수 잔차 신경망 (길이 5차원, 각도 8차원)
        self.length_calibrator = NeuralResidualField(input_dim=5, hidden_dim=16).to(self.device)
        self.angle_calibrator = NeuralResidualField(input_dim=8, hidden_dim=32).to(self.device)

        self.register_buffer('a0', torch.tensor(self.core.a0, dtype=torch.float64, device=self.device))
        self.register_buffer('omega_1', torch.tensor(self.core.omega_1, dtype=torch.float64, device=self.device))
        self.register_buffer('alpha', torch.tensor(self.core.alpha, dtype=torch.float64, device=self.device))
        self.register_buffer('T_cov', torch.tensor(self.core.T_cov, dtype=torch.float64, device=self.device))

    def _compute_rho_tensor(self, nu: torch.Tensor) -> torch.Tensor:
        """DERIVATION_v2 Section 1~2: 원자 번호 텐서로부터 연속 위상 코어 밀도(rho_C) 벡터 산출"""
        delta = self.alpha * self.omega_1
        nodes = torch.tensor([2.0, 10.0, 18.0, 36.0, 54.0], dtype=torch.float64, device=self.device)
        n_cont = 1.0 + torch.sum(0.5 + 0.5 * torch.tanh((nu.unsqueeze(-1) - (nodes + 0.5)) / delta), dim=-1)
        N_shell = torch.round(n_cont)

        idx = torch.clamp(torch.floor(nu).to(torch.long), min=1, max=86)
        r_cov = self.T_cov[idx]

        zeta = 1e-5 + torch.log(1.0 + (self.omega_1 * F.softplus(nu - 1.0)) / (2.0 * torch.pi))
        theta_shell = 0.5 + 0.5 * torch.tanh((N_shell - 1.5) / 0.1)
        a_C = r_cov * (1.0 + self.alpha * zeta * theta_shell) * torch.tanh((self.omega_1 * nu) / (self.alpha * (N_shell ** 2) * (torch.pi ** 2)))

        kepler = torch.pi / torch.sqrt(torch.tensor(18.0, dtype=torch.float64, device=self.device))
        return torch.tanh(((a_C / self.a0) ** 2) * kepler) * (1.0 + self.alpha * zeta), n_cont

    def calibrate_bond_length(self, nu_A: torch.Tensor, nu_B: torch.Tensor, raw_R: torch.Tensor, nu_bond: torch.Tensor = None) -> torch.Tensor:
        """DERIVATION_v2 Section 6.1: 하드 게이트 없는 순수 잔차 결합 길이 보정기"""
        if nu_bond is None:
            nu_bond = torch.ones_like(raw_R, dtype=torch.float64, device=self.device)

        nu_min = torch.minimum(nu_A, nu_B)
        nu_max = torch.maximum(nu_A, nu_B)

        rho_A, _ = self._compute_rho_tensor(nu_min)
        rho_B, _ = self._compute_rho_tensor(nu_max)
        delta_rho = torch.sqrt((rho_A - rho_B) ** 2 + 1e-8) / (rho_A + rho_B + 1e-8)

        features = torch.stack([nu_min, nu_max, nu_bond, raw_R, delta_rho], dim=-1)
        return raw_R + self.length_calibrator(features)


    def calibrate_bond_angle(self, nu_center: torch.Tensor, nu_side1: torch.Tensor, nu_side2: torch.Tensor, 
                             raw_angle: torch.Tensor, nu_b1: torch.Tensor = None, nu_b2: torch.Tensor = None,
                             sigma_bond: torch.Tensor = None, return_radians: bool = False) -> torch.Tensor:
        """
        [DERIVATION_v2 Section 6.2]
        하드 게이트를 완전히 제거하고 물리 피처(rho_C, rho_LP)와 logit-sigmoid 투영을 결합한
        범용 순수 잔차 결합 각도 보정기입니다.
        """
        if nu_b1 is None:
            nu_b1 = torch.ones_like(nu_center, dtype=torch.float64, device=self.device)
        if nu_b2 is None:
            nu_b2 = torch.ones_like(nu_center, dtype=torch.float64, device=self.device)
        if sigma_bond is None:
            sigma_bond = nu_b1 + nu_b2

        nu_s_min = torch.minimum(nu_side1, nu_side2)
        nu_s_max = torch.maximum(nu_side1, nu_side2)

        # 1. 중심 원자의 위상 코어 밀도(rho_C) 및 비공유 전자쌍 밀도(rho_LP) 산출
        rho_C, n_cont = self._compute_rho_tensor(nu_center)
        theta_2 = 0.5 + 0.5 * torch.tanh((n_cont - 1.5) / 0.1)
        theta_3 = 0.5 + 0.5 * torch.tanh((n_cont - 2.5) / 0.1)
        valence_electrons = nu_center - 2.0 * theta_2 - 8.0 * theta_3
        rho_LP = F.softplus((valence_electrons - sigma_bond) / 2.0, beta=50.0)

        # 2. 라디안(rad) 변환 및 8차원 물리 피처 구성
        raw_rad = raw_angle if return_radians else torch.deg2rad(raw_angle)
        features = torch.stack([nu_center, nu_s_min, nu_s_max, nu_b1, nu_b2, rho_C, rho_LP, raw_rad], dim=-1)
        neural_residual = self.angle_calibrator(features)

        # 3. 경계 안정화된 Logit-Sigmoid 투영 (eps_logit = 1e-6)
        eps_logit = 1e-6
        p_raw = torch.clamp(raw_rad / torch.pi, min=eps_logit, max=1.0 - eps_logit)
        calibrated_rad = torch.pi * torch.sigmoid(torch.logit(p_raw) + neural_residual)

        return calibrated_rad if return_radians else torch.rad2deg(calibrated_rad)



if __name__ == "__main__":
    print("═" * 80)
    print(" [V3 TEST] TDTNeuralCalibrator Pure Residual & Logit-Sigmoid Verification")
    print("═" * 80)

    calibrator = TDTNeuralCalibrator(core_engine=None, device='cpu')

    # 1. 결합 길이 순수 잔차(5차원 피처) 및 역전파 검증: [LiH, CO, HCl]
    nu_A = torch.tensor([1.0, 6.0, 1.0], dtype=torch.float64)
    nu_B = torch.tensor([3.0, 8.0, 17.0], dtype=torch.float64)
    nu_bond = torch.tensor([1.0, 3.0, 1.0], dtype=torch.float64)
    raw_R = torch.tensor([1.6233, 1.1223, 1.3086], dtype=torch.float64, requires_grad=True)

    calibrated_R = calibrator.calibrate_bond_length(nu_A, nu_B, raw_R, nu_bond=nu_bond)
    target_R = torch.tensor([1.5959, 1.1283, 1.2746], dtype=torch.float64)
    loss_R = torch.mean((calibrated_R - target_R) ** 2)
    loss_R.backward()

    print(" ▶ [1] 결합 길이 항등 보존 출력 (0-Epoch): \n   ", calibrated_R.detach())
    print(" ▶ [1] 결합 길이 역전파 그라디언트 (raw_R.grad): \n   ", raw_R.grad)
    print("-" * 80)

    # 2. 결합 각도 순수 잔차(8차원 피처 + Logit-Sigmoid) 검증: [CH4, CO2, H2O, H2S]
    nu_center = torch.tensor([6.0, 6.0, 8.0, 16.0], dtype=torch.float64)
    nu_side1  = torch.tensor([1.0, 8.0, 1.0, 1.0], dtype=torch.float64)
    nu_side2  = torch.tensor([1.0, 8.0, 1.0, 1.0], dtype=torch.float64)
    nu_b1     = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    nu_b2     = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    sigma_b   = torch.tensor([4.0, 4.0, 2.0, 2.0], dtype=torch.float64)
    raw_angle = torch.tensor([137.09, 161.21, 64.08, 42.79], dtype=torch.float64, requires_grad=True)

    calibrated_rad = calibrator.calibrate_bond_angle(
        nu_center, nu_side1, nu_side2, raw_angle,
        nu_b1=nu_b1, nu_b2=nu_b2, sigma_bond=sigma_b, return_radians=True
    )
    target_rad = torch.deg2rad(torch.tensor([109.47, 180.00, 104.50, 92.11], dtype=torch.float64))

    angle_loss = torch.mean((calibrated_rad - target_rad) ** 2)
    angle_loss.backward()

    print(" ▶ [2] 결합 각도 항등 보존 출력 (Degree): \n   ", torch.rad2deg(calibrated_rad.detach()))
    print(" ▶ [2] 결합 각도 역전파 그라디언트 (raw_angle.grad): \n   ", raw_angle.grad)
    print("-" * 80)

    if (raw_R.grad is not None and not torch.isnan(raw_R.grad).any() and
        raw_angle.grad is not None and not torch.isnan(raw_angle.grad).any()):
        print(" 🎉 [V3 GRADIENT HIGHWAY OPEN] 길이·각도 순수 잔차 역전파 보존.")
    else:
        print(" ❌ [GRAPH DETACHED] 연산 그래프가 단절되었습니다.")
    print("═" * 80)

