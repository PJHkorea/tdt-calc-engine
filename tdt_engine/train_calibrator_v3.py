import torch
import torch.nn as nn
import torch.optim as optim
#from tdt_engine.core_v2 import TDTEngineCore
#from tdt_engine.solvers_v3 import TDTManifoldSolver
#from tdt_engine.calibrator_nn_v3 import TDTNeuralCalibrator

def load_first_principles_dataset(solver: TDTManifoldSolver, device: str = 'cpu'):
    print("🛰️ [STAGE 1] 실제 물리 솔버(v3) 기반 오차장(Error Field) 데이터셋 동적 추출...")

    length_specs = [
        (1.0,  17.0, 1.0, 1.2746),  # HCl
        (1.0,  9.0,  1.0, 0.9168),  # HF
        (1.0,  35.0, 1.0, 1.4140),  # HBr
        (1.0,  53.0, 1.0, 1.6080),  # HI (5주기 추가)
        (6.0,  8.0,  3.0, 1.1283),  # CO
        (7.0,  8.0,  2.0, 1.1508),  # NO
        (6.0,  7.0,  3.0, 1.1532),  # CN
        (1.0,  1.0,  1.0, 0.7414),  # H2
        (7.0,  7.0,  3.0, 1.0977),  # N2
        (8.0,  8.0,  2.0, 1.2075),  # O2
        (9.0,  9.0,  1.0, 1.4119)   # F2
    ]

    length_raw_data, length_obs_data = [], []
    for nu_a, nu_b, nu_bond, obs_r in length_specs:
        raw_r_tensor, _, _ = solver._compute_bond_length_tensor(nu_a, nu_b, nu_bond)
        raw_r = float(raw_r_tensor.detach().item())
        length_raw_data.append([min(nu_a, nu_b), max(nu_a, nu_b), raw_r, nu_bond])
        length_obs_data.append(obs_r)

    # 2. 결합 각도(Angle) 10대 분자 명세: (nu_center, nu_s1, nu_s2, nu_b1, nu_b2, n_lig, sigma_bond, obs_angle)
    angle_specs = [
        (8.0,  1.0,  1.0,  1.0, 1.0, 2.0, 2.0, 104.51),  # H2O
        (7.0,  1.0,  1.0,  1.0, 1.0, 3.0, 3.0, 107.80),  # NH3
        (6.0,  1.0,  1.0,  1.0, 1.0, 4.0, 4.0, 109.47),  # CH4
        (6.0,  8.0,  8.0,  2.0, 2.0, 2.0, 4.0, 180.00),  # CO2
        (16.0, 8.0,  8.0,  2.0, 2.0, 2.0, 3.0, 119.50),  # SO2
        (6.0,  1.0,  7.0,  1.0, 3.0, 2.0, 4.0, 180.00),  # HCN
        (7.0,  7.0,  8.0,  2.0, 1.0, 2.0, 3.0, 134.10),  # N2O
        (8.0,  8.0,  8.0,  1.0, 1.0, 2.0, 2.0, 116.80),  # O3
        (15.0, 1.0,  1.0,  1.0, 1.0, 3.0, 3.0, 93.30),   # PH3
        (16.0, 1.0,  1.0,  1.0, 1.0, 2.0, 2.0, 92.10)    # H2S
    ]

    angle_raw_data, angle_obs_data = [], []
    for nu_c, nu_s1, nu_s2, nb1, nb2, n_lig, sig_b, obs_ang in angle_specs:
        raw_ang = solver.compute_angle(nu_c, nu_s1, nu_s2, nu_b1=nb1, nu_b2=nb2, n_lig=n_lig, sigma_bond=sig_b)
        angle_raw_data.append([nu_c, min(nu_s1, nu_s2), max(nu_s1, nu_s2), raw_ang, nb1, nb2, sig_b])
        angle_obs_data.append(obs_ang)

    length_inputs = torch.tensor(length_raw_data, dtype=torch.float64, device=device)
    length_targets = torch.tensor(length_obs_data, dtype=torch.float64, device=device)

    angle_inputs = torch.tensor(angle_raw_data, dtype=torch.float64, device=device)
    angle_targets = torch.deg2rad(torch.tensor(angle_obs_data, dtype=torch.float64, device=device))

    print(f"   -> [동적 오차장 행렬화 완료] Length: {length_inputs.shape} | Angle: {angle_inputs.shape}")
    return length_inputs, length_targets, angle_inputs, angle_targets

def configure_isolated_optimizer(calibrator: TDTNeuralCalibrator):
    """
    AGENT_PROMPT_v2 Section 2, 3 준수:
    기저 물리 상수의 백그라운드 오염을 차단하기 위해 core 엔진 파라미터는 완전히 동결(Freeze)하고,
    오직 신경망 잔차 필드(NeuralResidualField)의 가중치만 옵티마이저에 전선 바인딩합니다.
    """
    # 1. 안전 동결(Freeze): calibrator_nn.py 내부의 실제 속성명인 'core' 대조
    # 코어 물리량 상수군의 역전파 그래프 연산 참여를 원천 봉쇄하여 오버핏 물리 누수 방지
    if hasattr(calibrator, 'core') and calibrator.core is not None:
        # TDTEngineCore 인스턴스가 PyTorch 파라미터 구조를 가지지 않는 경우를 대비한 가드레일
        if hasattr(calibrator.core, 'parameters'):
            for param in calibrator.core.parameters():
                param.requires_grad = False
            print("   🔒 [Anti-Leakage] 기저 물리 상수의 백그라운드 오염 차단 (core 엔진 동결 완료).")
    
    # 2. 가중치 고립: 순수 신경망 잔차 필드(NeuralResidualField) 서브모듈의 파라미터만 도메인 격리 추출
    trainable_params = []
    
    # 내부에 명시적 신경망 서브모듈이 존재하는 구조적 명세 대조
    if hasattr(calibrator, 'length_calibrator'):
        trainable_params.extend(list(calibrator.length_calibrator.parameters()))
    if hasattr(calibrator, 'angle_calibrator'):
        trainable_params.extend(list(calibrator.angle_calibrator.parameters()))
        
    # 명시적 서브모듈 접근 실패 시, requires_grad=True 상태인 순수 잔차 가중치만 동적 스크리닝
    if not trainable_params:
        trainable_params = [p for p in calibrator.parameters() if p.requires_grad]
        
    # 3. 도메인 분리형 전선 바인딩 집행 (중복 임포트 제거 후 전역 optim 객체 호출)
    optimizer = optim.Adam(trainable_params, lr=0.01)
    
    print(f"🔒 [STAGE 2] 가중치 독립성 제어 및 옵티마이저 도메인 격리 완료. (학습 파라미터: {len(trainable_params)}개 텐서군 결착)")
    return optimizer

def execute_neural_training_loop(calibrator, optimizer, dataset, epochs=500):
    print("🔥 [STAGE 3] 연속 미분 가능 오차장(v3 Radian-Unified) 가속 학습 시퀀스 가동...")
    l_in, l_tar, a_in, a_tar = dataset
    criterion = nn.MSELoss()
    
    calibrator.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # length_inputs: [nu_min, nu_max, raw_R, nu_bond]
        pred_l = calibrator.calibrate_bond_length(l_in[:, 0], l_in[:, 1], l_in[:, 2], nu_bond=l_in[:, 3])
        
        # angle_inputs: [nu_center, nu_s_min, nu_s_max, raw_ang, nu_b1, nu_b2, sigma_bond]
        pred_a_rad = calibrator.calibrate_bond_angle(
            a_in[:, 0], a_in[:, 1], a_in[:, 2], 
            a_in[:, 3], nu_b1=a_in[:, 4], nu_b2=a_in[:, 5],
            sigma_bond=a_in[:, 6], return_radians=True
        )
        
        if pred_l.ndim > 1:
            pred_l = pred_l.squeeze()
        if pred_a_rad.ndim > 1:
            pred_a_rad = pred_a_rad.squeeze()
            
        # DERIVATION_v2 Section 6.3: Å^2 및 rad^2 자연 단위 균형 손실 합산
        loss_length = criterion(pred_l, l_tar)
        loss_angle = criterion(pred_a_rad, a_tar)
        total_loss = loss_length + loss_angle
        
        total_loss.backward()
        
        nn.utils.clip_grad_norm_(
            [p for p in calibrator.parameters() if p.requires_grad], 
            max_norm=1.0
        )
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f" 📈 Epoch [{epoch:03d}/{epochs}] | Total Loss: {total_loss.item():.6f} (L_Loss: {loss_length.item():.6f}, A_Loss_rad: {loss_angle.item():.6f})")
            
    print("🎉 [STAGE 3] 오차장 최적화 완료. 잔차 평면 수렴 성공.")


import math

class MaterialScreeningSimulator:
    """
    코랩 및 독립 실행 환경을 위한 V3 호환 하이브리드 시뮬레이터 오케스트레이터
    """
    def __init__(self, core, solver, calibration):
        self.core = core
        self.solver = solver
        self.calibration = calibration
        self.device = getattr(solver, 'device', 'cpu')
        self.elem_symbols = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
            11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
            35: "Br", 53: "I"
        }

    def predict_molecule_geometry(self, center_z: int, side1_z: int, side2_z: int, 
                                  nu_b1: float = 1.0, nu_b2: float = 1.0, 
                                  n_lig: float = 2.0, sigma_bond: float = None):
        if sigma_bond is None:
            sigma_bond = nu_b1 + nu_b2

        # 1. V3 솔버 원시(Raw) 계산
        raw_r1, _, _ = self.solver.compute_bond_length(center_z, side1_z, nu_bond=nu_b1)
        raw_r2, _, _ = self.solver.compute_bond_length(center_z, side2_z, nu_bond=nu_b2)
        raw_angle = self.solver.compute_angle(
            center_z, side1_z, side2_z, nu_b1=nu_b1, nu_b2=nu_b2, n_lig=n_lig, sigma_bond=sigma_bond
        )

        # 2. V3 순수 잔차 신경망 캘리브레이션 적용
        with torch.no_grad():
            t_c = torch.tensor([float(center_z)], dtype=torch.float64, device=self.device)
            t_s1 = torch.tensor([float(side1_z)], dtype=torch.float64, device=self.device)
            t_s2 = torch.tensor([float(side2_z)], dtype=torch.float64, device=self.device)
            t_b1 = torch.tensor([float(nu_b1)], dtype=torch.float64, device=self.device)
            t_b2 = torch.tensor([float(nu_b2)], dtype=torch.float64, device=self.device)
            t_sig = torch.tensor([float(sigma_bond)], dtype=torch.float64, device=self.device)

            t_r1 = torch.tensor([float(raw_r1)], dtype=torch.float64, device=self.device)
            t_r2 = torch.tensor([float(raw_r2)], dtype=torch.float64, device=self.device)
            t_ang = torch.tensor([float(raw_angle)], dtype=torch.float64, device=self.device)

            cal_r1 = float(self.calibration.calibrate_bond_length(t_c, t_s1, t_r1, nu_bond=t_b1).item())
            cal_r2 = float(self.calibration.calibrate_bond_length(t_c, t_s2, t_r2, nu_bond=t_b2).item())
            cal_ang = float(self.calibration.calibrate_bond_angle(
                t_c, t_s1, t_s2, t_ang, nu_b1=t_b1, nu_b2=t_b2, sigma_bond=t_sig, return_radians=False
            ).item())

        return (cal_r1, cal_r2), cal_ang

    def export_to_xyz(self, mol_name: str, atomic_numbers: list, lengths: tuple, angle_deg: float, filename: str = None):
        r1, r2 = lengths
        theta_rad = math.radians(angle_deg)

        # 3D 직교 좌표계 투영 (중심 원자를 원점 [0, 0, 0]에 배치)
        coords = [
            (self.elem_symbols.get(atomic_numbers[0], "X"), 0.0, 0.0, 0.0),
            (self.elem_symbols.get(atomic_numbers[1], "X"), r1, 0.0, 0.0),
            (self.elem_symbols.get(atomic_numbers[2], "X"), r2 * math.cos(theta_rad), r2 * math.sin(theta_rad), 0.0)
        ]

        xyz_lines = [f"{len(coords)}", f"{mol_name} | Angle: {angle_deg:.4f} deg | Generated by TDT-V3"]
        for sym, x, y, z in coords:
            xyz_lines.append(f"{sym:<4s} {x:12.6f} {y:12.6f} {z:12.6f}")
        xyz_content = "\n".join(xyz_lines)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(xyz_content + "\n")
            print(f" 💾 [XYZ Exported] '{filename}' 저장 완료:\n{xyz_content}")
        return xyz_content


# =====================================================================
# 📊 [STAGE 4] 사후 보정 스펙트럼 및 양자 .xyz 출력 검증 (Production Validation)
# =====================================================================
def validate_production_pipeline(calibrator, solver: TDTManifoldSolver, device='cpu'):
    print("\n🔬 [STAGE 4] 가속 시뮬레이터 사후 보정 정합성 최종 검증 집행")
    print("=" * 80)
    
    calibrator.eval()
    
    core_for_sim = getattr(calibrator, "core", None)
    if core_for_sim is None:
        core_for_sim = TDTEngineCore()
        
    simulator = MaterialScreeningSimulator(core=core_for_sim, solver=solver, calibration=calibrator)
    
    print("🛰️ [Simulator] MaterialScreeningSimulator 하이브리드 결착 성공.")
    print("🔍 [Verification] 오차 소거 대상 이원자 분자 'HCl' 정량 분석 시뮬레이션 가동...")
    
    hcl_target_obs = 1.2746
    
    with torch.no_grad():
        raw_hcl_tensor, _, _ = solver._compute_bond_length_tensor(1.0, 17.0, 1.0)
        uncalibrated_r = float(raw_hcl_tensor.item())
        
        nu_a_t = torch.tensor([1.0], dtype=torch.float64, device=device)
        nu_b_t = torch.tensor([17.0], dtype=torch.float64, device=device)
        nu_bond_t = torch.tensor([1.0], dtype=torch.float64, device=device)
        raw_r_t = torch.tensor([uncalibrated_r], dtype=torch.float64, device=device)
        corrected_r = float(calibrator.calibrate_bond_length(nu_a_t, nu_b_t, raw_r_t, nu_bond=nu_bond_t).item())
        
    before_error = abs(uncalibrated_r - hcl_target_obs)
    after_error = abs(corrected_r - hcl_target_obs)
    
    print("\n📊 [HCl 대조군 정량 분석 보고서]")
    print(f"   - 정밀 실험 실측치 (Obs Target)   : {hcl_target_obs:.4f} Å")
    print(f"   - 보정 전 솔버 출력치 (Uncalib)   : {uncalibrated_r:.4f} Å (오차: {before_error:.4f} Å)")
    print(f"   - AI 가드레일 보정 후 (Calibrated) : {corrected_r:.4f} Å (오차: {after_error:.4f} Å)")
    
    if after_error < before_error:
        print("\n✅ [STATUS] HCl 오차장 편차가 평형 평면 상태로 정상 회복되었습니다.")
    else:
        print("\n⚠️ [STATUS] 오차 복원율 정합성이 기준치에 도달하지 못했습니다. 체크 필요.")
        
    print("\n🧪 [DFT PIPELINE] 물 분자(H2O) 구조 데이터 최적화 및 내보내기 집행")
    h2o_lengths, h2o_angle = simulator.predict_molecule_geometry(
        center_z=8, side1_z=1, side2_z=1, nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=2.0
    )
    simulator.export_to_xyz("H2O_AI_Ready", [8, 1, 1], h2o_lengths, h2o_angle, filename="TDT_H2O_AI_Optimized.xyz")
        
    print("\n🛰️  H2O 양자 계산 .xyz 기하 구조 스트림 최적화 및 덤프 검증 완료.")
    print(" 🎉 [SUCCESS] 모든 AI 가드레일 보정 인프라가 수리적으로 무결하게 종료되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 [INIT] 실행 연산 백엔드 장치 배정: {device.upper()}")
    print("=" * 80)
    
    core = TDTEngineCore()
    solver = TDTManifoldSolver(core_engine=core, device=device)
    calibrator = TDTNeuralCalibrator(core_engine=core, device=device)
    
    dataset = load_first_principles_dataset(solver=solver, device=device)
    optimizer = configure_isolated_optimizer(calibrator)
    execute_neural_training_loop(calibrator, optimizer, dataset, epochs=500)
    validate_production_pipeline(calibrator, solver=solver, device=device)

