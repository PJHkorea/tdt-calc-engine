import numpy as np
import os

# [V3 통합 연동] 코랩/노트북 및 패키지 환경 모두 지원
try:
    from tdt_engine.core_v2 import TDTEngineCore
except ImportError:
    pass

# 단일 공급원 체계(Single Source of Truth) 및 1~6주기 주요 원소 심볼 테이블
if "TDTEngineCore" in globals() and hasattr(TDTEngineCore, "PERIODIC_TABLE"):
    GLOBAL_PERIODIC_TABLE = TDTEngineCore.PERIODIC_TABLE
else:
    GLOBAL_PERIODIC_TABLE = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
        19: "K", 20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn", 32: "Ge", 35: "Br", 36: "Kr",
        47: "Ag", 50: "Sn", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 79: "Au", 82: "Pb"
    }

class MaterialScreeningSimulator:
    """
    [V3 Hybrid] TDT 계산 엔진 최상위 오케스트레이터 레이어입니다.
    Core(v2), Solver(v3), Calibrator(v3)를 의존성 주입(DI) 방식으로 결합하여
    분자 기하 구조 예측 및 양자 화학 계산용 .xyz 스트림을 출력합니다.
    """
    def __init__(self, core: "TDTEngineCore", solver: "TDTManifoldSolver", calibration: "AbstractCalibrator"):
        self.core = core
        self.solver = solver
        self.calibration = calibration
        print("🚀 [MaterialScreeningSimulator] V3 하이브리드 가속 시뮬레이터 가동 준비 완료.\n")

    def predict_molecule_geometry(self, center_z: int, side1_z: int, side2_z: int,
                                  nu_b1: float = 1.0, nu_b2: float = 1.0,
                                  n_lig: float = 2.0, sigma_bond: float = None):
        """
        [핵심 비즈니스 로직 - V3 하이브리드 가속 인터페이스]
        배치 리만 SO(2) 솔버(solvers_v3)와 순수 잔차 신경망(calibrator_nn_v3)을 연동하여
        결합 길이(Å)와 각도(°)를 정밀 보정합니다.
        """
        if sigma_bond is None:
            sigma_bond = nu_b1 + nu_b2

        # 1. 리만 다양체 V3 솔버 연산 수행
        raw_r1, _, _ = self.solver.compute_bond_length(center_z, side1_z, nu_bond=nu_b1)
        raw_r2, _, _ = self.solver.compute_bond_length(center_z, side2_z, nu_bond=nu_b2)
        raw_angle = self.solver.compute_angle(
            center_z, side1_z, side2_z,
            nu_b1=nu_b1, nu_b2=nu_b2, n_lig=n_lig, sigma_bond=sigma_bond
        )

        # 2. PyTorch V3 순수 잔차 보정기 연동
        if hasattr(self.calibration, "length_calibrator") and hasattr(self.calibration, "angle_calibrator"):
            import torch
            device = getattr(self.calibration, "device", "cpu")

            with torch.no_grad():
                t_center = torch.tensor([float(center_z)], dtype=torch.float64, device=device)
                t_side1  = torch.tensor([float(side1_z)], dtype=torch.float64, device=device)
                t_side2  = torch.tensor([float(side2_z)], dtype=torch.float64, device=device)
                t_b1     = torch.tensor([float(nu_b1)], dtype=torch.float64, device=device)
                t_b2     = torch.tensor([float(nu_b2)], dtype=torch.float64, device=device)
                t_sig    = torch.tensor([float(sigma_bond)], dtype=torch.float64, device=device)
                t_r1     = torch.tensor([float(raw_r1)], dtype=torch.float64, device=device)
                t_r2     = torch.tensor([float(raw_r2)], dtype=torch.float64, device=device)
                t_angle  = torch.tensor([float(raw_angle)], dtype=torch.float64, device=device)

                opt_r1 = float(self.calibration.calibrate_bond_length(t_center, t_side1, t_r1, nu_bond=t_b1).item())
                opt_r2 = float(self.calibration.calibrate_bond_length(t_center, t_side2, t_r2, nu_bond=t_b2).item())
                opt_angle = float(self.calibration.calibrate_bond_angle(
                    t_center, t_side1, t_side2, t_angle,
                    nu_b1=t_b1, nu_b2=t_b2, sigma_bond=t_sig, return_radians=False
                ).item())
        else:
            opt_r1 = self.calibration.calibrate_bond_length(center_z, side1_z, raw_r1)
            opt_r2 = self.calibration.calibrate_bond_length(center_z, side2_z, raw_r2)
            opt_angle = self.calibration.calibrate_bond_angle(
                center_z, side1_z, side2_z, raw_angle, nu_b1=nu_b1, nu_b2=nu_b2
            )

        return [opt_r1, opt_r2], opt_angle


    def export_to_xyz(self, molecule_name: str, elements: list[int], 
                      lengths: list[float], angle_deg: float, filename: str = None) -> str:
        """
        최적화가 완료된 분자 좌표 텐서를 유클리드 3차원 공간으로 사영하여 
        상용 시각화 툴(VMD, Avogadro)과 호환되는 표준 양자 계산 규격 .xyz 스트림으로 내보냅니다.
        """
        if len(elements) != 3 or len(lengths) != 2:
            raise ValueError("AX2 분자 구조는 정확히 3개의 원소 번호와 2개의 결합 길이를 요구합니다.")
            
        symbols = [GLOBAL_PERIODIC_TABLE.get(z, f"X_{z}") for z in elements]
        
        x0, y0, z0 = 0.0, 0.0, 0.0
        
        r1 = lengths[0]
        x1, y1, z1 = r1, 0.0, 0.0
        
        r2 = lengths[1]
        angle_rad = np.radians(angle_deg)
        x2 = r2 * np.cos(angle_rad)
        y2 = r2 * np.sin(angle_rad)
        z2 = 0.0
        
        xyz_lines = [
            "3",
            f"TDT Ultimate Engine V3 Production Geometry - Molecule: {molecule_name} | Opt Angle: {angle_deg:.4f}°",
            f"{symbols[0]:<4s} {x0:12.6f} {y0:12.6f} {z0:12.6f}",
            f"{symbols[1]:<4s} {x1:12.6f} {y1:12.6f} {z1:12.6f}",
            f"{symbols[2]:<4s} {x2:12.6f} {y2:12.6f} {z2:12.6f}"
        ]
        xyz_buffer = "\n".join(xyz_lines) + "\n"
        
        if filename:
            try:
                dir_name = os.path.dirname(filename)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                    
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(xyz_buffer)
                print(f" 📂 [.xyz File Export] 최적화 완료 스트림 디스크 저장 -> {filename}")
            except (IOError, OSError) as e:
                print(f" ❌ [.xyz File Export] 디스크 출력 스트림 가동 실패: {e}")
                
        return xyz_buffer




if __name__ == "__main__":
    import sys

    for module_key in list(sys.modules.keys()):
        if 'tdt_engine' in module_key:
            del sys.modules[module_key]

    try:
        from tdt_engine.core_v2 import TDTEngineCore
        from tdt_engine.solvers_v3 import TDTManifoldSolver
        from tdt_engine.calibrator_nn_v3 import TDTNeuralCalibrator
    except ImportError:
        pass

    print("=" * 80)
    print(" TDT Ultimate Engine V3 - Material Screening Simulator Benchmark ")
    print("=" * 80)

    core_engine = TDTEngineCore()
    solver_instance = TDTManifoldSolver(core_engine=core_engine)
    calib_module = TDTNeuralCalibrator(core_engine=core_engine)

    simulator = MaterialScreeningSimulator(core=core_engine, solver=solver_instance, calibration=calib_module)


    length_cases = [
        ("H2",   1,  1, 1.0, 0.7414), ("HF",   1,  9, 1.0, 0.9168),
        ("HCl",  1, 17, 1.0, 1.2746), ("HBr",  1, 35, 1.0, 1.4140),
        ("HI",   1, 53, 1.0, 1.6080), ("LiF",  3,  9, 1.0, 1.5639),
        ("CO",   6,  8, 3.0, 1.1283), ("N2",   7,  7, 3.0, 1.0977),
        ("NaF", 11,  9, 1.0, 1.9260), ("F2",   9,  9, 1.0, 1.4119),
        ("BF",   5,  9, 1.3, 1.2626), ("LiH",  3,  1, 1.0, 1.5959)
    ]

    angle_cases = [
        ("CO2",   6,  8,  8, 2.0, 2.0, 2.0, 4.0, 180.00),
        ("CH4",   6,  1,  1, 1.0, 1.0, 4.0, 4.0, 109.47),
        ("H2O",   8,  1,  1, 1.0, 1.0, 2.0, 2.0, 104.50),
        ("NH3",   7,  1,  1, 1.0, 1.0, 3.0, 3.0, 106.70),
        ("H2S",  16,  1,  1, 1.0, 1.0, 2.0, 2.0,  92.11),
        ("PH3",  15,  1,  1, 1.0, 1.0, 3.0, 3.0,  93.50),
        ("NF3",   7,  9,  9, 1.0, 1.0, 3.0, 3.0, 102.30),
        ("OF2",   8,  9,  9, 1.0, 1.0, 2.0, 2.0, 103.30),
        ("SO2",  16,  8,  8, 1.5, 1.5, 2.0, 3.0, 119.30),
        ("ClO2", 17,  8,  8, 1.5, 1.5, 2.0, 3.0, 111.00)
    ]

    print("📊 [BENCHMARK RUN] V3 22종 분자 파이프라인 검증 시퀀스 가동...")


    print("\n[Phase 1] 결합 길이 검증 (12 Cases)")
    print("-" * 65)
    for mol, z_A, z_B, nu_b, obs_R in length_cases:
        opt_lengths, _ = simulator.predict_molecule_geometry(z_A, z_B, z_B, nu_b1=nu_b, nu_b2=nu_b)
        print(f" - {mol:<5} (Z: {z_A:>2}-{z_B:<2}, ν={nu_b:.1f}) | 보정값: {opt_lengths[0]:.4f} Å (Obs: {obs_R:.4f} Å)")

    print("\n[Phase 2] 결합 각도 검증 (10 Cases)")
    print("-" * 65)
    for mol, z_C, z_S1, z_S2, nb1, nb2, n_lig, sig_b, obs_ang in angle_cases:
        _, opt_angle = simulator.predict_molecule_geometry(
            z_C, z_S1, z_S2, nu_b1=nb1, nu_b2=nb2, n_lig=n_lig, sigma_bond=sig_b
        )
        print(f" - {mol:<5} (Z: {z_S1}-{z_C}-{z_S2}) | 보정값: {opt_angle:8.4f}° (Obs: {obs_ang:8.4f}°)")

    print("\n" + "=" * 80)

    print("🧪 [DFT PIPELINE] 물 분자(H2O) 구조 데이터 최적화 및 내보내기 집행")
    print("-" * 80)

    h2o_elements = [8, 1, 1]
    opt_lengths, opt_angle = simulator.predict_molecule_geometry(
        8, 1, 1, nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=2.0
    )

    output_filename = "TDT_H2O_Optimized.xyz"
    xyz_result = simulator.export_to_xyz(
        molecule_name="H2O_DFT_Ready",
        elements=h2o_elements,
        lengths=opt_lengths,
        angle_deg=opt_angle,
        filename=output_filename
    )

    print("\n[생성된 XYZ 데이터 스트림 미리보기]")
    print("." * 50)
    print(xyz_result.strip())
    print("." * 50)
    print("\n🎉 모든 시뮬레이션 인프라 및 V3 벤치마크 파이프라인이 정식 코어로 정상 종료되었습니다.")
    print("=" * 80)
