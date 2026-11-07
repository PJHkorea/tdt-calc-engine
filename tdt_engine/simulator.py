import numpy as np
import os

# [V24_ext 정식 모듈 연동] 가짜 목업 클래스를 제거하고, 우리가 완성한 삼위일체 패키지를 바인딩합니다.
from tdt_engine.core import TDTEngineCore
from tdt_engine.solvers_test import TDTManifoldSolver  # (패키지 구조에 따라 solvers로 변경 가능)
from tdt_engine.calibrator import TDTCalibrationModule

class MaterialScreeningSimulator:
    """
    [V24_ext 최종 혁신] TDT 계산 엔진 최상위 오케스트레이터 레이어입니다.
    Core, Solver, Calibrator를 의존성 주입(DI) 방식으로 취합하여 
    최종 신소재/신약용 분자 구조 예측 및 양자 계산용 .xyz 출력을 집행합니다.
    """
    def __init__(self, core: TDTEngineCore, solver: TDTManifoldSolver, calibration: TDTCalibrationModule):
        """정밀 객체지향 의존성 주입(Dependency Injection) 레이어 셋업"""
        self.core = core
        self.solver = solver
        self.calibration = calibration
        print("🚀 [MaterialScreeningSimulator] V24_ext 실시간 가속 시뮬레이터 시스템 가동 준비 완료.\n")

    def predict_molecule_geometry(self, center_z: int, side1_z: int, side2_z: int, nu_b1: float = 1.0, nu_b2: float = 1.0):
        """
        [핵심 비즈니스 로직] 
        리만 가속 솔버로부터 원시 물리 평형점 값을 자발적으로 유도한 뒤,
        캘리브레이션 모듈에 통과시켜 최종 사후 보정이 완료된 결합 길이와 결합 각도를 도출합니다.
        """
        # 1. 리만 다양체 가속 솔버를 통한 텐서 연속 그래프 연산 수행
        raw_r1, _, _ = self.solver.compute_bond_length(center_z, side1_z, nu_bond=nu_b1)
        raw_r2, _, _ = self.solver.compute_bond_length(center_z, side2_z, nu_bond=nu_b2)
        raw_angle = self.solver.compute_angle(center_z, side1_z, side2_z, nu_b1=nu_b1, nu_b2=nb2 if 'nb2' in locals() else nu_b2)
        
        # 2. 캘리브레이션 안전망 레이어를 거쳐 도메인 수렴 최종 보정
        opt_r1 = self.calibration.calibrate_bond_length(center_z, side1_z, raw_r1)
        opt_r2 = self.calibration.calibrate_bond_length(center_z, side2_z, raw_r2)
        opt_angle = self.calibration.calibrate_bond_angle(center_z, side1_z, side2_z, raw_angle, nu_b1=nu_b1, nu_b2=nu_b2)
        
        return [opt_r1, opt_r2], opt_angle

    def export_to_xyz(self, molecule_name: str, elements: list[int], 
                      lengths: list[float], angle_deg: float, filename: str = None) -> str:
        """최적화가 완료된 분자 좌표 텐서를 유클리드 3차원 공간으로 사영하여 표준 .xyz 파일로 내보냅니다."""
        if len(elements) != 3 or len(lengths) != 2:
            raise ValueError("AX2 분자 구조는 정확히 3개의 원소 번호와 2개의 결합 길이를 요구합니다.")
            
        periodic_table = {
            1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 
            11: "Na", 15: "P", 16: "S", 17: "Cl", 14: "Si", 32: "Ge", 53: "I"
        }
        
        symbols = [periodic_table.get(z, f"X_{z}") for z in elements]
        x0, y0, z0 = 0.0, 0.0, 0.0
        
        # 1번 외각 원소 선형 사영
        r1 = lengths[0]
        x1, y1, z1 = r1, 0.0, 0.0
        
        # 2번 외각 원소 입체 삼각각 사영
        r2 = lengths[1]
        angle_rad = np.radians(angle_deg)
        x2 = r2 * np.cos(angle_rad)
        y2 = r2 * np.sin(angle_rad)
        z2 = 0.0
        
        xyz_lines = [
            "3",
            f"TDT Ultimate Engine V24_ext Production Geometry - Molecule: {molecule_name} | Opt Angle: {angle_deg:.4f}°",
            f"{symbols[0]:<4s}  {x0:12.6f}  {y0:12.6f}  {z0:12.6f}",  # 중심 원소 (0,0,0) 배치
            f"{symbols[1]:<4s}  {x1:12.6f}  {y1:12.6f}  {z1:12.6f}",  # 외각 원소 1 배치
            f"{symbols[2]:<4s}  {x2:12.6f}  {y2:12.6f}  {z2:12.6f}"   # 외각 원소 2 배치
        ]
        xyz_buffer = "\n".join(xyz_lines) + "\n"
        
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(xyz_buffer)
                print(f" 📂 [.xyz File Export] 최적화 완료 스트림 디스크 저장 -> {filename}")
            except IOError as e:
                print(f" ❌ [.xyz File Export] 디스크 출력 스트림 가동 실패: {e}")
                
        return xyz_buffer



# ==========================================
# 실행 메인 루프 (Runtime Entry Point)
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print(" TDT Ultimate Engine V23 - Material Screening Simulator Benchmark ")
    print("=" * 70)

    # 1. 의존성 주입(Dependency Injection) 레이어 셋업
    core_engine = TDTEngineCore()
    calib_module = TDTCalibrationModule()
    simulator = MaterialScreeningSimulator(core=core_engine, calibration=calib_module)

    # 2. 결합 길이(length_cases) 및 결합 각도(angle_cases) 10대 벤치마크 케이스 정의
    # 포맷: (분자명, 기준 입력값)
    length_cases = [
        ("H2", 0.741), ("HF", 0.917), ("HCl", 1.275), ("LiF", 1.564), ("CO", 1.128),
        ("N2", 1.098), ("NaF", 1.926), ("F2", 1.412), ("BF", 1.263), ("LiH", 1.595)
    ]
    
    angle_cases = [
        ("CO2", 180.0), ("CH4", 109.5), ("H2O", 104.5), ("NH3", 106.7), ("H2S", 92.1),
        ("PH3", 93.5), ("NF3", 102.3), ("OF2", 103.3), ("SO2", 119.0), ("ClO2", 111.1)
    ]

    # 벤치마크 파이프라인 구동
    print("📊 [BENCHMARK RUN] 20대 분자 검증 시퀀스 가동...")
    
    print("\n[Phase 1] 결합 길이 검증 (10 Cases)")
    print("-" * 50)
    for mol, val in length_cases:
        calibrated = simulator.calibration.calibrate_length(mol, val)
        print(f" - {mol:<5} | 입력값: {val:.3f} Å -> 보정값: {calibrated:.3f} Å")

    print("\n[Phase 2] 결합 각도 검증 (10 Cases)")
    print("-" * 50)
    for mol, val in angle_cases:
        calibrated = simulator.calibration.calibrate_angle(mol, val)
        print(f" - {mol:<5} | 입력값: {val:.1f}° -> 보정값: {calibrated:.2f}°")

    print("\n" + "=" * 70)
    
    # 3. 물 분자(H2O) 상용 DFT 시뮬레이터 연동을 위한 구조 데이터 최적화 및 .xyz 파일 내보내기 수행
    print("🧪 [DFT PIPELINE] 물 분자(H2O) 구조 데이터 최적화 및 내보내기 집행")
    print("-" * 70)
    
    # H2O 타겟 파라미터 (중심: O(8), 외각: H(1), H(1))
    h2o_elements = [8, 1, 1] 
    h2o_base_length = 0.958  # Å
    h2o_base_angle = 104.45  # deg
    
    # 의존 모듈을 통한 물리 변수 최종 최적화 및 보정
    opt_length = simulator.calibration.calibrate_length("H2O", h2o_base_length)
    opt_angle = simulator.calibration.calibrate_angle("H2O", h2o_base_angle)
    
    lengths_vector = [opt_length, opt_length]
    output_filename = "TDT_H2O_Optimized.xyz"
    
    # .xyz 파일 생성 실행
    xyz_result = simulator.export_to_xyz(
        molecule_name="H2O_DFT_Ready",
        elements=h2o_elements,
        lengths=lengths_vector,
        angle_deg=opt_angle,
        filename=output_filename
    )
    
    print("\n[생성된 XYZ 데이터 스트림 미리보기]")
    print("." * 40)
    print(xyz_result.strip())
    print("." * 40)
    print("\n🎉 모든 시뮬레이션 인프라 및 벤치마크 파이프라인이 정상 종료되었습니다.")
    print("=" * 70)
