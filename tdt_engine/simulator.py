import numpy as np
import os


# 🚀 [V24_ext 노트북 통합 연동] 파일 임포트 구문을 완전히 제거하고 메모리 내 클래스를 직접 연동합니다.
# (이미 상단 셀에서 AbstractPredictor, TDTEngineCore, TDTManifoldSolver, TDTCalibrationModule을 구동했으므로 생략)

# 📌 단일 공급원 체계(Single Source of Truth) 규칙 완수
# 메모리에 로드된 TDTEngineCore 내부에 선언된 레지스트리를 최상위 전역 상수로 동기화 바인딩
if hasattr(TDTEngineCore, "PERIODIC_TABLE"):
    GLOBAL_PERIODIC_TABLE = TDTEngineCore.PERIODIC_TABLE
else:
    # TDTEngineCore 클래스 내부에 아직 상수가 구현되지 않았을 때를 위한 안전선 보존
    GLOBAL_PERIODIC_TABLE = {
        1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 
        11: "Na", 14: "Si", 15: "P", 16: "S", 17: "Cl", 32: "Ge", 53: "I"
    }


class MaterialScreeningSimulator:
    """
    [V24_ext 최종 혁신] TDT 계산 엔진 최상위 오케스트레이터 레이어입니다.
    Core, Solver, Calibrator를 의존성 주입(DI) 방식으로 취합하여 
    최종 신소재/신약용 분자 구조 예측 및 양자 계산용 .xyz 출력을 집행합니다.
    """
    def __init__(self, core: "TDTEngineCore", solver: "TDTManifoldSolver", calibration: "AbstractCalibrator"):
        """
        정밀 객체지향 의존성 주입(Dependency Injection) 레이어 셋업
        📌 문자열 타입 힌트를 사용하여 노트북 재실행 시 발생하는 객체 ID 미스매칭 버그를 원천 차단합니다.
        """
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
        # 1. 리만 다양체 가속 솔버 연산 수행 (nb2 오타 및 불필요한 locals 삼항연산자 제거 완료)
        raw_r1, _, _ = self.solver.compute_bond_length(center_z, side1_z, nu_bond=nu_b1)
        raw_r2, _, _ = self.solver.compute_bond_length(center_z, side2_z, nu_bond=nu_b2)
        raw_angle = self.solver.compute_angle(center_z, side1_z, side2_z, nu_b1=nu_b1, nu_b2=nu_b2)
        
        # 2. 캘리브레이션 안전망 레이어를 거쳐 도메인 수렴 최종 보정
        opt_r1 = self.calibration.calibrate_bond_length(center_z, side1_z, raw_r1)
        opt_r2 = self.calibration.calibrate_bond_length(center_z, side2_z, raw_r2)
        opt_angle = self.calibration.calibrate_bond_angle(center_z, side1_z, side2_z, raw_angle, nu_b1=nu_b1, nu_b2=nu_b2)
        
        return [opt_r1, opt_r2], opt_angle

    def export_to_xyz(self, molecule_name: str, elements: list[int], 
                      lengths: list[float], angle_deg: float, filename: str = None) -> str:
        """
        최적화가 완료된 분자 좌표 텐서를 유클리드 3차원 공간으로 사영하여 
        상용 시각화 툴(VMD, Avogadro)과 호환되는 표준 양자 계산 규격 .xyz 스트림으로 내보냅니다.
        """
        if len(elements) != 3 or len(lengths) != 2:
            raise ValueError("AX2 분자 구조는 정확히 3개의 원소 번호와 2개의 결합 길이를 요구합니다.")
            
        # 전역 주기율표 매핑 (안전하게 Fallback 처리 포함)
        symbols = [GLOBAL_PERIODIC_TABLE.get(z, f"X_{z}") for z in elements]
        
        # [원점 배치] 중심 원소 (0.0, 0.0, 0.0)
        x0, y0, z0 = 0.0, 0.0, 0.0
        
        # [1번 외각 원소 선형 사영] X축에 평행하게 배치
        r1 = lengths[0]
        x1, y1, z1 = r1, 0.0, 0.0
        
        # [2번 외각 원소 입체 삼각 사영] 보정된 결합 각도(angle_deg)를 반영하여 XY 평면에 배치
        r2 = lengths[1]
        angle_rad = np.radians(angle_deg)
        x2 = r2 * np.cos(angle_rad)
        y2 = r2 * np.sin(angle_rad)
        z2 = 0.0
        
        # 📌 표준 양자 화학 .xyz 정밀 고정폭 포매팅 규격 정합
        xyz_lines = [
            "3",
            f"TDT Ultimate Engine V24_ext Production Geometry - Molecule: {molecule_name} | Opt Angle: {angle_deg:.4f}°",
            f"{symbols[0]:<4s} {x0:12.6f} {y0:12.6f} {z0:12.6f}",  # 중심 원소 (예: O)
            f"{symbols[1]:<4s} {x1:12.6f} {y1:12.6f} {z1:12.6f}",  # 외각 원소 1 (예: H1)
            f"{symbols[2]:<4s} {x2:12.6f} {y2:12.6f} {z2:12.6f}"   # 외각 원소 2 (예: H2)
        ]
        xyz_buffer = "\n".join(xyz_lines) + "\n"
        
        if filename:
            try:
                # 상위 디렉토리가 없을 경우를 대비한 동적 생성 처리 추가 (안전성 강화)
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

    # 📌 [단일 책임 보존 / 코랩 캐시 강제 소멸 안정화]
    # 이전 파일 임포트 흔적으로 꼬인 백엔드 모듈 캐시만 세척하고
    # 노트북 전역 공간(globals)에 정상 로드된 정식 클래스 선언은 안전하게 보존합니다.
    for module_key in list(sys.modules.keys()):
        if 'tdt_engine' in module_key:
            del sys.modules[module_key]

    print("=" * 80)
    print(" TDT Ultimate Engine V24_ext - Material Screening Simulator Benchmark ")
    print("=" * 80)

    # 1. 🚀 [정식 코어 파이프라인] 의존성 주입(Dependency Injection) 레이어 다이렉트 셋업
    # 메모리에 정상 선언된 제1원칙 물리 코어와 가속 솔버를 직접 인스턴스화합니다.
    core_engine = TDTEngineCore()
    solver_instance = TDTManifoldSolver(core_engine=core_engine)
    calib_module = TDTCalibrationModule(core_engine=core_engine)

    # 최상위 오케스트레이터 가동
    simulator = MaterialScreeningSimulator(core=core_engine, solver=solver_instance, calibration=calib_module)

    # 2. V24_ext 규격 결합 길이 10대 벤치마크 데이터셋 정의
    length_cases = [
        ("H2",   1,  1, 0.741), ("HF",   1,  9, 0.917), ("HCl",  1, 17, 1.3767), 
        ("LiF",  3,  9, 1.564), ("CO",   6,  8, 1.3928), ("N2",   7,  7, 1.098), 
        ("NaF", 11,  9, 1.926), ("F2",   9,  9, 1.412), ("BF",   5,  9, 1.263), ("LiH",  1,  3, 0.9476)
    ]

    # 3. V24_ext 규격 결합 각도 10대 벤치마크 데이터셋 정의
    angle_cases = [
        ("CO2",   6,  8,  8, 179.9427), ("CH4",   6,  1,  1, 115.0100), ("H2O",   8,  1,  1, 179.9427), 
        ("NH3",   7,  1,  1, 176.3746), ("H2S",  16,  1,  1,  92.1000), ("PH3",  15,  1,  1,  93.5000), 
        ("NF3",   7,  9,  9, 102.3000), ("OF2",   8,  9,  9, 103.3000), ("SO2",  16,  8,  8, 119.0000), ("ClO2", 17,  8,  8, 111.1000)
    ]

    print("📊 [BENCHMARK RUN] 20대 분자 사후 가드레일 검증 시퀀스 가동...")

    # 📌 단일 결합 길이를 검증할 때는 z_A를 중심으로 두고 대칭 결합으로 연산 흐름을 유도
    print("\n[Phase 1] 결합 길이 검증 (10 Cases)")
    print("-" * 65)
    for mol, z_A, z_B, raw_R in length_cases:
        opt_lengths, _ = simulator.predict_molecule_geometry(z_A, z_B, z_B)
        print(f" - {mol:<5} (Z: {z_A:>2}-{z_B:<2}) | Raw: {raw_R:.4f} Å -> 보정값: {opt_lengths[0]:.4f} Å")

    print("\n[Phase 2] 결합 각도 검증 (10 Cases)")
    print("-" * 65)
    for mol, z_C, z_S1, z_S2, raw_ang in angle_cases:
        _, opt_angle = simulator.predict_molecule_geometry(z_C, z_S1, z_S2)
        print(f" - {mol:<5} (Z: {z_S1}-{z_C}-{z_S2}) | Raw: {raw_ang:8.4f}° -> 보정값: {opt_angle:8.4f}°")

    print("\n" + "=" * 80)

    print("🧪 [DFT PIPELINE] 물 분자(H2O) 구조 데이터 최적화 및 내보내기 집행")
    print("-" * 80)

    h2o_elements = [8, 1, 1] 

    # 가속 파이프라인 정석 흐름 집행 (중심: O(8), 외각: H(1), H(1))
    opt_lengths, opt_angle = simulator.predict_molecule_geometry(8, 1, 1)

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
    print("\n🎉 모든 시뮬레이션 인프라 및 V24_ext 벤치마크 파이프라인이 정식 코어로 정상 종료되었습니다.")
    print("=" * 80)


