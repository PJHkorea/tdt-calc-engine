`tdt_engine`은 제1원칙 수론적 물리 상수를 기반으로 분자 기하 구조를 고속 산출하고, 체계적 오차 장을 미분 가능한 순수 잔차 신경망 필드로 보정하는 **V3 하이브리드 물리-AI 가드레일 엔진**입니다.

---

## 📂 디렉토리 및 모듈 구조

최신 `v3` 파이프라인의 가독성과 유지보수성을 위해 메인 디렉토리에는 최신 코어만 배치하고, 구버전(`v1`, `v2`) 및 룩업 게이트 기반 모듈은 `legacy/`로 격리되어 있습니다.

* **메인(`v3`) 모듈:**
  * `core_v2.py`: 1~6주기(\(Z \in [1, 86]\)) 확장 및 단일 패스 폐형식(Closed-Form) 결합 길이 기저 엔진
  * `solvers_v3.py`: UFF-VSEPR 입체수(\(\mathcal{S}_{\text{eff}}\)) 및 `[B x 2]` 배치 리만 SO(2) 다양체 솔버
  * `calibrator_nn_v3.py`: 하드 게이트를 전면 제거한 물리 피처 기반 순수 잔차 + `Logit-Sigmoid` 보정기
  * `train_calibrator_v3.py`: 동적 오차장 추출, 라디안 통일 학습 및 `STAGE 1~5`(미학습 OOS 포함) 통합 파이프라인
  * `simulator_v2.py`: 의존성 주입(DI) 기반 최상위 오케스트레이터 및 양자 계산용 `.xyz` 좌표 출력기
* **`legacy/` 폴더:** 초기 결정론적 계산기(`calibrator.py`), `v1`/`v2` 솔버 및 보정기(`solvers.py`, `solvers_v2.py`, `calibrator_nn.py`, `calibrator_nn_v2.py`, `train_calibrator_v2.py`) , 구버전 `simulator.py`

---

## 🚀 두 가지 핵심 활용 모드 (Usage Modes)

### 1. V3 하이브리드 물리-AI 가드레일 파이프라인 모드 (권장)
다원자 배위수(`n_lig`, `sigma_bond`)와 결합차수(`nu_bond`)를 반영하여 분자 기하 구조를 고속 산출하고, 3D `.xyz` 좌표 스트림으로 내보냅니다.

* **실행 방법:**
  ```bash
  python -m tdt_engine.core_v2                  # 제1원칙 폐형식 코어 단독 검증
  python -m tdt_engine.solvers_v3               # UFF-VSEPR 및 [B=10] 배치 리만 솔버 검증
  python -m tdt_engine.calibrator_nn_v3         # 순수 잔차 + Logit-Sigmoid 역전파 검증
  python -m tdt_engine.train_calibrator_v3      # STAGE 1~5 학습 및 미학습(OOS) 일반화 검증
  python -m tdt_engine.simulator_v2                # 22종 분자 벤치마크 및 .xyz 스트림 출력
  ```
* **파이썬 코드 연동 예시:**
  ```python
  from tdt_engine.core_v2 import TDTEngineCore
  from tdt_engine.solvers_v3 import TDTManifoldSolver
  from tdt_engine.calibrator_nn_v3 import TDTNeuralCalibrator
  from tdt_engine.simulator_v2 import MaterialScreeningSimulator

  core = TDTEngineCore()
  solver = TDTManifoldSolver(core_engine=core)
  calibrator = TDTNeuralCalibrator(core_engine=core)
  simulator = MaterialScreeningSimulator(core=core, solver=solver, calibration=calibrator)

  # 물 분자(H2O) 기하 구조 산출 및 .xyz 파일 저장
  lengths, angle = simulator.predict_molecule_geometry(
      center_z=8, side1_z=1, side2_z=1,
      nu_b1=1.0, nu_b2=1.0, n_lig=2.0, sigma_bond=2.0
  )
  simulator.export_to_xyz("H2O_AI_Ready", [8, 1, 1], lengths, angle, filename="TDT_H2O_Optimized.xyz")
  ```

### 2. 초경량 순수 NumPy 폐형식 계산기 모드 (Zero-PyTorch Standalone)
PyTorch나 GPU 없이 순수 NumPy만으로 1~6주기 분자의 결합 길이를 단일 패스 폐형식 수식으로 즉시 계산합니다.

```python
from tdt_engine.core_v2 import TDTEngineCore

core = TDTEngineCore()
calc_R, a_A_mod, a_B_mod = core.compute_bond_length(nu_A=1.0, nu_B=53.0, nu_bond=1.0)
print(f"HI Closed-Form Bond Length: {calc_R:.4f} Å")  # 1.6629 Å (Obs: 1.6080 Å)
```

---

## 🛡️ 개발 및 AI 에이전트 수칙 (`AGENT_PROMPT_v3.md` 준수)

* **기저 코어 및 상수 테이블 동결 (`core_v2.py`):** 제1원칙 불변량($a_0, \omega_1, \alpha$)과 1~86번 표준 상수 버퍼($\mathbf{T}_{\text{cov}}, \mathbf{T}_{\chi}$)는 읽기 전용으로 보호하며 옵티마이저가 수정하지 않습니다.
* **Branch-Free 연속 미분 연산 유지:** 물리 및 보정 루프 내 `if-else` 분기문을 전면 배제하고 `torch.tanh`, `F.softplus`, `torch.acos`, `torch.logit`만 사용합니다.
* **Zero Lookup-Gate & 라디안 통일 손실:** 특정 분자 정답지 게이트 없이 5D/8D 물리 피처 기반 순수 잔차 학습을 수행하며, 길이($\text{\AA}^2$)와 각도($\text{rad}^2$) 손실을 자연 단위로 균형 학습합니다.

