`tdt_engine`은 제1원칙 수론적 물리 상수를 기반으로 분자 기하 구조를 고속 산출하고, 체계적 오차 장을 미분 가능한 신경망 잔차 필드로 보정하는 하이브리드 물리-AI 가드레일 형태입니다.

---

## 📂 디렉토리 및 모듈 구조

최신 `v2` 파이프라인의 가독성과 유지보수성을 위해 메인 디렉토리에는 최신 코어만 배치하고, 초기 `v1` 및 정적 딕셔너리 계산기 모듈은 `legacy/`로 격리되어 있습니다.

* **메인(`v2`) 모듈:** `core.py`(제1원칙 기저 물리 상수 및 장 엔진), `solvers_v2.py`(최신 솔버), `calibrator_nn_v2.py`(미분 가능 AI 보정기), `train_calibrator_v2.py`(학습 스크립트), `simulator.py`(최상위 오케스트레이터 및 `.xyz` 좌표 출력기)
* **`legacy/` 폴더:** PyTorch 없이 동작하는 정적 딕셔너리 계산기(`calibrator.py`), v1 신경망 보정기(`calibrator_nn.py`), v1 리만 다양체 솔버(`solvers.py`), v1 학습 스크립트(`train_calibrator.py`)

---

## 🚀 두 가지 핵심 활용 모드 (Usage Modes)

### 1. AI 가드레일 & 미분 가능 물리 파이프라인 모드 (`v2` 권장)
생성형 AI의 추론 루프나 역전파(`Autograd`) 손실 함수에 결합하여 비물리적 구조를 실시간으로 필터링합니다.

* **실행 방법:**
  ```bash
  python -m tdt_engine.solvers_v2               # 솔버 단독 물리 평형 검증
  python -m tdt_engine.calibrator_nn_v2         # AI 가드레일 연속 미분 검증
  python -m tdt_engine.train_calibrator_v2      # 동적 오차장 학습 및 프로덕션 실행
  ```
* **파이썬 코드 연동 예시:**
  ```python
  from tdt_engine.core import TDTEngineCore
  from tdt_engine.solvers_v2 import TDTManifoldSolver
  from tdt_engine.calibrator_nn_v2 import TDTNeuralCalibrator
  from tdt_engine.simulator import MaterialScreeningSimulator

  core = TDTEngineCore()
  solver = TDTManifoldSolver(core_engine=core)
  calibrator = TDTNeuralCalibrator(core_engine=core)
  simulator = MaterialScreeningSimulator(core=core, solver=solver, calibration=calibrator)

  lengths, angle = simulator.predict_molecule_geometry(center_z=8, side1_z=1, side2_z=1)
  simulator.export_to_xyz("H2O_AI_Ready", [8, 1, 1], lengths, angle, filename="TDT_H2O_Optimized.xyz")
  ```

### 2. 초경량 결정론적 계산기 모드 (`Legacy Standalone`)
PyTorch나 GPU 없이 순수 NumPy와 결정론적 룰셋만으로 분자의 결합 길이와 각도를 즉시 계산합니다.

```python
from tdt_engine.core import TDTEngineCore
from tdt_engine.legacy.calibrator import TDTCalibrationModule

core = TDTEngineCore()
calc_mod = TDTCalibrationModule(core_engine=core)
raw_R, _, _ = core.compute_bond_length(1, 17, nu_bond=1.0)
calibrated_R = calc_mod.calibrate_bond_length(1, 17, raw_R)
print(f"HCl Calibrated Bond Length: {calibrated_R:.4f} Å")
```

---

## 🛡️ 개발 및 AI 에이전트 수칙 (`AGENT_PROMPT_v2.md` 준수)

* **기저 코어 동결 (`core.py`):** 제1원칙 수식($a_0, \omega_1, \alpha$ 등)은 어떠한 경우에도 수정하지 않습니다.
* **Branchless 연산 유지:** 하드코딩된 `if-else` 분기문 대신 `torch.tanh` 및 `torch.where`를 사용합니다.
* **데이터 누수 차단:** `train_calibrator_v2.py`는 `solver`가 실시간으로 계산한 원시 출력값(`Raw`)을 신경망 입력으로 사용합니다.
