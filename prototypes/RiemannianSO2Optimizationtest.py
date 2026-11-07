import torch

class RiemannianSO2Optimization:
    """
    TDTUltimateEngineV23 전용 리만 다양체 SO(2) 공간 각도 수렴 커널
    인위적 클램핑 없이 구형 위상 표면(Unit Circle)의 지지선을 따라 최적점을 추적함.
    """
    def __init__(self, lr=0.01, max_iter=800):
        self.lr = lr
        self.max_iter = max_iter

    def optimize(self, target_angle, loss_fn_factory):
        # 🚀 90도 부근(중립 공간)의 다양체 좌표 벡터로 초기화 [cos(theta), sin(theta)]
        x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
        
        # 기하학적 다양체 경사하강 루프 전개
        for step in range(self.max_iter):
            # 복소 매니폴드 상의 실제 각도(theta) 복원 유도
            theta_current = torch.atan2(x[1], x[0])
            
            # 외부 수리 장 변수들과 결합된 커스텀 매듭 에너지 손실값 계산
            loss = loss_fn_factory(theta_current)
            
            # [수정 완완] 텐서 개별 미분 버퍼 제로화를 위해 zero_grad() 대신 zero_() 적용
            if x.grad is not None:
                x.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                grad_u = x.grad
                # 🧬 [리만 사영]: 유클리드 미분 계수를 구형 위상 접공간(Tangent Space)으로 투영
                proj_factor = torch.dot(x, grad_u)
                riemannian_grad = grad_u - proj_factor * x
                
                if torch.norm(riemannian_grad) < 1e-7:
                    break  # 기하학적 전하 바닥 평형 상태 도달 시 조기 수렴
                    
                # 🧬 [지지선 레트랙션]: 다양체 표면 곡률을 따라 벡터 회전 이동 업데이트
                step_size = self.lr
                direction = -riemannian_grad
                norm_d = torch.norm(direction)
                
                if norm_d > 0:
                    x.copy_(x * torch.cos(step_size * norm_d) + 
                            (direction / norm_d) * torch.sin(step_size * norm_d))
                    
        final_theta = torch.atan2(x[1], x[0])
        # 음수 각도 구역 존재 시 거울상 대칭 복원
        if final_theta < 0:
            final_theta = final_theta + 2.0 * torch.pi
            
        return torch.rad2deg(final_theta).item()

# =====================================================================
# 🧪 RiemannianSO2Optimization 단독 극한 스트레스 테스트 세션
# =====================================================================
if __name__ == "__main__":
    riemannian_solver = RiemannianSO2Optimization(lr=0.01, max_iter=1000)
    
    print("═" * 80)
    print(" [TEST] Riemannian SO(2) Manifold Solver Stress Test ")
    print("═" * 80)
    
    target_rad = torch.tensor(104.5 * torch.pi / 180.0, dtype=torch.float64)
    fake_well_rad = torch.tensor(128.0 * torch.pi / 180.0, dtype=torch.float64)
    
    def dummy_molecular_loss_factory(theta):
        restoring_term = (theta - target_rad) ** 2
        fake_local_minimum = 0.5 * torch.exp(-((theta - fake_well_rad) ** 2) / 0.01)
        pauli_geometrical_barrier = 0.01 / (torch.sin(theta) ** 2 + 1e-9)
        return restoring_term + fake_local_minimum + pauli_geometrical_barrier

    print("\n[단계 1] 중립 공간(90도)에서 출발 구형 위상 표면(Manifold) 탐색을 시작")
    try:
        calculated_angle = riemannian_solver.optimize(
            target_angle=target_rad, 
            loss_fn_factory=dummy_molecular_loss_factory
        )
        
        print("-" * 80)
        print(f" ▶ 리만 솔버 최종 수렴 각도 : {calculated_angle:.4f}°")
        print(f" ▶ 실제 가상 물리 목표값   : 104.5000°")
        print(f" ▶ 연산 오차 (Error)       : {calculated_angle - 104.5000:+.4f}°")
        print("-" * 80)
        
        if abs(calculated_angle - 104.5) < 0.5:
            print(" 🎉 [SUCCESS] 리만 다양체 솔버가 작동")
        else:
            print(" ⚠️ [STAGNATION] 특정 곡률 장벽에 갇혔습니다. 보폭(lr) 및 매니폴드 척도 조정이 필요")
            
    except Exception as e:
        print(f" ❌ [ERROR] 연산 그래프 전선 내에 런타임 오류 발생: {e}")
        
    print("═" * 80)
