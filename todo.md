1. cartpole - 진자 스윙업 제어
  - 시스템: 카트 위 진자 (Cart-Pole)
  - 상태: [x, theta, dx, dtheta] (카트 위치, 진자 각도, 속도들)
  - 제어: 카트에 가하는 힘 [-Fmax, Fmax]
  - 목표: 진자를 아래에서 위로 스윙업해서 수직으로 세우고 유지
  - 특징: RK4 적분기, 간단한 구조, 렌더링 포함

  2. chain - 3D 질량-스프링 체인
  - 시스템: 연결된 질량-스프링 체인 (3D)
  - 상태: 각 질량의 3D 위치 + 속도
  - 제어: 첫 번째 자유 질량의 3D 속도
  - 목표: 마지막 질량을 목표 위치로 이동
  - 특징:
    - CasADi 기반 discrete dynamics (dynamics.py)
    - RestingChainSolver 사용
    - 타원체(Ellipsoid) 제약 포함

  3. hvac - 건물 냉난방 제어
  - 시스템: 3-state RC thermal model (실내/라디에이터/외벽 온도)
  - 상태: [Ti, Th, Te] + 시간/예측 정보
  - 제어: 라디에이터 열량 입력
  - 목표: 쾌적 온도 유지하면서 전기요금 최소화
  - 특징:
    - 실제 데이터 사용 (spot_prices.csv, weather.csv)
    - AR1 모델로 온도/일사량 예측 오차 모델링
    - 시간대별 쾌적 온도 범위 다름
    - acados_ocp.py 없음 (config.py로 대체)

  4. pointmass - 2D 네비게이션
  - 시스템: 바람 교란이 있는 2D 점 질량
  - 상태: [x, y, vx, vy] + wind forces
  - 제어: 2D 힘 [Fx, Fy]
  - 목표: 시작점 → 목표점 이동
  - 특징:
    - WindField 클래스로 교란 모델링
    - 난이도 레벨 (easy/hard)
    - Circle 클래스로 시작/목표 영역 정의

  5. race_cars (작업 중)
  - 시스템: Spatial bicycle model (레이스카)
  - 상태: [s, n, alpha, v, D, delta]
    - s: 중심선 따라 진행거리
    - n: 횡방향 편차
    - alpha: 헤딩 오차
    - v: 속도
    - D: 스로틀 상태
    - delta: 조향각
  - 제어: [derD, derDelta] (비율 입력)
  - 목표: 트랙 경계 내에서 주행
  - 특징:
    - 트랙 데이터 (LMS_Track.txt)
    - kappa_ref (곡률) 함수
    - time2spatial.py로 좌표 변환
    - plotFcn.py로 시각화

  ---
  race_cars 구현 시 참고할 점:
  - cartpole: RK4 적분, 간단한 구조
  - chain: CasADi 사용법, 복잡한 dynamics 분리
  - hvac: 실제 데이터 통합, 예측 기반 제어
  - pointmass: Matplotlib 렌더링, 영역/제약 처리