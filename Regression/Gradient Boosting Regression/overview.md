# 📘 그래디언트 부스팅 회귀 (Gradient Boosting Regression)

## 1. Gradient Boosting Regression이란?

**Gradient Boosting Regression**은 여러 개의 **약한 예측기(weak learners)**  
(보통은 결정 트리)를 **순차적으로 연결해서** 점점 더 정교한 예측을 만드는 회귀 모델입니다.

> 🔁 이전 모델의 **오차를 줄이는 방향**으로 계속 학습해 나가는 방식!

---

## 2. 어떻게 작동하나요?

1. **첫 번째 모델**이 데이터를 예측
2. **오차(실제값 - 예측값)를 계산**
3. 이 오차를 예측하기 위한 **새로운 모델**을 학습
4. 두 모델의 결과를 **합산**
5. 이 과정을 여러 번 반복하여 성능을 점점 향상

### 📌 예시 흐름

y = 예측1 + 예측2 + 예측3 + ...

→ 이렇게 하나씩 모델을 추가하면서 **잔차(오차)**를 계속 줄여가는 방식

---

## 3. 랜덤 포레스트와의 차이점

| 항목                  | 랜덤 포레스트               | 그래디언트 부스팅            |
|-----------------------|------------------------------|-------------------------------|
| 트리 훈련 방식        | 병렬(Parallel)               | 순차(Sequential)              |
| 목적                  | 각 트리의 예측을 평균       | 오차를 줄이는 방향으로 개선   |
| 과적합 방지           | 비교적 쉬움                  | 과적합 위험 높아 관리 필요     |
| 학습 속도             | 빠름                          | 느림                          |
| 예측 성능             | 높음                         | 매우 높음 (튜닝 시 탁월)      |

---

## 4. 왜 "Gradient"?

오차를 줄이기 위해 **경사하강법(Gradient Descent)**을 사용해서  
"어떤 방향으로 모델을 수정해야 예측이 좋아질까?"를 계산합니다.

즉, **손실 함수의 기울기**를 따라 조금씩 개선해 나가는 거예요!

---

## 5. Python 예제 코드 (with scikit-learn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 데이터 생성
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 5, 7, 10, 15, 16, 17, 18, 19, 20])

# 모델 생성 및 학습
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
model.fit(X, y)

# 예측
X_test = np.linspace(1, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# 시각화
plt.scatter(X, y, color='red', label='실제 데이터')
plt.plot(X_test, y_pred, color='green', label='Gradient Boosting 예측')
plt.title("Gradient Boosting Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
```

## 6. 주요 하이퍼파라미터
|파라미터|	설명|
|---|---|
|n_estimators|	몇 개의 약한 모델(트리)을 쌓을지|
|learning_rate|	매번 오차 수정할 때 반영 비율 (작을수록 천천히 학습)|
|max_depth|	각 트리의 최대 깊이 (복잡도 조절)|
|subsample|	부스팅에 사용할 데이터 비율 (0.5~1.0)|

## 7. 장점과 단점
✅ 장점
매우 강력한 회귀 모델 (성능 우수)

비선형 관계, 복잡한 패턴도 잘 예측

하이퍼파라미터 튜닝 시 높은 정밀도

⚠️ 단점
학습 속도가 느림 (트리를 순차적으로 학습)

과적합 위험이 있음 (learning_rate를 작게, early stopping 활용)

해석이 어렵고, 트리 구조가 복잡함
