# 📘 서포트 벡터 회귀 (Support Vector Regression, SVR)

## 1. SVR이란?

**SVR**은 SVM(Support Vector Machine)을 회귀 문제에 응용한 모델입니다.  
즉, **Support Vector Machine의 회귀 버전**이라고 보면 됩니다.

- SVM: 분류(Classification)에 자주 사용
- **SVR**: 숫자 예측(Regression)에 사용

---

## 2. 선형 회귀와의 차이점은?

일반적인 선형 회귀는 **모든 데이터 포인트와의 오차를 최소화**하려고 합니다.  
하지만 **SVR은**:

> "데이터가 허용 가능한 범위 안에 있으면 오차로 간주하지 않고,  
> 그 범위를 벗어난 데이터만 학습에 사용한다."

이렇게 오차에 **유연한 경계선(마진)**을 둡니다.

---

## 3. SVR의 개념 (epsilon-tube)

- SVR은 예측 선 주변에 **ε(엡실론)이라는 폭을 가진 튜브**를 만듭니다.
- 이 **ε 튜브 안에 들어오는 데이터 포인트는 오차가 없는 것으로 간주**합니다.
- 튜브 밖에 있는 점들만 오차로 계산하여 **최적의 평면**을 찾습니다.

### 🎯 목표:
> ε 튜브 안에 최대한 많은 데이터를 넣고,  
> 튜브 밖 데이터의 오차는 최소화!

---

## 4. 시각적 이미지로 비유

- 선형 회귀: 한 줄을 긋고, 모든 점들과의 거리(오차)를 최소화
- SVR: 예측선을 중심으로 일정 폭의 튜브를 만들고,  
  그 안에 최대한 많은 데이터를 포함시킴

---

## 5. SVR의 주요 하이퍼파라미터

| 파라미터  | 설명 |
|-----------|------|
| **C**     | 오차 허용 한계. 클수록 오차를 적게 하려 함 (복잡도 증가) |
| **ε (epsilon)** | 튜브의 폭. 작을수록 정밀, 클수록 여유 있음 |
| **kernel** | 데이터를 고차원으로 변환하는 방식 (linear, poly, rbf 등) |

---

## 6. Python 예제 코드 (with scikit-learn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 데이터 생성
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # 노이즈 포함한 sin 함수

# 모델 정의 (RBF 커널)
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X, y)

# 예측
y_pred = model.predict(X)

# 시각화
plt.scatter(X, y, color='gray', label='데이터')
plt.plot(X, y_pred, color='blue', label='SVR 예측')
plt.title("Support Vector Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
```

## 7. 커널 함수 (Kernel Trick)
SVR은 커널 트릭을 사용해서 비선형 회귀도 가능하게 합니다.

|커널 종류|	설명|
|---|---|
|linear|	선형 회귀와 유사|
|poly|	다항 회귀처럼 곡선 표현|
|rbf|	가장 일반적인 비선형 커널 (곡선 형태에 강함)|
