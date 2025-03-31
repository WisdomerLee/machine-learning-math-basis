# 📘 랜덤 포레스트 회귀 (Random Forest Regression)

## 1. 랜덤 포레스트 회귀란?

**Random Forest Regression**은 여러 개의 **결정 트리(Decision Trees)**를 만들고,  
그 결과들을 **평균내어 예측하는 모델**입니다.

> 🌲 여러 개의 나무(Tree)를 만들어서 숲(Forest)을 만든다는 의미에서 Random Forest라는 이름이 붙었습니다.

---

## 2. 어떻게 작동하나요?

1. **여러 개의 결정 트리**를 훈련 데이터로 생성합니다.
2. 각 트리는 데이터의 일부(샘플링)를 사용해 독립적으로 학습합니다.
3. 예측 시에는 **각 트리의 예측값을 평균**하여 최종 예측값을 만듭니다.

> 🎯 단일 트리보다 훨씬 더 안정적이고 예측력이 좋습니다.

---

## 3. 왜 “랜덤”인가요?

두 가지 랜덤 요소를 사용합니다:

1. **랜덤 샘플링**: 전체 데이터에서 일부만 뽑아서 각 트리를 학습
2. **랜덤 특성 선택**: 각 분기에서 일부 특성만 보고 분할 결정

→ 이렇게 함으로써 **다양한 트리**를 만들고, 과적합(overfitting)을 줄일 수 있습니다!

---

## 4. 결정 트리 회귀와 비교

| 항목                  | 결정 트리 회귀             | 랜덤 포레스트 회귀         |
|-----------------------|----------------------------|-----------------------------|
| 예측 방식             | 하나의 트리                 | 여러 트리의 평균             |
| 과적합 가능성         | 높음                        | 낮음 (앙상블 효과)         |
| 안정성                | 데이터 변화에 민감함        | 데이터 변화에 덜 민감함     |
| 해석력                | 쉬움 (구조 해석 가능)       | 해석이 다소 어려움          |
| 예측 성능             | 보통                        | 보통 이상 (강력한 성능)     |

---

## 5. Python 예제 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 데이터 생성
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 5, 7, 10, 15, 16, 17, 18, 19, 20])

# 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
model.fit(X, y)

# 예측
X_test = np.linspace(1, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# 시각화
plt.scatter(X, y, color='red', label='실제 데이터')
plt.plot(X_test, y_pred, color='blue', label='랜덤 포레스트 예측')
plt.title("Random Forest Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
```

## 6. 주요 하이퍼파라미터
|파라미터|	설명|
|---|---|
|n_estimators|	트리의 개수 (많을수록 성능 ↑, 속도 ↓)|
|max_depth|	각 트리의 최대 깊이|
|min_samples_split|	분기할 최소 샘플 수|
|random_state|	결과 재현을 위한 난수 설정|

## 7. 장점과 단점
✅ 장점
과적합에 강함

비선형 문제도 잘 처리

변수 선택 없이도 자동 처리 (변수 중요도 확인 가능)

실무에서 널리 사용됨

⚠️ 단점
단일 트리보다 해석이 어려움

학습 속도와 예측 속도가 느릴 수 있음 (트리가 많으면)

매우 큰 데이터셋에는 메모리 소모가 큼
