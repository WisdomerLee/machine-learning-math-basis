# 📘 다항 회귀 (Polynomial Regression)

## 1. 다항 회귀란?

**다항 회귀(Polynomial Regression)**는 **선형 회귀(Linear Regression)**의 확장 버전입니다.

기본 선형 회귀는 입력 변수(x)와 출력 변수(y)가 **직선적인 관계**에 있다고 가정합니다.  
하지만 현실 세계에서는 곡선적인 관계가 훨씬 더 많죠!

### ✅ 예:
> 자동차 속도(x)와 연비(y)의 관계는 곡선일 수 있습니다.  
> → 속도가 너무 낮거나 너무 높으면 연비가 떨어지고, 중간에서 최고 효율이 나오는 식.

---

## 2. 선형 회귀 vs 다항 회귀

| 구분           | 선형 회귀                  | 다항 회귀                          |
|----------------|----------------------------|-------------------------------------|
| 수식 형태      | y = a * x + b              | y = a1 * x + a2 * x² + a3 * x³ + ... + b |
| 관계           | 직선 관계                  | 곡선 관계 (비선형)                 |
| 복잡도         | 단순                       | 더 복잡하고 유연함                 |

---

## 3. 다항 회귀의 수식 형태

다항 회귀는 다음과 같은 형태를 가집니다:

y = a1 * x + a2 * x² + a3 * x³ + ... + an * xⁿ + b

- `x`, `x²`, `x³`, ..., `xⁿ` → 각각을 **다항항**이라고 부릅니다.
- `n`이 클수록 더 복잡한 곡선을 만들 수 있음

---

## 4. 시각적 이해

- **선형 회귀**는 모든 점들을 최선의 직선으로 연결하려 함
- **다항 회귀**는 최선의 곡선(곡선의 차수에 따라 다양)을 찾아냄

데이터가 곡선을 따라 분포되어 있다면 선형 회귀보다 다항 회귀가 더 잘 맞음


---

## 5. Python 예제 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 데이터 생성
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # 완벽한 x^2 형태

# 다항 특성 변환 (2차)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

# 모델 훈련
model = LinearRegression()
model.fit(X_poly, y)

# 예측
x_test = np.linspace(1, 5, 100).reshape(-1, 1)
x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)

# 시각화
plt.scatter(x, y, color='red', label='실제 데이터')
plt.plot(x_test, y_pred, label='다항 회귀 곡선')
plt.legend()
plt.title("Polynomial Regression (2차)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
```
## 6. 다항 회귀의 차수 선택
차수가 너무 낮으면 곡선을 충분히 표현 못함 (언더피팅)

차수가 너무 높으면 데이터에 너무 민감해짐 (오버피팅)

→ 일반적으로 2~4차 정도가 가장 많이 쓰이며, 성능을 보고 선택합니다.

## 7. 다항 회귀도 선형 회귀다?
✅ 다항 회귀는 비선형 모델처럼 보이지만,
사실 x, x², x³ 등을 새로운 변수로 바꿔서 학습하기 때문에
선형 회귀 알고리즘으로 학습 가능합니다!

즉, **"변수는 비선형이지만 계수는 선형"**입니다.

