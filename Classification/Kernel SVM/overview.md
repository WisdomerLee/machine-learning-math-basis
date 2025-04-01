# 커널 서포트 벡터 머신(Kernel SVM) 이해하기

## 1. 커널 SVM이란 무엇인가요?

**커널 서포트 벡터 머신(Kernel SVM)** 은 선형으로 분리할 수 없는 데이터를 분류하기 위해, **SVM에 커널 트릭(Kernel Trick)** 을 적용한 버전입니다.

즉, 데이터를 **고차원으로 매핑**하여 그 공간에서는 선형 분리가 가능하도록 만든 뒤, **SVM으로 초평면을 찾는 방식**입니다.

---

## 2. 왜 커널이 필요한가요?

일상적인 데이터는 아래와 같이 선으로 분리되지 않을 수 있습니다.

```text
○ ○ ○ ● ● ●
○ ○ ○ ● ● ●
○ ○ ○ ● ● ●
```
이 경우, 일반적인 선형 SVM은 적절한 분리선을 찾을 수 없습니다.

그러나 데이터를 고차원으로 바꾸면, 다음과 같이 분리가 가능해집니다.

(고차원 공간에서 보면 선형 분리 가능)


이때 직접 데이터를 고차원으로 변환하지 않고, 커널 함수를 통해 내적(inner product) 만으로 계산할 수 있게 하는 것이 커널 트릭(Kernel Trick) 입니다.

## 3. 커널 함수란?
커널 함수는 두 벡터 간의 유사도를 측정하는 함수이며, 데이터를 고차원 공간으로 암묵적으로 매핑하는 역할을 합니다.

✔ 대표적인 커널 함수
|커널 종류|	수식 (간단하게)|	특징|
|---|---|---|
|선형 커널|	K(x, x') = x·x'|	선형 SVM과 동일|
|다항 커널|	K(x, x') = (x·x' + c)^d|	다항식 형태의 결정 경계|
|RBF 커널|	K(x, x') = exp(-γ‖x - x'‖²)|	국소적, 비선형 분류에 강력|
|시그모이드 커널|	K(x, x') = tanh(κx·x' + θ)|	신경망과 유사한 효과|

## 4. RBF 커널 (가장 자주 쓰임)
RBF (Radial Basis Function) 커널은 데이터 간의 거리(유클리드 거리)를 기반으로 유사도를 계산합니다.

가까운 점: 유사도 높음 → 큰 값

먼 점: 유사도 낮음 → 0에 가까운 값

γ (감마) 값의 역할
작은 γ: 넓은 영역에서 영향 → 부드러운 결정 경계

큰 γ: 좁은 영역에서 영향 → 복잡하고 민감한 결정 경계

## 5. 시각적으로 이해하기
📍 2차원에서 분리가 안 되는 예시
● ○ ● ○ ● ○
○ ● ○ ● ○ ●
고차원에서는 분리가 가능한 경우
데이터를 고차원으로 매핑하면, 원래 공간에서는 곡선 경계로 분리되지만
고차원 공간에서는 평면으로 분리됩니다.

## 6. Python 예제 (RBF 커널 사용)
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 데이터 로드 (아이리스 데이터)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 클래스 0과 1만 사용 (이진 분류)
X = X[y != 2]
y = y[y != 2]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVM 모델 (RBF 커널 사용)
model = SVC(kernel='rbf', gamma=0.5, C=1.0)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))

```
## 7. 하이퍼파라미터 정리
파라미터	설명
C	- 마진 폭과 오차 허용 사이의 균형 (규제 파라미터)
gamma -	RBF 커널에서 얼마나 가까운 데이터만 영향을 미치는지 조절

## 8. 정리
커널 SVM은 복잡한 데이터 분포도 잘 분류할 수 있는 강력한 방법입니다.

커널 함수를 통해 비선형 문제를 선형처럼 해결할 수 있습니다.

가장 많이 사용되는 커널은 RBF 커널입니다.

하지만 하이퍼파라미터 설정(C, gamma) 에 따라 성능이 크게 달라집니다.

