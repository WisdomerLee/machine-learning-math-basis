# Support Vector Machine (SVM) 이해하기

## 1. SVM이란 무엇인가요?

**Support Vector Machine (SVM)**은 분류(classification)와 회귀(regression)에 모두 사용할 수 있는 **지도 학습(Supervised Learning)** 알고리즘입니다. 특히 **이진 분류(binary classification)** 문제에서 매우 강력한 성능을 보입니다.

SVM은 데이터들을 분류하기 위해 **가장 넓은 여백(margin)을 가진 경계선(초평면, hyperplane)** 을 찾는 것이 핵심 아이디어입니다.

---

## 2. 핵심 개념

### 🔹 초평면 (Hyperplane)

- n차원 공간에서 데이터를 나누는 경계입니다.
- 2차원에서는 **선(line)**, 3차원에서는 **평면(plane)**, 그 이상은 **초평면(hyperplane)** 이라고 부릅니다.

### 🔹 마진 (Margin)

- 데이터를 나누는 초평면과 각 클래스에서 가장 가까운 데이터 점 사이의 거리입니다.
- SVM은 이 마진이 **최대한 넓어지도록** 초평면을 설정합니다. 이것을 **최대 마진 분류기(Maximum Margin Classifier)** 라고 합니다.

### 🔹 서포트 벡터 (Support Vector)

- 마진에 가장 가까이 있는 데이터 포인트들을 말합니다.
- 이 서포트 벡터들이 초평면의 위치를 결정하며, SVM은 이 점들만을 이용해 학습합니다.

---

## 3. 시각적으로 이해하기

2차원 데이터를 예로 들어 보겠습니다.

```text
클래스 1 (●)      클래스 2 (○)

●    ●             ○      ○
      ●       ↑     ○
             초평면
      ●       ↓     ○
●    ●             ○      ○
```
SVM은 위의 두 클래스 사이에 있는 초평면을 찾아서 분리하며, 마진을 최대화합니다.

## 4. 선형 vs 비선형 SVM

✔ 선형 SVM
데이터가 직선(또는 초평면)으로 완벽하게 나뉠 수 있을 때 사용합니다.

예: 직선으로 두 클래스를 쉽게 나눌 수 있는 경우

✔ 비선형 SVM (Kernel Trick 사용)
현실에서는 대부분의 데이터가 선형적으로 구분되지 않습니다.

커널 함수(Kernel Function) 를 사용해 데이터를 고차원으로 매핑한 후, 그 공간에서 선형 분리를 시도합니다.

대표적인 커널 함수
선형 커널 (Linear Kernel)

다항 커널 (Polynomial Kernel)

RBF 커널 (Radial Basis Function) ← 가장 많이 사용됨

시그모이드 커널 (Sigmoid Kernel)

## 5. SVM의 장단점
✅ 장점
고차원 데이터에서도 잘 작동

적은 수의 서포트 벡터로 모델 결정 → 메모리 효율적

일반화 능력이 뛰어남

❌ 단점
큰 데이터셋에서는 느릴 수 있음

커널과 하이퍼파라미터 선택이 민감함

다중 클래스 분류에서는 직접 사용이 어려움 (One-vs-One, One-vs-Rest 방식 필요)

## 6. 간단한 Python 예제 (scikit-learn)
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터 로드 (아이리스 데이터)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 이진 분류만 위해 클래스 0과 1만 사용
X = X[y != 2]
y = y[y != 2]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVM 모델 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 예측 및 정확도 출력
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
```
## 7. 결론
SVM은 단순하면서도 강력한 분류 알고리즘입니다.

특히 마진을 최대화하여 일반화 성능이 좋습니다.

다양한 커널을 통해 비선형 데이터도 잘 처리할 수 있습니다.

📚 참고 키워드
SVM (Support Vector Machine)

Hyperplane

Margin

Support Vectors

Kernel Trick

Scikit-learn
