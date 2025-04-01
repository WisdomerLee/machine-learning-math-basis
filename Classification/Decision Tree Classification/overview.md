# 의사결정 트리(Decision Tree) 이해하기

## 1. 의사결정 트리란 무엇인가요?

**의사결정 트리(Decision Tree)** 는 분류(classification)와 회귀(regression)에 모두 사용 가능한 **지도 학습 알고리즘**입니다. 마치 **"20 Questions"** 게임처럼 질문을 던지며 데이터를 나누어 최종 분류를 수행합니다.

트리 구조로 되어 있어서, **루트 노드(Root Node)에서 시작해 조건에 따라 가지(Branch)를 따라가며**, **리프 노드(Leaf Node)** 에 도달하면 분류 결과가 나옵니다.

---

## 2. 핵심 개념

### 🌳 노드(Node)

- **루트 노드(Root)**: 트리의 시작점
- **내부 노드(Internal Node)**: 조건 분기점
- **리프 노드(Leaf Node)**: 최종 분류 결과

### 🌿 분할(Splitting)

- 데이터를 **특성(feature)** 값을 기준으로 나누는 과정
- 목표: **데이터를 가능한 한 순수하게 나누는 것**

### 🔍 순도(Purity)

- 하나의 노드에 동일한 클래스의 데이터만 있을수록 순도가 높음
- 의사결정 트리는 각 분할에서 순도를 최대한 높이려 함

---

## 3. 분할 기준 (지니 지수 vs 정보 이득)

### ✔ 지니 지수 (Gini Impurity)

- “랜덤하게 뽑은 두 개의 데이터가 서로 다를 확률”
- 값이 **0에 가까울수록 순도 높음**
- 계산이 빠르기 때문에 **기본값으로 자주 사용**

### ✔ 정보 이득 (Information Gain)

- 분할 전과 후의 **엔트로피(entropy)** 차이
- 엔트로피: 불확실성 측정 지표 (값이 클수록 혼잡함)
- **ID3 알고리즘**에서 사용됨

---

## 4. 시각적으로 이해하기

예시: 키와 체중에 따라 '운동선수'인지 분류

```text
[루트 노드]
키 < 180?
├── 예: [왼쪽 자식] 체중 < 70?
│       ├── 예: ❌ (비운동선수)
│       └── 아니오: ✅ (운동선수)
└── 아니오: ✅ (운동선수)
```
## 5. 장점과 단점
✅ 장점
이해하기 쉬움 (모델이 직관적)

전처리 적음 (정규화나 스케일링 불필요)

범주형, 수치형 모두 사용 가능

모델 시각화 가능

❌ 단점
과적합(overfitting) 되기 쉬움

작은 변화에도 구조가 크게 바뀔 수 있음

불균형 데이터에 민감

→ 이를 보완하기 위해 랜덤 포레스트(Random Forest) 같은 앙상블 기법이 자주 사용됨


## 6. Python 예제 (scikit-learn 사용)
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 의사결정 트리 모델 생성
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(X_train, y_train)

# 예측 및 정확도
y_pred = clf.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))

```
## 7. 하이퍼파라미터 요약
|파라미터|	설명|
|---|---|
|criterion|	분할 기준 ("gini" 또는 "entropy")|
|max_depth|	트리의 최대 깊이 제한|
|min_samples_split|	내부 노드가 분할되기 위한 최소 샘플 수|
|min_samples_leaf|	리프 노드가 되기 위한 최소 샘플 수|

## 8. 결정 트리 시각화 (옵션)
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

```
## 9. 결론
의사결정 트리는 직관적이며 해석하기 쉬운 강력한 분류 모델입니다.

하지만 과적합에 취약하기 때문에 하이퍼파라미터 조정이 중요합니다.

성능을 더 높이고 싶다면 랜덤 포레스트나 그래디언트 부스팅과 같은 앙상블 모델을 고려해볼 수 있습니다.
