# 랜덤 포레스트(Random Forest) 이해하기

## 1. 랜덤 포레스트란 무엇인가요?

**랜덤 포레스트(Random Forest)** 는 여러 개의 **의사결정 트리(Decision Tree)** 를 모아서 하나의 강력한 모델을 만드는 **앙상블 학습(Ensemble Learning)** 방법입니다.

즉, **많은 나무(트리)들을 만들고**, 각각이 투표하거나 평균을 내는 방식으로 **더 나은 예측**을 하도록 도와줍니다.

> 🎯 핵심 아이디어: "여러 개의 약한 학습기(weak learners)를 모아 강한 학습기(strong learner)를 만든다!"

---

## 2. 왜 "랜덤"인가요?

랜덤 포레스트는 **두 가지 방법으로 무작위성(randomness)** 을 도입합니다:

### 🌱 1. 부트스트래핑(Bootstrap)

- 학습 데이터를 무작위로 **샘플링(with replacement)** 해서 **각각의 트리**를 학습시킵니다.
- 즉, 모든 트리가 같은 데이터를 보는 것이 아니라 **부분집합**을 봅니다.

### 🍃 2. 랜덤 특성 선택(Feature Bagging)

- 각 트리의 분기(split)에서, 전체 특성 중 일부만 **무작위로 선택**해서 최적 분할을 결정합니다.
- 이를 통해 트리들 간의 **다양성**을 증가시킵니다.

---

## 3. 동작 방식 요약

```text
1. 데이터 샘플링 → 여러 개의 데이터 부분집합 생성 (부트스트랩)
2. 각 부분집합으로 결정 트리 훈련
3. 각 트리는 예측 결과를 생성
4. 분류 문제 → 투표(Voting), 회귀 문제 → 평균(Averaging)
```
## 4. 장점과 단점
✅ 장점
과적합에 강함 (여러 트리의 예측을 평균)

다양한 데이터 타입에 잘 작동

자동으로 특성 선택 효과

해석 가능한 부분도 존재 (특성 중요도 등)

❌ 단점
모델이 크고 느림 (많은 트리 사용)

개별 트리보다 덜 직관적

너무 많은 트리는 추론 속도 저하

## 5. Python 예제 (scikit-learn)
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 예측 및 정확도 평가
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))

```

## 6. 중요한 하이퍼파라미터
파라미터	설명
n_estimators	생성할 트리 수 (많을수록 안정적이지만 느려짐)
max_depth	각 트리의 최대 깊이 (과적합 방지용)
max_features	각 분기에서 사용할 특성의 수
bootstrap	부트스트랩 샘플링 여부 (True가 일반적)

## 7. 특성 중요도 보기
랜덤 포레스트는 각 특성이 예측에 얼마나 기여했는지를 알려주는 기능도 제공합니다.
```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = iris.feature_names

plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

```
## 8. 정리
랜덤 포레스트는 여러 결정 트리를 결합해 안정적이고 강력한 예측을 제공하는 분류/회귀 알고리즘입니다.

무작위성 도입을 통해 트리들 간의 상관관계를 줄이고 성능을 향상시킵니다.

대부분의 경우, 의사결정 트리보다 성능이 좋고 과적합에 덜 민감합니다.
