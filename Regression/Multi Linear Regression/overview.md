# 📘 다중 선형 회귀 (Multiple Linear Regression)

## 1. 회귀(Regression)란?

**회귀**는 어떤 **숫자 값을 예측하는 모델**입니다.  
예: 공부 시간에 따라 시험 점수를 예측

---

## 2. 단순 선형 회귀(Simple Linear Regression)

가장 기본적인 회귀로, 하나의 입력 변수만 사용하는 경우입니다.

[점수 = 공부시간 * a + b]

- 입력 변수: 공부시간
- 출력 변수: 점수

---

## 3. 다중 선형 회귀(Multiple Linear Regression)

현실 세계에서는 점수에 영향을 주는 요소가 하나뿐만이 아닙니다.

예:
- 공부시간
- 수면시간
- 수업 참여도
- 과거 평균 성적 등

이처럼 **여러 개의 입력 변수(독립 변수)**를 사용해 **하나의 출력 값(종속 변수)**을 예측하는 모델이 바로 **다중 선형 회귀**입니다.
[점수 = a1 * 공부시간 + a2 * 수면시간 + a3 * 수업참여도 + a4 * 과거성적 + b]


- a1, a2, ... : 각 입력 변수의 가중치 (회귀 계수)
- b : 절편 (intercept)

---

## 4. 시각적 이해

- **단순 선형 회귀**: 2차원 그래프 상의 직선
- **다중 선형 회귀**: 고차원 공간의 평면 또는 초평면

---

## 5. 예시 데이터

| 공부시간 | 수면시간 | 과거성적 | 점수 (예측 대상) |
|----------|----------|----------|------------------|
| 3        | 7        | 80       | 85               |
| 5        | 6        | 90       | 92               |
| 1        | 8        | 70       | 75               |

이 데이터를 기반으로 모델이 `점수`를 예측합니다.

---

## 6. Python 예시 코드

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# 데이터 준비
data = {
    '공부시간': [3, 5, 1],
    '수면시간': [7, 6, 8],
    '과거성적': [80, 90, 70],
    '점수': [85, 92, 75]
}
df = pd.DataFrame(data)

# 독립 변수(X), 종속 변수(y)
X = df[['공부시간', '수면시간', '과거성적']]
y = df['점수']

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측
new_data = [[4, 7, 85]]
predicted_score = model.predict(new_data)
print(f"예측된 점수: {predicted_score[0]:.2f}")
```

# 변수 선택 방법
## 1. All-in (전부 다 넣기)
✅ 개념:
모든 변수(피처)를 그대로 다 넣고 회귀 모델을 만드는 방법

📌 특징:
가장 간단한 방식

변수 선택 과정이 없음

하지만 불필요한 변수까지 포함되면 성능이 떨어질 수 있음

🎯 언제 사용?
변수가 적고, 모두 의미 있다고 생각될 때

## 2. Backward Elimination (뒤에서 제거하기)
✅ 개념:
처음엔 모든 변수를 다 넣고, 하나씩 **의미 없는 변수(통계적으로 유의하지 않은 변수)**를 제거해 나가는 방식

🔁 방법:
모든 변수 포함 → 모델 학습

변수의 p-value 계산

p-value가 가장 높은 변수(0.05보다 큰 값)를 제거

반복

p-value: 해당 변수가 의미 있는지 통계적으로 판단하는 지표
→ 작을수록 의미 있음 (보통 0.05 기준)

📌 특징:
중요한 변수만 남음

변수 수가 많을수록 유리

## 3. Forward Selection (앞에서 추가하기)
✅ 개념:
처음엔 아무 변수도 없이 시작, 성능이 좋아지는 변수를 하나씩 추가하는 방식

🔁 방법:
변수 없이 시작

모든 변수 중 가장 성능 향상 큰 변수 1개 선택

반복적으로 성능이 좋아지는 변수 추가

더 이상 성능 향상이 없을 때 종료

📌 특징:
변수 개수가 많고, 그중 일부만 중요할 때 유리

계산량이 많을 수 있음

## 4. Stepwise Selection (두 가지 혼합)
✅ 개념:
Forward와 Backward를 섞은 방식
변수를 추가하면서 동시에 필요 없는 변수는 제거

🔁 방법:
Forward 방식으로 변수 추가

추가 후, 기존 변수 중 다시 제거할 게 있는지 확인

성능 좋아질 때까지 반복

📌 특징:
가장 현실적인 방법

성능 개선이 뛰어남



다중 선형 회귀의 모델을 만드는 방법은 다섯가지가 있음
1. All-in
2. Backward Elimination
3. Forward Selection
4. Bidirectional Elimination
5. Score Comparison

2,3,4는 Stepwise Regression

대체로 Stepwise Regression이라고 하면 4번을 가리키는 경우가 많음

All-in은 모든 종속변수를 모두 넣는 방법
이 경우는 이미 해당 내용에 대해 모두 알고 있기 때문에 굳이 모델을 만들어야 할 필요가 없음
혹은 Backward Elimination을 시행하기 전의 준비 단계

Backward Elinination
모형 내의 유의미한 확률을 설정해야 함 - SL = 0.05로 하면 95%의 신뢰도를 갖는 확률을 설정
모델에서 가능한 종속변수들을 모두 넣기(All-in)
P-value값이 높은지 확인, 즉 SL보다 P값이 크면, 다음으로 진행, 그렇지 않으면 종료
연관성이 낮은 종속변수 제거
모델에 해당 변수 없이 맞추기 - 변수 하나를 없애는 순간 다른 변수들의 조건들이 모두 변동....

Forward Selection
모형 내의 유의미한 확률을 설정해야 함 - SL = 0.05로 하면 95%의 신뢰도를 갖는 확률을 설정
가능한 모든 단순 회귀 모형들을 넣을 것... P-값이 가장 낮은 독립 변수가 있는 모형을 하나 선정
해당 변수를 유지하고, 또 다른 단순 회귀 모형을 하나 더 추가할 것...
그리고 P-value값이 SL보다 낮으면, 다시 위의 과정을 반복하고, 그렇지 않으면 종료함

Bidirectional Eliminiation
SLENTER, SLSTAY 값을 설정
Forward Selection을 선택 - 새 variables는 P < SLENTER를 만족해야 함
Backward Elimination의 모든 과정을 진행 (기존 variables들은 P< SLSTAY를 만족해야 함)

All Possible Models
선형회귀를 잘 맞추기 위한 좋은 방법을 선택 (Akaike criterion같은 것)
모든 가능한 Regression models들을 만들기 (2^n -1)개의 조합 가능
그 중에 가장 최상의 criterion을 선택
