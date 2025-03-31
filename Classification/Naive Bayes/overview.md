# 🧠 Naive Bayes 분류기 정리 & 실습 (Python)

## 📌 Naive Bayes란?

Naive Bayes는 **확률 이론(베이즈 정리)**에 기반한 **지도 학습 분류 알고리즘**입니다.  
**모든 특성(Feature)이 서로 독립적**이라는 가정을 기반으로 작동하기 때문에 "Naive(순진한)"이라는 이름이 붙었습니다.

---

## 🔧 베이즈 정리

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

- **P(A|B)**: B라는 정보가 주어졌을 때 A일 확률 (Posterior)
- **P(B|A)**: A일 때 B가 나타날 확률 (Likelihood)
- **P(A)**: A가 발생할 확률 (Prior)
- **P(B)**: B가 발생할 확률 (Evidence)

---

## 📈 Naive Bayes 분류의 핵심

특성들이 주어졌을 때, 각 클래스가 될 확률을 계산하고,  
**가장 확률이 높은 클래스를 선택**합니다.

예시:
```text
P(스팸 | [무료, 지금, 쿠폰]) = P(무료|스팸) × P(지금|스팸) × P(쿠폰|스팸) × P(스팸)
```

|장점|단점|
|---|---|
|빠르고 간단함|특성 간 독립 가정이 비현실적일 수 있음|
|적은 데이터에도 작동|	복잡한 상호작용을 반영 못함|
|텍스트 분류에 강함|	연속적인 특성 처리 한계 있음|

```bash
pip install scikit-learn
```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 학습 데이터
texts = [
    "지금 할인 이벤트 진행 중",    # 스팸
    "무료 쿠폰 드려요",          # 스팸
    "오늘 회의 일정 공유합니다",    # 정상
    "안녕하세요, 내일 점심 어때요?", # 정상
    "지금 바로 클릭하세요",       # 스팸
    "이번 주말에 영화 볼래요?"     # 정상
]
labels = [1, 1, 0, 0, 1, 0]  # 1: 스팸, 0: 정상

# 모델 구성 및 학습
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# 테스트 데이터
test_msg = [
    "무료 영화 쿠폰 받기",
    "회의는 내일 오후 2시에 시작합니다",
    "지금 가입하고 혜택 받기"
]
predicted = model.predict(test_msg)

# 결과 출력
for msg, label in zip(test_msg, predicted):
    print(f"'{msg}' -> {'스팸' if label == 1 else '정상'}")
```
