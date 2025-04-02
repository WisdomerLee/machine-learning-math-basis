# Apriori 알고리즘
## 1. 개요
Apriori는 대량의 거래 데이터에서 항목 간의 연관 관계를 찾아주는 알고리즘입니다.
예: "맥주를 산 사람은 안주도 살 확률이 높다."

→ 이처럼, 자주 함께 발생하는 아이템의 조합을 찾아내고, 그 관계를 **연관 규칙(Association Rule)**으로 만드는 것이 목적입니다.

## 2. 주요 개념
### 1) Support (지지도)
어떤 아이템셋이 전체 거래에서 얼마나 자주 등장하는지

계산: Support(A) = 거래 중 A가 포함된 비율

### 2) Confidence (신뢰도)
A를 구매한 사람이 B도 함께 구매할 확률

계산: Confidence(A → B) = Support(A ∪ B) / Support(A)

### 3) Lift (향상도)
A가 B의 구매에 얼마나 영향을 주는지 (1보다 크면 양의 연관)

계산: Lift(A → B) = Confidence(A → B) / Support(B)

## 3. Apriori 알고리즘 작동 방식
최소 지지도(min_support) 이상인 **빈발 아이템셋(Frequent Itemsets)**을 찾는다.

빈발 아이템셋을 기반으로 **신뢰도(confidence)**가 높은 연관 규칙을 생성한다.

선택적으로 향상도(lift)를 기준으로 유의미한 규칙만 필터링한다.

핵심 아이디어: "모든 부분집합도 빈발해야 전체가 빈발할 수 있다"
(Apriori 속성이라고 부름)

## 4. Python 예제
```bash
pip install mlxtend
```

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 거래 데이터 (One-hot encoding 형태)
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
]

# 데이터프레임 변환
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# 빈발 아이템셋 추출 (지지도 최소 0.6)
frequent_items = apriori(df, min_support=0.6, use_colnames=True)
print(frequent_items)

# 연관 규칙 생성 (신뢰도 최소 0.8)
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.8)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

```

## 5. 장점과 단점
장점

구현이 간단하고 해석이 쉬움
다양한 품목 간 관계를 찾기 유용


단점

계산량이 많고 느림 (부분집합을 모두 생성)
고차원 데이터에선 성능 저하
## 6. Apriori는 언제 쓰일까?
마트, 온라인 쇼핑몰의 장바구니 분석

영화/음악의 추천 시스템

의학 데이터에서 증상-질병 간 연관 규칙 찾기

보안 데이터에서 이상 행위 패턴 탐지

## 7. 요약
Apriori는 거래 데이터에서 자주 발생하는 아이템 조합을 찾아 연관 규칙으로 만든다.

Support, Confidence, Lift 세 가지 지표로 규칙을 평가한다.

대용량일수록 속도 개선을 위한 다른 알고리즘(FP-Growth 등)이 고려된다.
