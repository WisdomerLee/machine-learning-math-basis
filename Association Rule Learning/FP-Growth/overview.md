# FP-Growth (Frequent Pattern Growth) 알고리즘

## 1. 개요

**FP-Growth**는 **연관 규칙 학습(Association Rule Learning)**에서 사용되는 알고리즘으로,  
**Apriori의 느린 속도 문제를 해결**하기 위해 등장했습니다.

> 핵심 아이디어:  
> **전체 데이터셋을 일일이 탐색하지 않고, 압축된 트리 구조(FP-Tree)를 사용해서 빈발 아이템셋을 효율적으로 찾자!**

---

## 2. 왜 FP-Growth인가?

| 항목 | Apriori | FP-Growth |
|------|---------|-----------|
| 방식 | 반복적인 후보 생성 & 스캔 | FP-Tree를 만들어 한 번에 처리 |
| 속도 | 느림 (조합이 많아질수록) | 빠름 (압축된 구조 활용) |
| 메모리 사용 | 적음 | 많을 수 있음 (FP-Tree 저장 필요) |

---

## 3. FP-Growth 작동 방식

### Step 1: 아이템별 지지도 계산 & 정렬  
- 각 아이템의 **support**를 구한 뒤, 높은 순서대로 정렬

### Step 2: FP-Tree 생성  
- 거래 데이터를 순서대로 정렬된 아이템들로 변환하고,  
  **공통된 경로는 트리의 같은 부분을 공유**하도록 FP-Tree에 저장

### Step 3: 조건부 FP-Tree 생성 & 빈발 패턴 추출  
- 각 아이템에 대해 **조건부 트리**를 만들어  
  재귀적으로 **빈발 패턴**을 탐색

---

## 4. Python 예제 (with `mlxtend`)

```bash
pip install mlxtend
```

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 거래 데이터
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
]

# One-hot 인코딩
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

# FP-Growth로 빈발 아이템셋 추출 (min_support=0.6)
frequent_items = fpgrowth(df, min_support=0.6, use_colnames=True)
print(frequent_items)

# 연관 규칙 추출
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.8)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

## 5. 장점과 단점

| 장점 | 단점 |
|------|------|
| 속도가 매우 빠름 | FP-Tree 생성 시 메모리 사용량 증가 가능 |
| 후보 아이템셋 생성 없음 | 트리 구조 이해가 다소 어려울 수 있음 |
| 대용량 데이터에 적합 | 희소하거나 다양한 아이템이 많으면 효과 ↓ |

---

## 6. 언제 사용할까?

- **수백만 건 이상의 거래 데이터**를 분석할 때
- **실시간 연관 규칙 탐색**이 필요한 경우
- **추천 시스템, 로그 분석, 보안 이상 탐지** 등에 유용

---

## 7. FP-Growth vs Apriori 한눈에 비교

| 항목 | Apriori | FP-Growth |
|------|---------|-----------|
| 탐색 방식 | 반복 스캔 | 트리 기반 탐색 |
| 속도 | 느림 | 빠름 |
| 구조 | 리스트 기반 | 트리 기반 |
| 메모리 | 상대적으로 적게 사용 | 상대적으로 많이 사용 |
| 장점 | 구현이 간단 | 성능이 우수 |

---

## 8. 요약

- **FP-Growth**는 Apriori보다 훨씬 효율적으로 빈발 아이템셋을 찾는다.
- **FP-Tree 구조**를 통해 거래를 압축 저장하고 빠르게 패턴을 탐색한다.
- 대용량 데이터 분석 시 유용하며, `mlxtend` 같은 Python 라이브러리로 쉽게 사용할 수 있다.

---
