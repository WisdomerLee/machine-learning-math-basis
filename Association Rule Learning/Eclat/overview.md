# Eclat 알고리즘이란?

## 1. 개요

**Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal)**는  
연관 규칙 학습에서 사용되는 알고리즘으로,  
**빈발 아이템셋(Frequent Itemsets)**을 찾기 위해 **트랜잭션 ID 리스트(TID list)**를 활용합니다.

> 핵심 아이디어:  
> 아이템이 포함된 **거래 번호의 집합(TID Set)**을 사용하여  
> **교집합 연산**으로 빈발 아이템셋을 빠르게 계산한다!

---

## 2. 기존 알고리즘과의 차이

| 알고리즘 | 방식 | 장점 | 단점 |
|----------|------|------|------|
| Apriori | 아이템 조합 기반 반복 탐색 | 간단 | 느림 |
| FP-Growth | 트리 기반 압축 탐색 | 빠름 | 메모리 사용 많음 |
| Eclat | 트랜잭션 ID 기반 탐색 | 빠름 & 직관적 | 메모리 사용 많음 (TID 저장) |

---

## 3. Eclat 작동 방식

1. 각 아이템에 대해 **TID 집합** 생성  
   예: `Milk → {T1, T2, T3}`, `Bread → {T1, T2, T4}`

2. 아이템셋 확장 (교집합 기반)  
   예: `Milk ∩ Bread = {T1, T2}` → support = 2/5

3. Support 기준으로 **빈발 아이템셋** 필터링  
4. 재귀적으로 아이템을 확장하며 새로운 조합 탐색

---

## 4. 간단한 예시

```
거래 데이터:

T1: Milk, Bread  
T2: Milk, Bread, Eggs  
T3: Milk, Eggs  
T4: Bread  
T5: Milk, Bread, Eggs

TID 집합:
Milk  → {T1, T2, T3, T5}  
Bread → {T1, T2, T4, T5}  
Eggs  → {T2, T3, T5}

Milk ∩ Bread → {T1, T2, T5} → support = 3/5
```

---

## 5. Python 코드 예제 (직접 구현한 간단 버전)

> 현재 `mlxtend` 등 주요 라이브러리에서는 Eclat을 직접 지원하지 않기 때문에 간단한 구현으로 예시를 보여드릴게요.

```python
from collections import defaultdict

# 거래 데이터
transactions = [
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Bread'],
    ['Milk', 'Bread', 'Eggs']
]

# Step 1: TID 리스트 생성
item_tid = defaultdict(set)

for tid, transaction in enumerate(transactions):
    for item in transaction:
        item_tid[item].add(tid)

# Step 2: 아이템셋 교집합으로 빈발 아이템셋 생성
min_support = 0.6
num_transactions = len(transactions)
frequent_pairs = []

items = list(item_tid.keys())

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        item1, item2 = items[i], items[j]
        intersection = item_tid[item1] & item_tid[item2]
        support = len(intersection) / num_transactions
        if support >= min_support:
            frequent_pairs.append(((item1, item2), support))

# 결과 출력
for pair, support in frequent_pairs:
    print(f"{pair} -> support: {support:.2f}")
```

---

## 6. 장점과 단점

| 장점 | 단점 |
|------|------|
| 교집합 연산으로 속도 빠름 | TID 저장으로 메모리 사용 많음 |
| 계산량이 비교적 적음 | 희소하고 큰 데이터셋에는 부적합 |
| 트리 없이 구현 가능 | 연관 규칙 추출은 따로 구현 필요 |

---

## 7. Eclat은 언제 사용할까?

- 거래 데이터 수보다 **아이템 수가 적을 때**  
- 메모리 여유가 있고, 빠른 빈발 아이템셋 탐색이 필요할 때  
- 연관 규칙보다는 **빈발 패턴 분석**이 목표일 때

---

## 8. 요약

- **Eclat**은 **트랜잭션 ID 기반의 빈발 아이템셋 탐색 알고리즘**
- 교집합 연산을 통해 빠르게 Support 계산 가능
- Apriori보다 빠르고 FP-Growth보다 간단한 구조
- 메모리 이슈만 조심하면 효율적인 대안이 될 수 있음

---
