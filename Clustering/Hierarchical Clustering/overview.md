# Hierarchical Clustering 이해하기
## 1. Hierarchical Clustering이란?
Hierarchical Clustering(계층적 군집화)은 데이터들을 유사도에 따라 계층적으로 묶어가는 군집화 기법입니다. 이름 그대로 나무(tree) 형태의 구조를 만들기 때문에 결과를 시각적으로 **덴드로그램(Dendrogram)**으로 표현할 수 있습니다.

## 2. 군집화(Clustering)란?
군집화는 비지도 학습(Unsupervised Learning)의 한 방법으로, 비슷한 데이터끼리 그룹화하는 작업입니다.

예: 쇼핑몰 고객 데이터를 바탕으로 구매 성향이 비슷한 사람들을 묶는다.

## 3. Hierarchical Clustering의 종류
Agglomerative (병합적) 방법
→ 가장 많이 쓰이며, bottom-up 방식
→ 각 데이터를 하나의 클러스터로 시작해서, 유사한 클러스터끼리 병합해 나감

Divisive (분할적) 방법
→ top-down 방식
→ 전체 데이터를 하나의 클러스터로 보고, 점점 분리해 나감

## 4. 클러스터 간 유사도 측정 방법 (Linkage)
Single Linkage: 가장 가까운 두 점 사이의 거리

Complete Linkage: 가장 먼 두 점 사이의 거리

Average Linkage: 모든 점 사이의 평균 거리

Ward Linkage: 클러스터 내 분산의 증가량 기준

## 5. Dendrogram (덴드로그램)
클러스터링 과정을 나무 구조로 시각화한 그래프

Y축은 두 클러스터 간의 거리(또는 비용)를 의미

적절한 위치에서 수평으로 자르면 원하는 클러스터 개수로 나눌 수 있음

## 6. Python 예제 코드
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 데이터 생성
X, _ = make_blobs(n_samples=30, centers=3, random_state=42)

# 계층적 클러스터링 수행
Z = linkage(X, method='ward')

# 덴드로그램 시각화
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

# 클러스터 결과 얻기 (3개로 자름)
labels = fcluster(Z, t=3, criterion='maxclust')

# 클러스터링 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Hierarchical Clustering Result")
plt.show()

```
## 7. 주요 특징 정리
|특징|	설명|
|---|---|
|거리 기반|	유사도(거리)에 따라 클러스터링|
|계층 구조|	덴드로그램으로 시각화 가능|
|사전 군집 수 필요 없음|	덴드로그램을 보고 나중에 결정 가능|
|계산 복잡도|	데이터 수가 많아지면 시간이 오래 걸림|

대규모 데이터셋에는 적용하기 부적절
