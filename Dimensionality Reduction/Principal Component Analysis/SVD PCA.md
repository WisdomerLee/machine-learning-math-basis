# 핵심 요약

> **PCA = SVD 기반 특이값 분해를 이용해 주성분을 추출하는 차원 축소 기법**

- PCA의 수학적 핵심은:  
  → 데이터를 가장 잘 설명하는 **직교 벡터 방향**(=주성분)을 찾는 것  
- 이 방향은 **SVD로 매우 안정적**이고 정확하게 구할 수 있어요.

---

# SVD란?

> SVD는 행렬을 세 개의 행렬로 분해하는 방법입니다.

수학적으로:

$$
X = U \Sigma V^T
$$

- $X$: 정규화된 입력 데이터 행렬 (n × d)
- $U$: 좌측 직교 행렬 (n × n)
- $\Sigma$: 대각선 형태의 특이값 행렬 (n × d)
- $V^T$: 우측 직교 행렬 (d × d)

👉 여기서 **$V$**의 열벡터들이 바로 PCA의 **주성분 벡터**예요!

---

# SVD와 PCA의 관계

| PCA 단계             | SVD에서 어떻게 나타나는가?         |
|----------------------|-------------------------------------|
| 공분산 행렬 고유벡터 | $V$: 주성분 (principal components) |
| 고유값               | $\Sigma^2 / (n-1)$              |
| 주성분 투영 결과     | $X \cdot V_k$                   |

즉,  
- 고전적인 방식: 공분산 행렬 → 고유값 분해  
- 실전에서는: **SVD를 통해 더 빠르고 수치적으로 안정적인 계산**

---

# SVD 기반 PCA 직접 구현 (넘파이 사용)

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 로드 및 정규화
X = load_iris().data
y = load_iris().target
X_std = StandardScaler().fit_transform(X)

# SVD 수행
U, S, VT = np.linalg.svd(X_std)

# 주성분 추출 (V의 전치 행렬 → VT)
PCs = VT.T[:, :2]  # 상위 2개 주성분

# 데이터 투영 (X × 주성분)
X_pca = X_std @ PCs

# 시각화
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', s=40)
plt.title("PCA via SVD (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='class')
plt.grid(True)
plt.show()
```

---

# SVD 기반 PCA의 장점

| 장점                     | 설명 |
|--------------------------|------|
| 안정적                   | 수치 오차에 강하고 고차원 데이터에도 잘 작동 |
| 빠름                     | 대규모 sparse 행렬에도 사용 가능 |
| 공분산 행렬을 만들 필요 없음 | SVD는 데이터 직접 분해 |

---

# 언제 유용할까?

- 데이터의 특성 수(features)가 샘플 수보다 많은 경우 (`d > n`)
- 공분산 행렬을 직접 계산하기 어려운 경우
- 대규모 행렬 처리 (예: 텍스트 TF-IDF, 이미지 압축 등)

---

# 추가: `sklearn` PCA도 내부적으로 SVD를 사용합니다

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
```

`PCA(svd_solver='full')`이 기본이며, 내부적으로 `SVD`를 이용해 주성분을 계산합니다.

---
