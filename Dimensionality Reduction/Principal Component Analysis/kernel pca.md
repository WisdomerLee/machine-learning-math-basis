# Kernel PCA란?

> **비선형 PCA**.  
> 데이터를 고차원으로 매핑한 뒤, **그 공간에서 PCA를 수행**해서 **비선형 구조도 분리하거나 축소**할 수 있도록 만든 기법입니다.

---

# 왜 Kernel PCA가 필요할까?

- 일반 PCA는 직선적인 경계만 학습 가능  
  → 복잡한 곡선 형태의 분류나 구조는 **잘 표현 못함**
- Kernel PCA는 **곡선 형태의 경계나 분포**도 잘 표현 가능

---

# 핵심 아이디어

> 데이터를 직접 고차원으로 바꾸지 않고, **커널 함수**를 통해 **두 점 사이의 내적만 계산**해서 PCA를 수행

즉,  
고차원 맵핑 \( \phi(x) \)를 직접 계산하지 않고,  
다음과 같은 커널 함수 \( K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle \)만 사용해서 계산함.

---

# 자주 쓰이는 커널 함수

| 커널 종류     | 수식                                                    | 특징                       |
|---------------|----------------------------------------------------------|----------------------------|
| 선형          | $K(x, x') = x^T x'$                                  | 일반 PCA와 동일            |
| RBF (가우시안)| $K(x, x') = \exp(-\gamma \|x - x'\|^2)$             | 복잡한 구조 잘 표현        |
| 다항식        | $K(x, x') = (x^T x' + c)^d$                          | 곡선 경계 학습 가능        |

---

# Kernel PCA는 이렇게 동작해요

1. 커널 행렬 $K$ 계산  
   → $K_{ij} = K(x_i, x_j)$
2. 커널 행렬 중심화 (mean 제거)
3. 고유값 분해 (eigendecomposition) 수행
4. 상위 고유값의 고유벡터들 선택 → 저차원 표현 생성

---

# Python 실습 예제 (RBF 커널 사용)

```python
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 비선형 구조 데이터 생성
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# Kernel PCA 적용 (RBF 커널)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# 시각화
plt.figure(figsize=(8,6))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='coolwarm', s=30)
plt.title("Kernel PCA with RBF Kernel")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
```

> 🔍 `gamma`는 RBF 커널의 반경(폭)을 조절하는 하이퍼파라미터입니다.

---

# ✅ 장단점 요약

| 장점                           | 단점                             |
|--------------------------------|----------------------------------|
| 비선형 구조를 잘 포착함        | 새로운 데이터 투영 어려움 (비고정) |
| 다양한 커널 함수 선택 가능     | 계산량 ↑ (n x n 커널 행렬 계산)   |
| 시각화/클러스터링에 유용함    | 튜닝이 중요 (커널 종류, gamma 등) |

---

# ✅ Kernel PCA vs 일반 PCA vs t-SNE/UMAP

| 기법         | 선형성 | 비선형 구조 | 해석 가능성 | 새로운 데이터 투영 | 속도/확장성 |
|--------------|--------|-------------|--------------|--------------------|--------------|
| PCA          | ✅     | ❌          | ✅           | ✅                 | 빠름         |
| Kernel PCA   | ✅     | ✅          | ❌ (고차원)  | ❌ (재학습 필요)    | 느림         |
| t-SNE        | ❌     | ✅          | ❌           | ❌                 | 매우 느림    |
| UMAP         | ❌     | ✅          | ❌           | ✅                 | 빠름         |

---
