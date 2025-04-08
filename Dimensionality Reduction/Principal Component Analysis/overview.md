# PCA란?

> 데이터를 구성하는 변수들의 **분산이 가장 큰 방향(주성분)을 찾아서**, 그 방향을 기준으로 데이터를 투영(projection)하여 **차원을 축소**하는 방법

---

# 왜 "분산이 큰 방향"을 찾을까?

- 분산이 크다는 것은 데이터의 **정보가 더 많이 퍼져 있다**는 뜻
- 반대로 분산이 작은 방향은 **노이즈일 가능성**이 높음
- 따라서, **정보를 최대한 보존하면서 차원을 줄이는 것**이 목표!

---

# 직관적으로 이해하기 (2D → 1D 예시)

1. 2차원 평면에 점들이 퍼져 있다고 해보세요.
2. 이 점들을 **분산이 가장 큰 축**(대각선 등)으로 회전시킵니다.
3. 그 축을 **새로운 축 (주성분, PC: Principal Component)**이라 부릅니다.
4. 이제 모든 점을 그 축 위에 "그림자처럼" 투영시키면 → 1차원 데이터가 됩니다.

---

# PCA의 주요 단계

1. **데이터 정규화** (평균 0, 분산 1)
2. **공분산 행렬 계산**
   - 변수들 간의 관계(분산, 상관성)를 담고 있음
3. **고유값 분해 (Eigen Decomposition)**
   - 공분산 행렬을 고유벡터(Eigenvector)와 고유값(Eigenvalue)로 분해
4. **가장 큰 고유값을 가진 고유벡터 선택**
   - 고유값이 클수록 분산이 크다는 의미
   - 상위 N개 고유벡터를 선택 → **주성분(PC)**가 됨
5. **원본 데이터를 주성분에 투영**
   - 차원이 줄어든 데이터 생성

---

# PCA 수식 간단 요약

- 공분산 행렬:  
  
  $$
  \Sigma = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X})
  $$
  
- 고유값 분해:  
  $$
  \Sigma v = \lambda v
  $$
- 주성분 선택: 고유값 λ가 가장 큰 v들 선택

---

# 시각적 이해 (코드 포함)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 데이터
data = load_iris()
X = data.data
y = data.target

# 표준화
X_std = StandardScaler().fit_transform(X)

# PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# 시각화
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', s=40)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA result (2D)')
plt.colorbar(label='class')
plt.show()
```

---

# 주성분의 해석

- `pca.explained_variance_ratio_` : 각 주성분이 **설명하는 분산의 비율**
- 예:
  ```python
  print(pca.explained_variance_ratio_)
  ```
  출력: `[0.7296, 0.2285]` → 첫 두 개 주성분이 전체 정보의 약 95.8%를 설명함

---

# PCA의 단점

| 한계 | 설명 |
|------|------|
| 선형만 가능 | 비선형 관계를 설명할 수 없음 |
| 해석의 어려움 | 축이 회전되기 때문에 원래 변수 의미와 해석이 어려움 |
| 데이터 스케일 민감 | 스케일링 전처리는 필수 |

---

# PCA는 언제 사용할까?

- 데이터 시각화 (특히 2D, 3D)
- 특성 선택 전 전처리
- 노이즈 제거
- 모델 훈련 속도 향상
- 고차원 이미지, 텍스트, 센서 데이터 등

---

