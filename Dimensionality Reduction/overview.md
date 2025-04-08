# 차원 축소란?
> **고차원 데이터를 더 낮은 차원의 공간으로 변환**하는 것.  
> 예: 1000개의 특성을 가진 데이터를 2~50개 정도의 특성으로 줄임.

---

# 왜 차원 축소가 필요할까?
1. **계산 효율 향상**: 입력 차원이 줄어들면 연산량 ↓
2. **노이즈 제거**: 불필요한 정보 제거 → 더 좋은 일반화 성능
3. **시각화 가능**: 2D 또는 3D로 변환해서 데이터 구조 이해
4. **차원의 저주(Curse of Dimensionality) 완화**  
   → 차원이 커질수록 학습이 어려워지는 문제를 해결

---

# 차원 축소 방법 2가지 분류
1. **선형(LINEAR)**: 데이터가 선형적으로 분포한다고 가정
2. **비선형(NONLINEAR)**: 복잡한 구조의 데이터도 다룰 수 있음

---

# 대표적인 차원 축소 기법

## 1. 📘 PCA (주성분 분석, Principal Component Analysis)
- **가장 널리 쓰이는 선형 차원 축소 기법**
- 데이터의 **분산이 가장 큰 방향**을 기준으로 새 축(주성분)을 잡고, 그 축에 투영해서 차원을 줄임
- 💡 핵심: 최대한 정보를 보존하는 방향으로 축을 회전시키는 것

👉 사용 예시: 이미지 압축, 사전 데이터 전처리

---

## 2. 📕 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- 비선형 기법, **고차원 데이터를 2D/3D로 시각화**할 때 탁월
- 고차원에서 가까운 점들이 낮은 차원에서도 가깝도록 유지
- 단점: 계산 비용 큼, 새로운 샘플 예측 어려움

👉 사용 예시: 이미지, 텍스트 임베딩 결과 시각화

---

## 3. 📗 UMAP (Uniform Manifold Approximation and Projection)
- t-SNE보다 빠르고 일반화가 쉬운 비선형 방법
- 군집 구조 보존이 잘 됨
- 💡 최근엔 t-SNE보다 더 많이 쓰이는 추세

👉 사용 예시: 클러스터링 전 전처리, 시각화

---

## 4. 📙 Autoencoder (오토인코더)
- 신경망 기반 차원 축소
- 인코더(압축) → 디코더(복원) 구조
- **잠재 공간(latent space)**을 사용해 데이터를 축소
- 학습 필요, 비선형 표현 가능

👉 사용 예시: 이상 탐지, 이미지 재구성, 텍스트 임베딩

---

# 딥러닝에서의 활용 예
| 적용 분야        | 차원 축소 역할                      |
|------------------|-------------------------------------|
| 이미지 처리      | 고해상도 이미지 → 낮은 차원 압축    |
| 텍스트 임베딩    | BERT 출력 → 2D 시각화               |
| 이상 탐지        | 오토인코더 latent space 사용         |
| 클러스터링       | UMAP 후 K-means 등 군집화 적용       |

---

# ✅ 이해를 위한 기본 개념 정리
- **고차원**: 특성(feature) 수가 많은 데이터
- **분산(Variance)**: 데이터의 퍼짐 정도
- **선형변환**: 행렬 곱 등으로 데이터를 회전/변형
- **비선형 변환**: 곡선처럼 구부러진 형태로 데이터 매핑
- **잠재 공간(latent space)**: 중요한 정보만 담긴 축소된 표현 공간

---

# 공통 준비 코드

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 데이터 불러오기
digits = load_digits()
X = digits.data  # (1797, 64) 이미지 데이터
y = digits.target  # (1797,) 라벨

def plot_2d(X_2d, y, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=15)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title(title)
    plt.show()
```

---

# 1. 📘 PCA (선형)

```python
from sklearn.decomposition import PCA

# PCA 적용 (2차원)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 시각화
plot_2d(X_pca, y, "PCA (2D)")
```

---

# 2. 📕 t-SNE (비선형)

```python
from sklearn.manifold import TSNE

# t-SNE 적용 (2차원)
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)

# 시각화
plot_2d(X_tsne, y, "t-SNE (2D)")
```

---

# 3. 📗 UMAP (비선형)

```python
import umap.umap_ as umap

# UMAP 적용 (2차원)
umap_model = umap.UMAP(n_components=2, random_state=0)
X_umap = umap_model.fit_transform(X)

# 시각화
plot_2d(X_umap, y, "UMAP (2D)")
```

> 💡 `pip install umap-learn` 필요

---

# 4. 📙 Autoencoder (신경망 기반)

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# 데이터 정규화
X_scaled = MinMaxScaler().fit_transform(X)

# 오토인코더 구성
input_dim = X.shape[1]
encoding_dim = 2  # 2차원 latent space

input_img = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='linear')(encoded)  # 잠재공간
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# 학습
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=64, verbose=0)

# 인코딩 결과 추출
X_ae = encoder.predict(X_scaled)

# 시각화
plot_2d(X_ae, y, "Autoencoder (2D)")
```

---

# 🔍 요약

| 기법       | 라이브러리              | 장점                 | 단점                   |
|------------|--------------------------|----------------------|------------------------|
| PCA        | `sklearn`                | 빠르고 직관적        | 선형성 한계            |
| t-SNE      | `sklearn`                | 시각화에 적합        | 느리고 일반화 어려움   |
| UMAP       | `umap-learn`             | 빠르고 군집 보존     | 초매개변수 영향 큼      |
| Autoencoder| `tensorflow/keras`       | 비선형 복잡 패턴 학습| 학습 필요, 튜닝 필요    |

---
