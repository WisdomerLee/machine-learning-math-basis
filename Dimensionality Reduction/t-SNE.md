# t-SNE 한 줄 요약

> 고차원 공간에서 **가까운 점은 낮은 차원에서도 가깝게**,  
> 먼 점은 낮은 차원에서도 멀게 유지하면서 **2D/3D로 시각화**하는 차원 축소 기법

---

# t-SNE는 어디에 쓰이나요?

| 활용 분야               | 설명                                         |
|------------------------|----------------------------------------------|
| 딥러닝 임베딩 시각화    | BERT, CNN 등에서 출력되는 벡터를 시각화       |
| 클러스터 구조 탐색      | 군집의 형태가 있는지 확인                    |
| 이상치 탐지             | 다른 군집과 동떨어진 포인트 식별             |
| 이미지/텍스트 분류 확인 | 클래스별로 잘 구분되는지 확인                |

---

# t-SNE 작동 원리 (직관적 설명)

1. **고차원 공간에서의 이웃 정보 계산**  
   - 각 데이터 포인트 주변에 있는 점들의 상대적 거리 정보를 **확률 분포로 표현**

2. **저차원 공간(예: 2D)에서도 확률 분포 생성**  
   - 두 점이 얼마나 가까운지를 **t-분포 기반**으로 표현  
   - (여기서 “t”는 t-distribution에서 온 것)

3. **두 확률 분포(P, Q)를 비슷하게 만들기 위해 비용 함수(KL Divergence)를 최소화**

> 💡 요약하면:  
> “**이웃 정보 보존**”을 목표로, 고차원에서 가깝던 점들이 **2D에서도 가깝게** 유지되도록 배치함.

---

# 핵심 파라미터

| 파라미터       | 설명 |
|----------------|------|
| `perplexity`   | 주변 이웃의 수. 일반적으로 5~50 사이 사용 |
| `n_iter`       | 학습 반복 횟수 (기본 1000 이상 권장) |
| `learning_rate`| 학습률. 10~1000 사이가 적절 |
| `metric`       | 거리 측정 방식 (기본은 유클리디안) |

---

# Python 예제

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 데이터 로드 (8x8 손글씨 이미지)
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE 적용
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
X_tsne = tsne.fit_transform(X)

# 시각화
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15)
plt.title("t-SNE (2D)")
plt.legend(*scatter.legend_elements(), title="Digits")
plt.grid(True)
plt.show()
```

---

# t-SNE의 특징

## 장점

✅ 비선형 구조 시각화 탁월  
✅ 군집 구조 파악 쉬움  
✅ 딥러닝 결과 해석 시 자주 사용됨

## 단점

❌ **새로운 데이터** 투영 불가능 (지도학습 불가)  
❌ **계산 비용** 큼 (n² 복잡도, 느림)  
❌ **하이퍼파라미터**에 민감  
❌ **거리의 절대값 의미 없음** (가깝다/멀다만 중요)

---

# t-SNE과 다른 기법 비교

| 기법       | 비선형 구조 | 지도학습 | 해석 용이성 | 계산 속도 | 새 데이터 처리 |
|------------|-------------|-----------|--------------|------------|----------------|
| **t-SNE**  | ✅           | ❌        | ❌           | 느림       | ❌             |
| **UMAP**   | ✅           | ✅ (부분) | ❌           | 빠름       | ✅             |
| **PCA**    | ❌           | ✅        | ✅           | 빠름       | ✅             |

---

# 실전 팁

- `perplexity`는 데이터 수의 1~5% 정도로 설정
- `n_iter`를 1000 이상 주면 결과 안정화됨
- 여러 번 실행하면 결과가 달라질 수 있음 → `random_state` 고정

