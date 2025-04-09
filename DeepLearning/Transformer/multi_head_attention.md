# 왜 Multi-Head Attention이 필요할까?

기본적인 **Self-Attention**은 한 번만 관계를 계산합니다. 하지만 자연어는 복잡하잖아요?

> 예시:  
> 문장: **"나는 [너를] 좋아하지만 [너는] 나를 모른다"**  
> → '나는'은 '좋아하지만'과도 관련이 있고, '너를', '너는' 모두 관련이 있음.

하나의 Attention으로는 **다양한 의미적 관계**를 한 번에 파악하기 어려워요.

그래서!  
→ 다양한 시각(다른 방향)으로 Attention을 동시에 여러 개 계산하면 좋겠다!  
→ 그것이 **Multi-Head Attention**입니다.

---

# Multi-Head Attention 구조 요약

1. 입력 (X)을 받아서
2. 여러 개의 **Query, Key, Value**를 만들고
3. 각각에 대해 Self-Attention 수행
4. 결과들을 모아서 다시 합치고
5. 최종 선형 변환

---

# 구조 그림으로 이해해보기

```
[Input Embedding]
      ↓
┌────────────────────────────┐
│        여러 Head           │
│  ┌────────┐   ┌────────┐   │
│  │Head 1  │   │Head 2  │  ... 총 h개
│  └────────┘   └────────┘   │
└────────────────────────────┘
      ↓ (Concat)
[Linear Projection]
      ↓
[Output]
```

---

# 수식으로 표현

Multi-Head Attention 전체 수식:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

각 head는:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- $$W_i^Q, W_i^K, W_i^V$$: 각 head를 위한 가중치 행렬
- $$W^O$$: 모든 head를 합친 뒤 다시 projection하는 가중치

---

# PyTorch로 구현해보기 (단순화 버전)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0  # 나눠떨어져야 함
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projection을 하나로
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, E = x.shape  # (Batch, SeqLen, Embedding dim)
        qkv = self.qkv_proj(x)  # (B, T, 3E)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # 각 shape: (B, num_heads, T, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)  # (B, num_heads, T, head_dim)
        
        attention = attention.transpose(1, 2).reshape(B, T, E)  # 원래 shape로
        output = self.out_proj(attention)
        return output
```

이 코드는 **입력 x**에 대해 Multi-Head Self-Attention을 적용하는 간단한 버전이에요.

---

# 요약 정리

| 개념 | 설명 |
|------|------|
| Self-Attention | 각 단어가 다른 단어들과 얼마나 관련 있는지 계산 |
| Multi-Head | Self-Attention을 여러 번 → 다양한 관계 파악 가능 |
| 장점 | 다양한 의미, 문맥 파악 / 병렬처리 가능 |

---
