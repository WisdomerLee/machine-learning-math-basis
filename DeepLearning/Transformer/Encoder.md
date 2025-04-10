# 1. Transformer Encoder란?

Transformer에서 **Encoder**는 입력 문장을 받아서, 그것을 더 **의미 있는 표현(벡터)**으로 바꾸는 역할을 합니다.

하나의 Encoder는 다음과 같은 구성으로 되어 있어요:

```
[Input Embedding + Positional Encoding]
     ↓
[Multi-Head Self Attention]
     ↓
[Add & LayerNorm]
     ↓
[Feed Forward Network]
     ↓
[Add & LayerNorm]
```

> ⛳️ 이 블록을 **N번 반복**해서 깊이를 쌓습니다.

---

# 2. 각 구성 요소 자세히 보기

## ① Input Embedding + Positional Encoding
- 입력 문장을 임베딩 벡터로 바꾸고,
- 위치 정보를 추가합니다 (sin/cos 방식)

## ② Multi-Head Self-Attention
- 앞서 설명한 대로, 입력 벡터끼리 상호관계를 계산합니다

## ③ Add & LayerNorm
- 입력을 **Residual Connection**으로 더하고, **Layer Normalization**을 합니다

$$
\text{Output} = \text{LayerNorm}(x + \text{Attention}(x))
$$

## ④ Feed Forward Network (FFN)
- 각 단어 벡터에 개별적으로 2층 MLP를 적용합니다 (비선형 포함)

## ⑤ 다시 Add & LayerNorm
- FFN 결과도 Residual 연결 후 정규화합니다

---

# 3. PyTorch로 Encoder Block 구현하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + Norm
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN + Residual + Norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

- `MultiheadAttention`은 PyTorch에서 이미 제공하는 모듈을 사용했어요 (우리가 앞서 직접 구현했던 것과 동일 개념)
- `mask`는 필요한 경우, 예를 들면 패딩 토큰을 무시하거나 디코더에서는 미래 단어를 못 보게 만들 때 사용합니다.

---

# 4. 전체 Encoder 구성

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len]
        for block in self.encoder_blocks:
            x = block(x)
        return x  # 최종 인코딩된 표현
```

이 구조는 입력 문장을 벡터로 바꾼 후, 여러 개의 **Encoder Block**을 통과시켜 최종 벡터를 반환해요.

---

# 5. 간단한 사용 예

```python
vocab_size = 10000
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 6

model = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers)

# 입력 예시: (batch_size, seq_len)
x = torch.randint(0, vocab_size, (2, 20))

output = model(x)
print("Output shape:", output.shape)  # (2, 20, 512)
```

---

# 요약

| 구성 요소 | 설명 |
|------------|------|
| Embedding + PosEncoding | 단어 + 위치 벡터 |
| Multi-Head Attention | 단어 간 관계 파악 |
| Add & Norm | 안정적인 학습 |
| Feed Forward | 복합 표현 생성 |
| 반복 | 깊이 있는 추상화 가능 |

---
