# 1. Transformer란?

Transformer는 2017년 논문 "Attention is All You Need"에서 처음 제안된 모델로, **자연어 처리(NLP)**를 중심으로 다양한 분야에서 널리 활용되고 있어요. 기존 RNN 계열 모델들과 달리 **전체 입력 시퀀스를 한 번에 처리**할 수 있고, **병렬화가 가능**해 학습 속도가 빠릅니다.

> 핵심 개념: **Self-Attention**

---

# 2. Transformer 이해를 위한 핵심 개념

## 1. **Tokenization**
문장을 단어(또는 subword) 단위로 나누는 작업  
예: `"나는 학교에 간다"` → `["나는", "학교", "에", "간다"]`

---

## 2. **Embedding**
각 단어를 고정된 길이의 벡터로 변환  
예: `"나는"` → `벡터 [0.1, -0.3, ...]`

---

## 3. **Position Encoding**
Transformer는 순서를 모르기 때문에, 단어의 **위치 정보**를 인코딩해야 함.  
주로 **사인/코사인 함수 기반**의 방식 사용


$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

---

## 4. **Self-Attention**
각 단어가 문장 내의 다른 단어들과 얼마나 관련이 있는지를 계산

- 각 단어 → 세 가지 벡터로 변환:  
  - **Query (Q)**  
  - **Key (K)**  
  - **Value (V)**


$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

이 수식은 한 단어(Q)가 다른 단어들(K)과 얼마나 관련 있는지를 점수로 계산하고, 그 점수를 바탕으로 값을(Value) 가중합해서 반환하는 구조입니다.

---

## 5. **Multi-Head Attention**
Self-Attention을 여러 번 병렬로 수행한 후 합칩니다.  
→ 다양한 관계를 동시에 파악할 수 있음

---

## 6. **Feed Forward Network (FFN)**
각 위치별로 동일한 두 개의 선형 변환을 적용하는 층 (비선형성 포함)

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

---

# 3. Transformer의 구조 요약

```
[Input Tokens] 
     ↓
[Embedding + Positional Encoding]
     ↓
[Multi-Head Self Attention]
     ↓
[Feed Forward Network]
     ↓
[Output (예: 번역된 문장)]
```

---

# 4. 파이썬 코드 예시 (Self-Attention 단순 구현)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output

# 임의의 값으로 Q, K, V 정의
Q = torch.randn(1, 3, 4)  # (batch_size, seq_len, d_k)
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)

output = scaled_dot_product_attention(Q, K, V)
print("Attention Output:\n", output)
```

이 코드는 간단한 Self-Attention 연산을 보여주는 예시예요.

---

# 5. 공부를 위한 추천 순서

1. **기본 개념 익히기**
   - Token, Embedding, Attention의 의미와 흐름
2. **수식 이해하기**
   - Attention 수식, Position Encoding 수식
3. **PyTorch로 간단한 구현**
   - Self-Attention → Multi-Head Attention → 전체 Transformer
4. **실제 모델 사용해보기**
   - HuggingFace Transformers 라이브러리로 BERT, GPT 등 실습

---
