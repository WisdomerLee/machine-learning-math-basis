# **자연어 처리 대표 알고리즘 요약**

## 딥러닝 **이전**의 주요 알고리즘
| 알고리즘 | 핵심 아이디어 | 주 용도 |
|----------|----------------|---------|
| **Rule-based** | 사람이 직접 규칙 작성 (예: 정규표현식, 사전 기반) | 간단한 텍스트 필터링, 정보 추출 |
| **TF-IDF** | 단어의 중요도를 수치화 (빈도 기반) | 문서 분류, 유사도 측정 |
| **Naive Bayes** | 확률 기반의 분류 모델 (조건부 독립 가정) | 감정 분석, 스팸 필터링 |
| **SVM (Support Vector Machine)** | 고차원 벡터 공간에서 분류 경계 학습 | 텍스트 분류 |
| **HMM (Hidden Markov Model)** | 순차 데이터에서 상태 전이와 관측 확률 이용 | 품사 태깅, 개체명 인식 |
| **CRF (Conditional Random Field)** | 문맥을 고려한 순차 라벨링 | 개체명 인식, 품사 태깅 |

---

## 딥러닝 **이후**의 주요 알고리즘
| 알고리즘 | 핵심 아이디어 | 주 용도 |
|----------|----------------|---------|
| **Word2Vec / GloVe** | 단어를 의미 있는 벡터로 임베딩 | 단어 유사도, 기초 임베딩 |
| **RNN (Recurrent Neural Network)** | 순차적 정보 처리 (메모리 존재) | 시퀀스 분류, 번역 |
| **LSTM / GRU** | RNN의 장기 기억 문제 해결 | 감정 분석, 번역, QA |
| **Transformer** | 모든 단어 간 관계를 병렬로 처리 (Self-Attention) | 문맥 이해, 고속 학습 |
| **BERT** | 양방향 문맥 임베딩 (Masked Language Model) | 분류, 질의응답, 요약 |
| **GPT** | 단방향 예측 기반 언어 생성 | 텍스트 생성, 요약, 대화형 AI |
| **T5 / BART** | 입력-출력을 텍스트로 통일한 범용 모델 | 번역, 요약, QA 등 다양한 태스크 |

---

# 한눈에 보는 흐름

```plaintext
Rule-based → Naive Bayes / SVM / HMM / CRF
          ↓
     Word2Vec / GloVe (임베딩 도입)
          ↓
       RNN → LSTM / GRU
          ↓
     Transformer → BERT / GPT / T5 / BART
```

좋은 질문이에요! 딥러닝 이후 가장 널리 사용되는 자연어 처리 모델 중 **BERT**는 구조와 활용 측면에서 매우 대표적입니다. 아래에 구조 개념과 코드 예제를 함께 정리해드릴게요.

---

# 대표 딥러닝 NLP 알고리즘: **BERT (Bidirectional Encoder Representations from Transformers)**

## 1. BERT의 핵심 구조

BERT는 **Transformer의 Encoder 부분만** 사용하고, 문장의 **양방향 문맥**을 이해할 수 있도록 학습된 모델입니다.

### 주요 특징
- **Bidirectional**: 단어 앞/뒤의 문맥을 모두 고려
- **Pretraining 방식**:  
  - MLM (Masked Language Modeling): 문장 중 일부 단어를 가리고 예측  
  - NSP (Next Sentence Prediction): 두 문장이 연결되는 문장인지 예측
- **Fine-tuning 가능**: 분류, 질문응답, 요약 등 다양한 NLP 태스크에 적용 가능

---

## 2. BERT 구조 요약 그림

```
[CLS]  나는  오늘  기분이  [MASK]  .  [SEP]
 ↓       ↓      ↓      ↓       ↓       ↓
Embedding + Positional Encoding
 ↓
Multi-head Self-Attention (x N)
 ↓
Feedforward Layer
 ↓
[Final Hidden State for Classification]
```

- `[CLS]`: 문장 분류 등에 사용하는 특별 토큰  
- `[MASK]`: MLM에서 가려진 토큰  
- `Multi-head Self-Attention`: 문맥 이해 핵심  
- Encoder Stack은 보통 12개 층 (BERT-base 기준)

---

## 3. BERT 실습 예제 코드 (🤗 Hugging Face 활용)

아래는 **문장 분류(Sentiment Analysis)** 예제입니다.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# 사전 학습된 BERT 모델 로드 (감정 분류용)
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 예측 실행
result = classifier("이 영화는 정말 재미있었어요!")
print(result)
# 출력 예: [{'label': '5 stars', 'score': 0.85}]
```

---

## 4. 커스텀 태스크에 Fine-tuning 하고 싶다면?
```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 데이터셋 로딩 (예: 감정 분석용 IMDb)
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 전처리 함수
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 모델 준비
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=2,
)

# Trainer로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=encoded_dataset["test"].select(range(500)),
)

trainer.train()
```

---

## 요약

| 요소 | 설명 |
|------|------|
| 구조 | Transformer Encoder 기반, 양방향 문맥 이해 |
| 특징 | MLM + NSP 사전학습, 다양한 태스크에 Fine-tuning 가능 |
| 사용 | 감정 분석, 문장 분류, QA, 요약, 개체명 인식 등 |

---

# Transformer 구조: 핵심 요약

## 핵심 개념
- **Self-Attention** 메커니즘을 통해 문장 내 단어들 간의 관계를 병렬로 처리  
- **Encoder-Decoder 구조** (기계 번역 등 시퀀스-투-시퀀스 태스크에 사용)

---

# Transformer 구조도 (전체)

```plaintext
Input Embedding + Positional Encoding
↓
[Encoder Layer] × N
  - Multi-Head Attention
  - Feed Forward Network
↓
Encoded Representation
↓
[Decoder Layer] × N
  - Masked Multi-Head Attention
  - Multi-Head Attention (인코더 출력 참조)
  - Feed Forward Network
↓
Linear + Softmax
↓
Output
```

---

## 핵심 알고리즘: Self-Attention

```plaintext
Q = Query, K = Key, V = Value

Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

- 각 단어 벡터가 다른 단어와의 관련도를 계산해 가중합
- 병렬 계산이 가능하다는 점에서 RNN보다 효율적

---

## Transformer 간단 구현 (PyTorch 기반)

다음은 **Encoder 한 층 수준**의 핵심 구현 예제입니다. (직접 Self-Attention 포함)

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        B, T, _ = x.size()

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.fc(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.attn(x)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x
```

---

## 요약: 핵심 컴포넌트 정리

| 구성 요소 | 설명 |
|-----------|------|
| **Multi-Head Attention** | 단어 간 관계를 다양한 시각으로 병렬 계산 |
| **Feed Forward Layer** | 각 단어 위치별 독립적인 변환 |
| **LayerNorm + Residual** | 안정적 학습을 위한 핵심 기법 |
| **Positional Encoding** | 위치 정보를 임베딩에 추가 |

---

## 추가 팁
- 직접 구현보다는 실무에서는 Hugging Face의 `transformers` 라이브러리를 통해 간단하게 사용 가능
- 예: `AutoModel.from_pretrained("bert-base-uncased")`

---

# 1. Transformer Encoder 구조

## 핵심 구성 요소
1. **Input Embedding + Positional Encoding**
2. **Multi-Head Self-Attention**  
   → 입력 문장 내부 단어들 간 관계를 병렬 계산
3. **Feed-Forward Layer**
4. **Residual Connection + LayerNorm**

## Encoder 코드 (한 층 예시)
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)   # Residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)     # Residual
        return x
```

---

# 2. Transformer Decoder 구조

## 핵심 구성 요소
1. **Input Embedding + Positional Encoding**
2. **Masked Multi-Head Self-Attention**  
   → 앞 단어까지만 참고해 다음 단어 예측
3. **Encoder-Decoder Attention**  
   → Encoder 출력과의 상호작용
4. **Feed-Forward Layer**
5. **Residual + LayerNorm**

## Decoder 코드 (한 층 예시)
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        x2 = self.self_attn(x)  # Masked Attention (추가 구현 필요)
        x = self.norm1(x + x2)

        x2 = self.enc_dec_attn(x, enc_output, enc_output)
        x = self.norm2(x + x2)

        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x
```

> 🔎 **Note**: `Masked Attention`은 future token을 보지 않도록 마스킹을 적용해야 합니다. (예: causal mask)

---

# Encoder vs Decoder 핵심 비교표

| 요소 | Encoder | Decoder |
|------|---------|---------|
| 입력 | 전체 입력 문장 | 출력 토큰 시퀀스 |
| Attention | Self-Attention (모든 단어 참조 가능) | **Masked** Self-Attention (앞만 참조) |
| Encoder-Decoder Attention | 없음 |  있음 (Encoder 출력을 참조) |
| 용도 | 문장 인코딩, 분류, BERT 등 | 생성 (번역, 요약), GPT 등 |
| 대표 모델 | BERT, RoBERTa | GPT (Decoder-only) |
| 둘 다 사용하는 모델 | Transformer (번역), T5, BART |

---

## 실제 활용 예
- **BERT**: Encoder만 사용 → 문장 분류, QA
- **GPT**: Decoder만 사용 → 텍스트 생성
- **Transformer (원조)**: Encoder-Decoder 사용 → 번역, 요약
- **T5 / BART**: Pretrain-Encoder-Decoder 모델

---

## 마무리 정리

- **Encoder**: 입력 문장을 이해하는 데 집중  
- **Decoder**: 새로운 문장을 생성하는 데 집중  
- **Encoder-Decoder Attention**: 인코딩된 입력을 바탕으로 출력 생성에 도움

---
