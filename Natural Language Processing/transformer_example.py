import torch
import torch.nn as nn
import math

# ====== 1. Scaled Dot-Product Attention ======
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

# ====== 2. Multi-Head Attention ======
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_linear = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, kv=None, mask=None):
        B, T, _ = x.size()
        kv = x if kv is None else kv
        qkv = self.qkv_linear(x if kv is None else torch.cat([x, kv], dim=1))
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(x): return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))

        out, _ = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)

# ====== 3. Feed Forward ======
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

# ====== 4. Encoder Layer ======
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

# ====== 5. Decoder Layer ======
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        x = self.norm1(x + self.self_attn(x))           # Masked Self Attention (생략됨)
        x = self.norm2(x + self.cross_attn(x, enc_out)) # Encoder-Decoder Attention
        x = self.norm3(x + self.ff(x))
        return x

# ====== 6. 전체 Transformer ======
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))  # 고정 Positional Encoding
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        for layer in self.encoder_layers:
            src = layer(src)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src)

        return self.fc_out(tgt)
