import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        # Q, K, Vの変換行列
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)  # 出力変換行列
        self.key_dim = key_dim

    def forward(self, x):  # x: (B, C, E)
        Q = self.W_q(x)    # Q: (B, C, D)
        K = self.W_k(x)    # K: (B, C, D)
        V = self.W_v(x)    # V: (B, C, E)

        # Attentionマップの計算
        K_t = K.transpose(-2, -1)       # (B, D, C)
        scores = torch.matmul(Q, K_t)   # (B, C, C)
        scores = scores / (self.key_dim ** 0.5)

        # マスクの運用
        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))   # tril=triangle lower
        scores = scores.masked_fill(mask == 0, float("-inf"))       # softmax関数適用後に0にしたいから-inf

        weights = F.softmax(scores, dim=1)
        hidden = torch.matmul(weights, V)

        output = self.W_o(hidden)
        return output


attention = Attention(embed_dim=256, key_dim=64)
x = torch.randn(2, 5, 256)
y = attention(x)

print("入力形状:", x.shape)
print("出力形状:", y.shape)