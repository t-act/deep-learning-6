import torch
import torch.nn as nn
import torch.nn.functional as F

B = 2   # バッチサイズ（batch_size）
C = 4   # コンテキスト長（context_len）
E = 16  # 埋め込みの次元数（embed_dim）
H = 3   # ヘッド数（n_head）
D = 8   # ヘッドの次元数（head_dim） キーとクエリの次元に相当

# 入力テンソル
x = torch.randn(B, C, E)

# 効率的な実装：全ヘッド分の重みを一つの行列にまとめる
W_q = nn.Linear(E, H*D, bias=False)
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)  # (B, C, H*D)
V = W_v(x)  # (B, C, H*D)

# 形状の変化
Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

scores = torch.matmul(Q, K.transpose(-2,-1))  # (B, H, C, C)
scores = scores / (D ** 0.5)

# マスク処理
mask = torch.tril(torch.ones(C, C, device=scores.device))
scores = scores.masked_fill(mask == 0, float("-inf"))

# Attentionの重み
weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
hidden = torch.matmul(weights, V)    # (B, H, C, D)

# 形状変換: (B, H, C, D) → (B, C, H*D)
hidden = hidden.transpose(1, 2)  # (B, C, H, D)
hidden = hidden.contiguous().view(B, C, H*D)  # (B, C, H*D)

# 出力変換: (B, C, H*D) -> (B, C, E)
W_o = nn.Linear(H*D, E, bias=False)
output = W_o(hidden)  # (B, C, E)