import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # Dropoutを追加
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, E = x.shape  # バッチサイズ、コンテキスト長、埋め込み次元
        H, D = self.n_head, self.head_dim

        # Q, K, Vの計算
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 各ヘッドに分割して並べ替え
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, C, C)
        scores = scores / (D ** 0.5)

        # マスク処理
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attentionの重み
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)  # weightsにDropoutを適用
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # ヘッドの結合と出力変換
        hidden = hidden.transpose(1,2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H*D)  # (B, C, E)
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)  # 最終結果にDropoutを適用

        return output