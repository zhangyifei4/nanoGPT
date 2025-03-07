import torch
import torch.nn as nn
from torch.nn import functional as F

class TopKGate(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, ck):
        # 输入形状：(B, T, full_size)
        probs = torch.softmax(ck, dim=-1)
        _, topk = torch.topk(probs, self.k, dim=-1, sorted=False)
        return torch.zeros_like(probs).scatter_(-1, topk, 1.0)

class DynamicMaskedLinear(nn.Linear):
    def forward(self, x, mask):
        """
        x: (B, T, in_features)
        mask: (B, out_features)
        输出: (B, T, out_features)
        """
        # 将权重调整为 (1, out_features, in_features)
        weight = self.weight.unsqueeze(0) 
        # 将mask调整为 (B, out_features, 1)
        mask = mask.unsqueeze(-1) 
        # 应用mask并执行批量矩阵乘法
        return torch.bmm(x, (weight * mask).transpose(1,2)) # + self.bias

class MaskedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        FULL_SCALE = 4

        self.iter_size = 4 * config.n_embd      # 假设n_embd=96 → 384
        self.full_size = FULL_SCALE * self.iter_size  # 384 * 3=1152

        # 控制网络，决定哪些神经元被调用
        self.control_net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd//16, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.n_embd//16, self.full_size, bias=config.bias),
            TopKGate(k=self.iter_size)
        )
        # 动态权重层的 MLP
        self.c_mfc = DynamicMaskedLinear(
            in_features=config.n_embd, 
            out_features=self.full_size, 
            bias=config.bias
        )
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(self.full_size, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 生成新mask（添加维度修正）
        mask = self.control_net(x.mean(dim=1))  # 平均时间维度 → (B, C)

        # 动态前向传播
        x = self.c_mfc(x, mask)  # (B, T, full_size)
        x = self.relu(x)
        x = self.c_proj(x)              # (B, T, C)
        x = self.dropout(x)

        return x