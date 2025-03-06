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

class LoopedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        FULL_SCALE = 3
        NOVELTY_THRESHOLD = 0.7
        self.r = NOVELTY_THRESHOLD

        self.iter_size = 4 * config.n_embd      # 假设n_embd=96 → 384
        self.full_size = FULL_SCALE * self.iter_size  # 384 * 3=1152

        # 初始化基础mask（这里需要修正full_size的计算）
        self.register_buffer('base_mask', torch.zeros(1, self.full_size))
        self.base_mask[:, :self.iter_size] = 1  # 初始激活的神经元

        # 动态权重层
        self.c_mfc = DynamicMaskedLinear(
            in_features=config.n_embd, 
            out_features=self.full_size, 
            bias=config.bias
        )
        
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(self.full_size, config.n_embd, bias=config.bias)
        
        # 控制网络（精简为更高效的版本）
        self.control_net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd//4, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.n_embd//4, self.full_size, bias=config.bias)
        )
        
        self.gate = TopKGate(k=self.iter_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        
        # 初始化mask系统（修正维度处理）
        active_mask = self.base_mask.repeat(B, 1).to(device)  # (B, full_size)
        history_mask = active_mask.clone()
        
        for _ in range(3):
            # 动态前向传播
            x = self.c_mfc(x, active_mask)  # (B, T, full_size)
            x = self.relu(x)
            x = self.c_proj(x)              # (B, T, C)
            x = self.dropout(x)
            
            # 生成新mask（添加维度修正）
            ck = self.control_net(x.mean(dim=1))  # 平均时间维度 → (B, C)
            new_mask = self.gate(ck)              # (B, full_size)
            
            # 计算新颖性指标（添加维度处理）
            combined = (history_mask + new_mask).clamp_(0, 1)
            novelty = (combined - history_mask).sum(dim=1).float().mean() / self.iter_size
            
            # 更新mask系统
            history_mask = combined
            active_mask = new_mask
            
            if novelty < self.r:
                break
                
        return x