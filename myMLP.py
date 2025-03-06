import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'mps'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.


class TopKGate(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, ck):
        # H shape: (batch_size, hidden_dim)
        p = torch.softmax(ck, dim=-1)
        # 获取topk的索引
        _, topk_indices = torch.topk(p, k=self.k, dim=-1)
        # 创建二进制掩码
        mask = torch.zeros_like(p)
        mask.scatter_(-1, topk_indices, 1.0)
        # 应用Straight-Through Estimator保持梯度流动
        mask = mask + (p - p.detach())
        return mask


class DynamicMaskedLinear(nn.Linear):
    def forward(self, x, mask):
        # 关键调整：为mask增加一个维度 [24576] → [24576, 1]
        masked_weight = self.weight * mask.view(-1, 1)  # 或 mask.unsqueeze(1)
        return F.linear(x, masked_weight, self.bias)


class LoopedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        FULL_SCALE = 3  # mlp 包含未激活节点的总节点，是激活节点的多少倍
        NOVELTY_THRESHOLD = 0.7
        self.r = NOVELTY_THRESHOLD

        self.iter_size = 4 * config.n_embd                # 每轮激活的 mlp 的节点个数
        self.full_size = FULL_SCALE * self.iter_size    # 包含未激活节点的 mlp 节点总个数

        # e.g. n_embd = 10
        # MLP = [ default=4 * 10 ][  hidden = 4 * 10 * 15  ]
        self.c_mfc = DynamicMaskedLinear(
            config.n_embd, self.full_size, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(
            self.full_size, config.n_embd, bias=config.bias)

        # use another MLP to check: whether should think on other hidden mlp nodes
        # update the mask to tell what MLP neuron should be active next
        self.c_fc_ck = nn.Linear(
            config.n_embd, 1 * config.n_embd, bias=config.bias)
        self.relu_ck = nn.ReLU()
        self.c_proj_ck = nn.Linear(
            1 * config.n_embd, self.full_size, bias=config.bias)
        # in next round, the ones are active, zeros are inactive
        self.gate = TopKGate(k=self.iter_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 用于统计都有哪些节点被激活过
        mask_merged = torch.zeros(self.full_size).to(device)
        mask = torch.zeros(self.full_size).to(device)
        mask[:self.iter_size] = 1    # 初始状态，固定前 iter_size 个节点激活

        for _ in range(3):
            mask_merged = mask.bool() | mask_merged.bool()   # 更新统计

            # 用 mask 处理权重, 只涉及部分神经元, 提高效率
            x = self.c_mfc(x, mask)
            x = self.relu(x)
            x = self.c_proj(x)
            x = self.dropout(x)

            # 下一轮想激活的节点
            ck = self.c_fc_ck(x)
            ck = self.relu_ck(ck)
            ck = self.c_proj_ck(ck)
            mask2 = self.gate(ck)
            novelty = ((mask2 == 1) & (mask_merged == 0)).sum()
            novelty = novelty / self.iter_size
            if novelty < self.r:  # 如果下一轮新的激活点太少，early exit，不再循环MLP
                return x
        return x
