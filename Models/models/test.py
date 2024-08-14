import torch
support = torch.arange(0, 25).view(5,5)

# 将这个张量重塑为 (5, 5)，这里需要调整值使其符合 1-5 的重复
# support = support % 5 + 1  # 使其值在 1 到 5 之间循环
print(support.view(5,5))
# 重塑张量为 (5, 5)，以形成所需的矩阵
# support = torch.cat([support[: i] for i in range(5)], dim=0)
support = support.permute(1,0)
print(support)
