# -*- coding: utf-8 -*-
"""
HGDC 消融实验模型（对应论文 Table 3 与 PPT 实验四）

三个消融模型，每次只改动一个组件，其余与完整 HGDC 保持一致，从而做公平对比：

1. AblationPPRNet（HGDC-PPR）：
   去掉图扩散辅助网络，只在原始生物分子网络上做消息聚合与传播。
   对应论文"移除公式 (4) 中的 H_PPR^(l+1)"。单分支网络，各层表示均为 H 维。

2. AblationSeparationNet（HGDC-Separation）：
   不再把"自我表征"和"邻居表征"分离开，而是在卷积时加自环（A+I），
   让节点把自身和邻居一起聚合。对应论文公式
   H^(l+1) = σ(D^-1/2(A+I)D^-1/2 H^l W^l) || σ(D_ppr^-1/2(A_ppr+I)D_ppr^-1/2 H_PPR^l W_ppr^l)。
   即把完整 HGDC 里所有 GCNConv 的 add_self_loops 由 False 改为 True。

3. AblationAttentionNet（HGDC-Attention）：
   取消分层注意力，把不同 GCN 层的中间表示 R0~R3 直接拼接后，喂给一个
   三层 MLP 输出打分。

维度记号：N 节点数，F 输入特征维，H 隐层维（默认 100）。
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


# ============================================================================
# 消融一：HGDC-PPR（去辅助网络，只在原始网络传播）
# ============================================================================
class AblationPPRNet(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels

        # 输入线性层（F -> H）
        self.linear1 = Linear(in_channels, hidden_channels)

        # 原始网络上的 3 层 GCNConv（单分支，输入输出均为 H，不加自环）
        self.conv_k1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

        # 4 个分层分类头（R0~R3 均为 H 维，H -> 1）
        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(hidden_channels, 1)
        self.linear_r2 = Linear(hidden_channels, 1)
        self.linear_r3 = Linear(hidden_channels, 1)

        # 分层注意力权重（可学习）
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index = data.edge_index

        # 边 dropout（仅训练时）
        edge_index, _ = dropout_adj(edge_index, p=0.5, force_undirected=True,
                                    num_nodes=x_input.shape[0], training=self.training)
        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))
        R1 = self.conv_k1(R0, edge_index)   # 单分支，H 维
        R2 = self.conv_k2(R1, edge_index)
        R3 = self.conv_k3(R2, edge_index)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out


# ============================================================================
# 消融二：HGDC-Separation（自我+邻居聚合，卷积加自环）
# ============================================================================
class AblationSeparationNet(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels

        self.linear1 = Linear(in_channels, hidden_channels)

        # 原始网络 3 层卷积（add_self_loops=True，把自身与邻居一起聚合）
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=True)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=True)
        # 辅助网络 3 层卷积（同样加自环）
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=True)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=True)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2 * hidden_channels, 1)
        self.linear_r2 = Linear(2 * hidden_channels, 1)
        self.linear_r3 = Linear(2 * hidden_channels, 1)

        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_adj(edge_index_1, p=0.5, force_undirected=True,
                                      num_nodes=x_input.shape[0], training=self.training)
        edge_index_2, _ = dropout_adj(edge_index_2, p=0.5, force_undirected=True,
                                      num_nodes=x_input.shape[0], training=self.training)
        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out


# ============================================================================
# 消融三：HGDC-Attention（去分层注意力，拼接各层后喂三层 MLP）
# ============================================================================
class AblationAttentionNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels

        self.linear1 = Linear(in_channels, hidden_channels)

        # 原始网络 3 层卷积（不加自环，与完整 HGDC 一致）
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        # 辅助网络 3 层卷积
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        # 三层 MLP 分类器。
        # 拼接 R0(H) + R1(2H) + R2(2H) + R3(2H) = 7H 作为输入。
        self.mlp1 = Linear(7 * hidden_channels, 3 * hidden_channels)
        self.mlp2 = Linear(3 * hidden_channels, hidden_channels)
        self.mlp3 = Linear(hidden_channels, 1)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_adj(edge_index_1, p=0.5, force_undirected=True,
                                      num_nodes=x_input.shape[0], training=self.training)
        edge_index_2, _ = dropout_adj(edge_index_2, p=0.5, force_undirected=True,
                                      num_nodes=x_input.shape[0], training=self.training)
        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        # 拼接所有层的表示，送入三层 MLP
        feat = torch.cat((R0, R1, R2, R3), 1)
        feat = F.dropout(feat, p=0.5, training=self.training)
        feat = torch.relu(self.mlp1(feat))
        feat = F.dropout(feat, p=0.5, training=self.training)
        feat = torch.relu(self.mlp2(feat))
        out = self.mlp3(feat)
        return out
