# -*- coding: utf-8 -*-
"""
HGDC 消融实验的 5 折交叉验证入口（10 轮 × 5 折），对应论文 Table 3。

通过 --ablation 选择要跑的消融版本：
  ppr         : HGDC-PPR（去辅助网络，只在原始网络传播）
  separation  : HGDC-Separation（卷积加自环，自我+邻居聚合）
  attention   : HGDC-Attention（去分层注意力，拼接各层后喂三层 MLP）

说明：
  - 消融一（ppr）不需要辅助图，加载数据时内部自动把 graph_diffusion 置为 none；
  - 消融二、三需要双分支，内部自动使用 ppr 生成辅助图。
最终输出 10 轮 × 5 折的平均 AUC、AUPR。
"""

import argparse
import copy
import numpy as np
import torch
from sklearn import metrics
from data.data_loader import load_net_specific_data
import torch.nn.functional as F
from models.ablation import AblationPPRNet, AblationSeparationNet, AblationAttentionNet

parser = argparse.ArgumentParser()
parser.add_argument('--graph_diffusion', type=str, default='ppr', help='由消融类型自动决定，一般无需手动指定')
parser.add_argument('--ppr_alpha', type=float, default=0.9, help='Return probability in PPR.')
parser.add_argument('--ppr_eps', type=float, default=0.0001, help='Threshold to bound edges at in PPR.')
parser.add_argument('--net_avg_deg', type=int, default=50, help='Average node degree of a network.')  # GGNet 111, PathNet 24, PPNet 50
parser.add_argument('--hk_t', type=int, default=1, help='Times of diffusion in Heat kernel PageRank.')
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='./data/PPNet/dataset_PPNet_ten_5CV.pkl', help='The path of the input pkl file.')
parser.add_argument('--net_file', type=str, default='./data/PPNet/PPNet.txt', help='The path of network file.')
parser.add_argument('--ablation', type=str, default='ppr', help='Ablation name: ppr, separation, attention.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()

# 消融模型类映射
model_dict = {
    'ppr': AblationPPRNet,
    'separation': AblationSeparationNet,
    'attention': AblationAttentionNet,
}

# 根据消融类型决定是否生成辅助图
data_args = copy.deepcopy(args)
if args.ablation == 'ppr':
    data_args.graph_diffusion = 'none'      # 消融一：不用辅助网络
else:
    data_args.graph_diffusion = 'ppr'       # 消融二、三：需要辅助网络

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

data = load_net_specific_data(data_args)
data = data.to(device)

@torch.no_grad()
def test(data, mask):
    """在测试集上计算 AUC、AUPR。"""
    model.eval()
    x = model(data)
    pred = torch.sigmoid(x[mask])
    precision, recall, _thresholds = metrics.precision_recall_curve(
        data.y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area


def build_optimizer(model, ablation):
    """按消融模型结构构建优化器（与完整 HGDC 一致：线性层带 weight_decay，注意力用 lr*0.1）。"""
    if ablation in ('ppr', 'separation'):
        return torch.optim.Adam([
            dict(params=model.linear1.parameters(), weight_decay=args.w_decay),
            dict(params=model.linear_r0.parameters(), weight_decay=args.w_decay),
            dict(params=model.linear_r1.parameters(), weight_decay=args.w_decay),
            dict(params=model.linear_r2.parameters(), weight_decay=args.w_decay),
            dict(params=model.linear_r3.parameters(), weight_decay=args.w_decay),
            dict(params=model.weight_r0, lr=args.lr * 0.1),
            dict(params=model.weight_r1, lr=args.lr * 0.1),
            dict(params=model.weight_r2, lr=args.lr * 0.1),
            dict(params=model.weight_r3, lr=args.lr * 0.1)
        ], lr=args.lr)
    else:
        # attention 消融：训练输入层与三层 MLP
        return torch.optim.Adam([
            dict(params=model.linear1.parameters(), weight_decay=args.w_decay),
            dict(params=model.mlp1.parameters(), weight_decay=args.w_decay),
            dict(params=model.mlp2.parameters(), weight_decay=args.w_decay),
            dict(params=model.mlp3.parameters(), weight_decay=args.w_decay),
        ], lr=args.lr)


# 10 轮 × 5 折结果容器
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        model = model_dict[args.ablation](args).to(device)
        optimizer = build_optimizer(model, args.ablation)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Training epoch: {epoch:03d}')

        AUC[i][cv_run], AUPR[i][cv_run] = test(data, te_mask)
        print('[%s] Round--%d CV--%d  AUC: %.5f, AUPR: %.5f'
              % (args.ablation, i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))

    print('[%s] Round--%d Mean AUC: %.5f, Mean AUPR: %.5f'
          % (args.ablation, i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

print('HGDC-%s 10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f'
      % (args.ablation, AUC.mean(), AUPR.mean()))
