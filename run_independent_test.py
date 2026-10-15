# ============================================================================
# 实验三：HGDC 独立测试集（Independent Test Set）评估
#
# 目的：检验 HGDC 是否对“特定癌症基因数据集”有偏好性。
# 做法：
#   1. 用原始训练标签（796 个 CGC 驱动基因 + 2187 个非驱动基因）做 10×5CV；
#   2. 每折训练完后，对全网络节点打分；
#   3. 构造“独立测试集”——
#      正样本：独立名单（OncoKB 或 CancerMine）中、本折没有参与训练的基因；
#      负样本：2187 非驱动名单中、本折没有参与训练的基因；
#   4. 在该独立测试集上计算 AUC / AUPR，再对 10×5 折取平均。
#
# 关键：独立名单里的基因只要在本折训练集出现过就排除（排除与训练重叠）。
# ============================================================================
import argparse
import numpy as np
import torch
from sklearn import metrics
from models.HGDC import HGDCNet
from data.data_loader import load_net_specific_data
import torch.nn.functional as F

# ---------------------------- 命令行参数 ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--graph_diffusion', type=str, default='ppr', help='HGDC uses PPR diffusion.')
parser.add_argument('--ppr_alpha', type=float, default=0.9, help='Return probability in PPR.')
parser.add_argument('--ppr_eps', type=float, default=0.0001, help='Threshold to bound edges at in PPR.')
parser.add_argument('--net_avg_deg', type=int, default=50, help='Average node degree of a network.') # GGNet 111, PathNet 24, PPNet 50
parser.add_argument('--hk_t', type=int, default=1, help='Times of diffusion in Heat kernel PageRank.')
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='./data/PPNet/dataset_PPNet_ten_5CV.pkl', help='The path of the input pkl file.')
parser.add_argument('--net_file', type=str, default='./data/PPNet/PPNet.txt', help='The path of network file, corresponding to dataset_file.')
# 独立名单文件：OncoKB 或 CancerMine
parser.add_argument('--independent_set', type=str, default='OncoKB', help='Name of the independent set: OncoKB / CancerMine.')
parser.add_argument('--independent_file', type=str, default='./data/OncoKB_genes.txt', help='Path of the independent gene list.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

# 加载数据：HGDC 内部会生成 PPR 辅助图
data = load_net_specific_data(args)
data = data.to(device)

# ---------------------------- 准备独立名单与负样本 ----------------------------
# 独立正样本名单（OncoKB / CancerMine）
ind_genes = set(open(args.independent_file, encoding='utf-8').read().split())
# 全部非驱动负样本名单
neg_genes = set(open('./data/2187_nondrivers.txt', encoding='utf-8').read().split())
# 基因名 → 节点索引
node_names = list(data.node_names)
name_to_idx = {name: i for i, name in enumerate(node_names)}

# ---------------------------- 独立测试集评估 ----------------------------
@torch.no_grad()
def eval_independent(model, tr_mask):
    model.eval()
    # 前向打分
    x = model(data)
    prob = torch.sigmoid(x).view(-1).cpu().numpy()

    # 独立正样本：在独立名单里、且本折没有训练、且在网络内
    pos_idx = [name_to_idx[g] for g in ind_genes
               if g in name_to_idx and not tr_mask[name_to_idx[g]]]
    # 独立负样本：在非驱动名单里、且本折没有训练
    neg_idx = [name_to_idx[g] for g in neg_genes
               if g in name_to_idx and not tr_mask[name_to_idx[g]]]

    test_idx = np.array(pos_idx + neg_idx)
    y_true = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
    y_prob = prob[test_idx]

    auc = metrics.roc_auc_score(y_true, y_prob)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall, precision)
    return auc, aupr, len(pos_idx), len(neg_idx)


# ---------------------------- 10×5 交叉验证 ----------------------------
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        # 每个折新建 HGDC 模型
        model = HGDCNet(args).to(device)

        # 优化器：沿用官方 HGDC 参数分组（不含卷积层参数）
        optimizer = torch.optim.Adam([
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

        # 训练：HGDC 使用普通二分类交叉熵
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
            loss.backward()
            optimizer.step()

        # 在独立测试集上评估（同时返回该折独立正/负样本数）
        AUC[i][cv_run], AUPR[i][cv_run], npos, nneg = eval_independent(model, tr_mask)
        print('HGDC  %s  Round--%d CV--%d  AUC: %.5f, AUPR: %.5f  (独立正:%d, 负:%d)' % (
            args.independent_set, i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run], npos, nneg))

    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

print('=' * 60)
print('HGDC on %s 10×5CV -- Mean AUC: %.4f, Mean AUPR: %.4f' % (
    args.independent_set, AUC.mean(), AUPR.mean()))
