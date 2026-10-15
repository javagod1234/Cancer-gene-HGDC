# -*- coding: utf-8 -*-
# ============================================================================
# 实验五、六：分析全网络基因平均分数
#
# 实验五（论文 Figure 2A）：HGDC 对 4 类基因的平均预测分数
#   类别（互斥）：
#     CDG       —— 已知驱动基因（796 CGC）
#     Potential —— 候选癌症基因（默认用 OncoKB 中、不在 CGC 的基因；
#                  严格对齐论文可换成 NCG 6.0 名单）
#     Nondriver —— 非驱动基因（2187）
#     Other     —— 其余所有基因
#
# 实验六（论文 Figure 2B）：三个网络各自的 top100 预测基因做 Venn 图，
#   统计重叠数量，并计算每个重叠区中已知驱动基因（CDG）的占比。
#
# 输入：predicted_scores/predicted_scores_<网络>.txt（由 run_exp56.sh 生成）
# 输出：图（PNG）、统计表（CSV）、重叠基因名单（TXT）
# ============================================================================
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------- 命令行参数 ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--score_dir', type=str, default='./predicted_scores', help='目录：predicted_scores_<网络>.txt')
parser.add_argument('--cgc_file', type=str, default='./data/796_drivers.txt', help='已知驱动基因名单（CDG）')
parser.add_argument('--nondriver_file', type=str, default='./data/2187_nondrivers.txt', help='非驱动基因名单')
parser.add_argument('--potential_file', type=str, default='./data/NCG6_candidate_genes.txt', help='候选基因名单。严格复现用 NCG 6.0 的 1661 个 candidate（默认）')
parser.add_argument('--use_prob', action='store_true', help='是否把 logits 转成 sigmoid 概率（论文称 score 为概率；默认用 logits，与官方名单一致）')
parser.add_argument('--topk', type=int, default=100, help='每个网络取前 top-k 个预测基因')
parser.add_argument('--out_dir', type=str, default='./experiment_results/exp56', help='结果输出目录')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

NETS = ['GGNet', 'PathNet', 'PPNet']

# 读名单（统一大写，便于匹配）
def read_set(path):
    return set(g.strip().upper() for g in open(path, encoding='utf-8').read().split() if g.strip())

cgc = read_set(args.cgc_file)
nondriver = read_set(args.nondriver_file)
potential = read_set(args.potential_file)

# 读三个网络的分数（index=基因名，列=score）
scores = {}
for net in NETS:
    f = os.path.join(args.score_dir, 'predicted_scores_%s.txt' % net)
    df = pd.read_csv(f, sep='\t', index_col=0)
    df.columns = ['score']
    # 索引统一大写
    df.index = [str(g).upper() for g in df.index]
    s = df['score']
    if args.use_prob:
        # 把 logits 转成 sigmoid 概率（论文称 score 为概率）
        s = pd.Series(1.0 / (1.0 + np.exp(-s.values)), index=s.index)
    scores[net] = s
    print('%s：%d 个基因有分数' % (net, len(df)))

# ============================================================================
# 实验五：4 类基因平均分数（Figure 2A）
# ============================================================================
def classify(gene):
    """把一个基因归入互斥的 4 类之一。"""
    if gene in cgc:
        return 'CDG'
    if gene in potential:
        return 'Potential'
    if gene in nondriver:
        return 'Nondriver'
    return 'Other'

categories = ['CDG', 'Potential', 'Nondriver', 'Other']
# 统计每个网络 × 每类的平均分数、中位数、数量
mean_tbl = pd.DataFrame(index=NETS, columns=categories, dtype=float)
count_tbl = pd.DataFrame(index=NETS, columns=categories, dtype=int)
dist_for_box = {}
for net in NETS:
    s = scores[net]
    cats = pd.Series([classify(g) for g in s.index], index=s.index)
    dist_for_box[net] = {c: s[cats == c].values for c in categories}
    for c in categories:
        vals = s[cats == c]
        mean_tbl.loc[net, c] = vals.mean()
        count_tbl.loc[net, c] = len(vals)

print('\n========== 实验五：各类基因平均分数（Figure 2A） ==========')
print(mean_tbl.round(4).to_string())
print('\n各类基因数量：')
print(count_tbl.to_string())
mean_tbl.to_csv(os.path.join(args.out_dir, 'exp5_category_mean_score.csv'), encoding='utf-8-sig')
count_tbl.to_csv(os.path.join(args.out_dir, 'exp5_category_count.csv'), encoding='utf-8-sig')

# 画图：每个网络一个子图，x=4类，y=平均分数（柱状）
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
colors = ['#d95f5f', '#e0a458', '#7fa8d9', '#9bbf9b']
for ax, net in zip(axes, NETS):
    means = [mean_tbl.loc[net, c] for c in categories]
    bars = ax.bar(categories, means, color=colors, edgecolor='white')
    ax.set_title(net, fontsize=12)
    ax.set_ylabel('Mean predicted score')
    ax.tick_params(axis='x', labelsize=9)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                '%.3f' % v, ha='center', va='bottom', fontsize=8)
fig.suptitle('Exp5 (Fig.2A): Mean predicted scores of four gene categories', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(args.out_dir, 'exp5_Fig2A_category_scores.png'), dpi=200)
plt.close(fig)

# ============================================================================
# 实验六：top100 三网络 Venn（Figure 2B）
# ============================================================================
# 每个网络按分数降序取 top-k（分数文件本身已排序，这里再排序保证）
top_sets = {}
for net in NETS:
    s = scores[net].sort_values(ascending=False)
    top_sets[net] = set(list(s.index)[:args.topk])
    # 保存每个网络的 top-k 名单
    with open(os.path.join(args.out_dir, 'top%d_%s.txt' % (args.topk, net)), 'w', encoding='utf-8') as f:
        for g in list(s.index)[:args.topk]:
            f.write(g + '\n')

# 用 matplotlib_venn 画三圆 Venn；没有该库则输出数据并跳过画图
venn_ok = False
try:
    from matplotlib_venn import venn3
    fig, ax = plt.subplots(figsize=(8, 7))
    venn3([top_sets['GGNet'], top_sets['PathNet'], top_sets['PPNet']],
          set_labels=('GGNet', 'PathNet', 'PPNet'), ax=ax)
    ax.set_title('Exp6 (Fig.2B): Overlap of top%d predicted genes' % args.topk, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'exp6_Fig2B_venn.png'), dpi=200)
    plt.close(fig)
    venn_ok = True
except Exception as e:
    print('\n（未画出 Venn：%s。可执行 pip install matplotlib-venn 后重跑，数值结果仍已输出。）' % e)

# 计算各重叠区的基因、数量、CDG 占比
A, B, C = top_sets['GGNet'], top_sets['PathNet'], top_sets['PPNet']
regions = {
    'GGNet only': A - B - C,
    'PathNet only': B - A - C,
    'PPNet only': C - A - B,
    'GGNet&PathNet': (A & B) - C,
    'GGNet&PPNet': (A & C) - B,
    'PathNet&PPNet': (B & C) - A,
    'All three': A & B & C,
}
rows = []
for name, genes in regions.items():
    n = len(genes)
    n_cdg = len(genes & cgc)
    frac = (n_cdg / n) if n > 0 else np.nan
    rows.append({'region': name, 'n_genes': n, 'n_CDG': n_cdg, 'CDG_fraction': frac})
overlap_tbl = pd.DataFrame(rows)
print('\n========== 实验六：top%d 各重叠区数量与 CDG 占比（Figure 2B） ==========' % args.topk)
print(overlap_tbl.round(4).to_string(index=False))
overlap_tbl.to_csv(os.path.join(args.out_dir, 'exp6_overlap_CDG_fraction.csv'),
                   index=False, encoding='utf-8-sig')

# 输出三网络共有、两网络共有的基因名单（用于论文叙述）
with open(os.path.join(args.out_dir, 'exp6_consensus_genes.txt'), 'w', encoding='utf-8') as f:
    f.write('### Predicted in all three networks (%d) ###\n' % len(regions['All three']))
    f.write('\n'.join(sorted(regions['All three'])))
    f.write('\n\n### Predicted in at least two networks ###\n')
    two_or_more = ((A & B) | (A & C) | (B & C))
    f.write('\n'.join(sorted(two_or_more)))

# ---------------------------- 收尾 ----------------------------
print('\n' + '=' * 60)
print('分析完成，结果在：%s' % args.out_dir)
print(' - exp5_Fig2A_category_scores.png （实验五，4类平均分数）')
print(' - exp6_Fig2B_venn.png （实验六，top%d Venn，venn_ok=%s）' % (args.topk, venn_ok))
print(' - 各 CSV 统计表 / 名单 TXT')
print('=' * 60)
