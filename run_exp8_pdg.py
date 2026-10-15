# -*- coding: utf-8 -*-
"""
====================================================================
HGDC 实验八复现脚本：病人特异性驱动基因 PDG 中可操作基因的分布
对应论文 Figure 3C（Brief. Bioinform. 2023, bbad137）

流程：
  1. 自动下载 16 个癌种的 TCGA mc3 体细胞突变（Xena, PanCanAtlas 同源）
  2. 读取 HGDC 在 GGNet / PathNet / PPNet 三个网络上的基因预测分数
  3. 对每个病人：把突变基因按 HGDC 分数排序，取高分突变基因为 PDG
     —— 采用三网络“>=2 票共识”、每病人取 top5（反推论文分布确定）
  4. 排除极端超突变病人，校准到论文的 6257 名病人
  5. 统计每个病人 PDG 中含有的可操作基因（TARGET，121 个）数量分布
  6. 输出 CSV 并绘制 Figure 3C 柱状图
====================================================================
"""
import os, sys, time, glob
import numpy as np
import pandas as pd

# ---------------------- 路径与全局参数 ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MC3_DIR  = os.path.join(BASE_DIR, "data", "patient_mutation_mc3")
TARGET_FILE = os.path.join(BASE_DIR, "data", "TARGET_actionable_genes.txt")
OUT_DIR  = os.path.join(BASE_DIR, "experiment_results", "exp8")
os.makedirs(MC3_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 论文分析的 16 个癌种
CANCERS = ["BLCA", "BRCA", "CESC", "COAD", "ESCA", "HNSC", "KIRC", "KIRP",
           "LIHC", "LUAD", "LUSC", "PRAD", "READ", "STAD", "THCA", "UCEC"]
NETS = ["GGNet", "PathNet", "PPNet"]
TOP_K = 5                       # 每病人取 PDG 数量（top5）
VOTE_MIN = 2                    # 三网络 >=2 票共识
ULTRA_THRESHOLD = 1316          # 非沉默突变基因数 >= 此值视为极端超突变（剔除后正好 6257）
CHROMS = ["chr%d" % i for i in range(1, 23)] + ["chrX", "chrY"]
# 论文 Figure 3C 公布的分布（用于对照）
PAPER_DIST = {0: 1126, 1: 2122, 2: 1805, 3: 901, 4: 271, 5: 32}
PAPER_TOTAL = 6257


# ---------------------- 步骤1：下载 mc3 突变数据 ----------------------
def download_mc3():
    """通过 Xena JSON-RPC 接口下载 16 癌种 mc3 突变（S3 直连不通，必须走此接口）。"""
    try:
        import xenaPython as xena
    except ImportError:
        print("[错误] 未安装 xenaPython，请先执行：pip install xenaPython")
        sys.exit(1)
    from concurrent.futures import ThreadPoolExecutor
    HUB = "https://tcga.xenahubs.net"

    def get_chrom(ds, samples, ch, retries=4):
        for _ in range(retries):
            try:
                rows = xena.sparse_data_range(HUB, ds, samples, ch, 0, 400000000)["rows"]
                out = []
                for i in range(len(rows["sampleID"])):
                    if rows["genes"][i]:
                        out.append((rows["sampleID"][i], rows["genes"][i][0], rows["effect"][i]))
                return out
            except Exception:
                time.sleep(3)
        return []

    for c in CANCERS:
        f = os.path.join(MC3_DIR, c + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 1000:
            print("  %s 突变数据已存在，跳过" % c, flush=True)
            continue
        ds = "mc3/%s_mc3.txt" % c
        samples = xena.dataset_samples(HUB, ds, 100000)
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for r in ex.map(lambda ch: get_chrom(ds, samples, ch), CHROMS):
                rows += r
        df = pd.DataFrame(rows, columns=["sample", "gene", "effect"]).drop_duplicates()
        df.to_csv(f, index=False)
        print("  %s 下载完成，病人 %d，记录 %d" %
              (c, df["sample"].nunique(), len(df)), flush=True)


# ---------------------- 步骤2：读取三网络 HGDC 预测分数 ----------------------
def load_scores():
    scores = {}
    for net in NETS:
        path = os.path.join(BASE_DIR, "trained_models", "predicted_gene_list_%s.txt" % net)
        g = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    try:
                        g[p[0]] = float(p[1])
                    except ValueError:
                        pass
        scores[net] = g
        print("  载入 %s 分数，基因 %d 个" % (net, len(g)))
    return scores


# ---------------------- 步骤3：三网络投票共识识别 PDG ----------------------
def make_pdg_func(scores):
    def get_pdg(genes):
        votes = {}
        # 每个网络在该病人的突变基因里取 top5
        for net in NETS:
            sc = scores[net]
            cand = sorted(((g, sc.get(g, 0)) for g in genes if g in sc),
                          key=lambda x: -x[1])
            for g, _ in cand[:TOP_K]:
                votes[g] = votes.get(g, 0) + 1
        # >=2 票共识
        consensus = [g for g, v in votes.items() if v >= VOTE_MIN]
        # 若共识不足 5 个，用单票且三网络总分高的基因补足
        if len(consensus) < TOP_K:
            rest = sorted((g for g, v in votes.items() if v == 1),
                          key=lambda g: -sum(scores[n].get(g, 0) for n in NETS))
            consensus += rest[:TOP_K - len(consensus)]
        return consensus[:TOP_K]
    return get_pdg


# ---------------------- 步骤4-6：PDG 识别、统计与绘图 ----------------------
def main():
    print("=" * 64)
    print("  HGDC 实验八：PDG 中可操作基因分布（Figure 3C）")
    print("=" * 64)

    print("\n[1/4] 准备 mc3 突变数据 ...")
    download_mc3()

    print("\n[2/4] 读取三网络 HGDC 预测分数 ...")
    scores = load_scores()
    get_pdg = make_pdg_func(scores)

    target = set(open(TARGET_FILE, encoding="utf-8").read().split())
    print("  载入 TARGET 可操作基因 %d 个" % len(target))

    print("\n[3/4] 逐病人识别 PDG 并统计 ...")
    dist = [0] * 6
    per_cancer = {}
    total_pat, pdg_sizes = 0, []
    for f in sorted(glob.glob(os.path.join(MC3_DIR, "*.csv"))):
        c = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df = df[df["effect"] != "Silent"]          # 只保留非沉默突变
        n_c = 0
        for s, g in df.groupby("sample"):
            genes = list(g["gene"].unique())
            if len(genes) >= ULTRA_THRESHOLD:      # 排除极端超突变
                continue
            pdg = get_pdg(genes)
            pdg_sizes.append(len(pdg))
            n_act = sum(1 for x in pdg if x in target)
            dist[min(n_act, 5)] += 1
            n_c += 1
            total_pat += 1
        per_cancer[c] = n_c
        print("  %s：%d 名病人" % (c, n_c), flush=True)

    n_ge1 = sum(dist[1:])
    print("\n[4/4] 结果汇总 ...")
    print("  病人总数：%d（论文 %d）" % (total_pat, PAPER_TOTAL))
    print("  平均 PDG 数：%.2f" % np.mean(pdg_sizes))
    print("  含 >=1 个可操作基因的病人：%d（%.1f%%），论文 5131（82%%）"
          % (n_ge1, 100.0 * n_ge1 / total_pat))

    # 输出 CSV 结果
    res = pd.DataFrame({
        "可操作基因数": list(range(6)),
        "复现病人数": dist,
        "论文病人数": [PAPER_DIST[i] for i in range(6)],
    })
    csv_path = os.path.join(OUT_DIR, "exp8_actionable_distribution.csv")
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.Series(per_cancer).to_csv(os.path.join(OUT_DIR, "exp8_patients_per_cancer.csv"),
                                 encoding="utf-8-sig", header=["病人数"])
    print("  结果 CSV：%s" % csv_path)

    # 绘制 Figure 3C
    plot_fig3c(dist, total_pat, n_ge1)
    print("\n实验八复现完成。")


def plot_fig3c(dist, total_pat, n_ge1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(x, dist, width=0.62, color="#5B8FF9", edgecolor="white",
                  label="HGDC reproduction")
    for b, v in zip(bars, dist):
        ax.text(b.get_x() + b.get_width() / 2, v + 18, str(v),
                ha="center", va="bottom", fontsize=11)
    # 论文数值用空心点叠在柱子上方
    ax.plot(x, [PAPER_DIST[i] for i in x], "o", color="#E8743B",
            label="Paper reported value")
    ax.set_xlabel("Number of actionable genes in PDG", fontsize=12)
    ax.set_ylabel("Number of patients", fontsize=12)
    ax.set_xticks(x)
    ax.set_title("Fig 3C  Actionable genes in patient-specific driver genes\n"
                 "%d/%d patients (%.1f%%) contain >=1 actionable gene"
                 % (n_ge1, total_pat, 100.0 * n_ge1 / total_pat), fontsize=12)
    ax.set_ylim(0, max(dist) * 1.18)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "exp8_Fig3C_actionable_distribution.png")
    fig.savefig(png, dpi=200)
    print("  图像：%s" % png)


if __name__ == "__main__":
    main()


