#!/bin/bash
# ============================================================================
# HGDC 全量复现脚本（单卡串行）
#   模块1：HGDC 主实验（GGNet / PathNet / PPNet）
#   模块2：6 个基准模型（MTGCN / EMOGI / GCN / GAT / Chebnet / SVM）× 3 网络
#   模块3：3 组消融实验（PPR / Separation / Attention）× 3 网络（默认关闭）
#
# 适配：NVIDIA RTX 4090，conda 环境 hgdc
# 说明：基准一律调用官方入口 run_baselines_5CV_test.py，不修改任何基准文件
# ============================================================================
set -uo pipefail

# ---------------------------- 路径与环境配置 ----------------------------
PROJECT_DIR="/root/private_data/users/wangxin/HGDC"
CONDA_BASE="/root/private_data/users/wangxin/miniconda3"
CONDA_ENV="hgdc"
DEVICE=0

# 是否运行消融（消融需要额外的 models/ablation.py 与 run_ablation_5CV.py，
# 官方仓库不自带，故默认 0 关闭；拿到文件后改为 1）
RUN_ABLATION=1

# 日志目录
RESULT_DIR="${PROJECT_DIR}/experiment_results"
CONSOLE_LOG="${RESULT_DIR}/run_all_console.log"
mkdir -p "${RESULT_DIR}"
: > "${CONSOLE_LOG}"   # 清空本次进度日志

# ---------------------------- 三个生物网络 ----------------------------
# 格式：名称:数据集pkl:网络边txt:网络平均度
NETWORKS=(
  "GGNet:./data/GGNet/dataset_GGNet_ten_5CV.pkl:./data/GGNet/GGNet.txt:111"
  "PathNet:./data/PathNet/dataset_PathNet_ten_5CV.pkl:./data/PathNet/PathNet.txt:24"
  "PPNet:./data/PPNet/dataset_PPNet_ten_5CV.pkl:./data/PPNet/PPNet.txt:50"
)
# ---------------------------- 进度日志函数 ----------------------------
log(){ echo "$@" | tee -a "${CONSOLE_LOG}"; }

# ---------------------------- 激活 conda 环境 ----------------------------
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}" || { echo "无法进入项目目录 ${PROJECT_DIR}"; exit 1; }

# ============================================================================
# MTGCN GPU 加速补丁（不修改任何基准文件）
#   根因：MTGCN 每个 epoch 都调用 negative_sampling，而 PyG 的 sparse/dense
#         默认实现把负采样放在 CPU 上（random.sample + np.isin），GGNet 大图
#         单次约 1.5~2 秒，累计 50×1000 次要二十多小时，GPU 却几乎闲置。
#   做法：动态生成一个临时 sitecustomize.py，通过 PYTHONPATH 注入，把
#         negative_sampling 替换为 GPU 实现（torch.randint + torch.isin 去重），
#         仅在跑 MTGCN 时生效，脚本结束自动清理。
# ============================================================================
PATCH_DIR="$(mktemp -d /tmp/hgdc_mtcn_patch.XXXXXX)"
cleanup(){ [ -d "${PATCH_DIR}" ] && rm -rf "${PATCH_DIR}"; }
trap cleanup EXIT

cat > "${PATCH_DIR}/sitecustomize.py" <<'PYEOF'
import torch
import torch_geometric.utils as _u


def _fast_negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                            method="sparse", force_undirected=False, **kw):
    """GPU 版负采样，语义与官方一致：返回真正不存在于正边集合中的负边。"""
    try:
        device = edge_index.device
        if isinstance(num_nodes, (tuple, list)):
            n_src, n_dst = int(num_nodes[0]), int(num_nodes[1])
        else:
            n = int(num_nodes) if num_nodes is not None \
                else int(edge_index.max().item()) + 1
            n_src = n_dst = n

        # 正边（含自环）编码为一维编号，便于集合排除
        row, col = edge_index[0].long(), edge_index[1].long()
        pos_code = torch.unique(row * n_dst + col)

        need = edge_index.size(1) if num_neg_samples is None \
            else int(num_neg_samples)
        if force_undirected:
            need //= 2

        out, remaining, guard = [], need, 0
        while remaining > 0 and guard < 20:
            guard += 1
            k = int(remaining * 1.2) + 1000
            cand = torch.unique(
                torch.randint(0, n_src * n_dst, (k,), device=device))
            cand = cand[~torch.isin(cand, pos_code)]
            if out:
                cand = cand[~torch.isin(cand, torch.cat(out))]
            if cand.numel() > remaining:
                cand = cand[:remaining]
            if cand.numel() > 0:
                out.append(cand)
                remaining -= cand.numel()

        if not out:
            raise RuntimeError("empty negative set")

        neg_code = torch.cat(out)
        src = torch.div(neg_code, n_dst, rounding_mode="floor")
        dst = neg_code % n_dst
        return torch.stack([src, dst], dim=0)

    except Exception:
        # 兜底：纯 GPU 随机（极个别可能为正边），保证不会卡死
        device = edge_index.device
        if isinstance(num_nodes, (tuple, list)):
            n_src, n_dst = int(num_nodes[0]), int(num_nodes[1])
        else:
            n_src = n_dst = int(num_nodes) if num_nodes is not None \
                else int(edge_index.max().item()) + 1
        k = int(num_neg_samples) if num_neg_samples is not None \
            else edge_index.size(1)
        src = torch.randint(0, n_src, (k,), device=device)
        dst = torch.randint(0, n_dst, (k,), device=device)
        return torch.stack([src, dst], dim=0)


_u.negative_sampling = _fast_negative_sampling
PYEOF

# ---------------------------- 启动提示 ----------------------------
log "============================================================"
log "HGDC 全量复现启动  $(date '+%F %T')"
log "环境: ${CONDA_ENV}   GPU: ${DEVICE}   结果目录: ${RESULT_DIR}"
log "============================================================"

# ============================================================================
# 模块 1：HGDC 主实验
# ============================================================================
log ""
log "########## 模块 1/3：HGDC 主实验 ##########"
for net in "${NETWORKS[@]}"; do
  IFS=':' read -r NAME DATA NET_FILE AVGDEG <<< "${net}"
  LOG="${RESULT_DIR}/${NAME}_HGDC_main.log"
  log ">>> HGDC 主实验 - ${NAME}（日志: ${LOG}）"
  python run_HGDC_5CV.py \
    --graph_diffusion ppr \
    --ppr_alpha 0.9 --ppr_eps 0.0001 \
    --net_avg_deg "${AVGDEG}" --hk_t 1 \
    --is_5_CV_test True \
    --dataset_file "${DATA}" --net_file "${NET_FILE}" \
    --epochs 100 --lr 0.001 --w_decay 0.00001 \
    --in_channels 58 --hidden_channels 100 \
    --device "${DEVICE}" > "${LOG}" 2>&1
  if [ $? -eq 0 ]; then log "  [完成] ${NAME} HGDC 主实验"; \
  else log "  [失败] ${NAME} HGDC 主实验，请查看 ${LOG}"; fi
done

# ============================================================================
# 模块 2：基准模型对比（全部走官方入口 run_baselines_5CV_test.py）
# ============================================================================
log ""
log "########## 模块 2/3：基准模型对比 ##########"
# 格式：模型名:训练轮数:学习率（按 README / 论文设置，不同模型不同）
BASELINES=(
  "MTGCN:1000:0.01"
  "EMOGI:5000:0.01"
  "GCN:5000:0.01"
  "GAT:5000:0.01"
  "Chebnet:5000:0.01"
  "SVM:5000:0.01"
)
for b in "${BASELINES[@]}"; do
  IFS=':' read -r M EPOCHS LR <<< "${b}"
  log ">>> 基准模型：${M}"
  for net in "${NETWORKS[@]}"; do
    IFS=':' read -r NAME DATA NET_FILE AVGDEG <<< "${net}"
    LOG="${RESULT_DIR}/${NAME}_${M}_baseline.log"
    if [ "${M}" = "MTGCN" ]; then
      # MTGCN 注入 GPU 负采样补丁（仅本命令生效）
      PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}" python run_baselines_5CV_test.py \
        --graph_diffusion none --is_5_CV_test True \
        --dataset_file "${DATA}" --model "${M}" \
        --epochs "${EPOCHS}" --lr "${LR}" --device "${DEVICE}" > "${LOG}" 2>&1
    else
      python run_baselines_5CV_test.py \
        --graph_diffusion none --is_5_CV_test True \
        --dataset_file "${DATA}" --model "${M}" \
        --epochs "${EPOCHS}" --lr "${LR}" --device "${DEVICE}" > "${LOG}" 2>&1
    fi
    if [ $? -eq 0 ]; then log "  [完成] ${NAME} - ${M}"; \
    else log "  [失败] ${NAME} - ${M}，请查看 ${LOG}"; fi
  done
done

# ============================================================================
# 模块 3：消融实验（默认关闭，需要额外文件）
# ============================================================================
log ""
if [ "${RUN_ABLATION}" = "1" ]; then
  log "########## 模块 3/3：消融实验 ##########"
  for A in ppr separation attention; do
    for net in "${NETWORKS[@]}"; do
      IFS=':' read -r NAME DATA NET_FILE AVGDEG <<< "${net}"
      LOG="${RESULT_DIR}/${NAME}_ablation_${A}.log"
      log ">>> 消融 ${A} - ${NAME}（日志: ${LOG}）"
      python run_ablation_5CV.py \
        --ablation "${A}" \
        --dataset_file "${DATA}" --net_file "${NET_FILE}" \
        --net_avg_deg "${AVGDEG}" \
        --epochs 100 --lr 0.001 --w_decay 0.00001 \
        --in_channels 58 --hidden_channels 100 \
        --device "${DEVICE}" > "${LOG}" 2>&1
      if [ $? -eq 0 ]; then log "  [完成] ${NAME} - 消融 ${A}"; \
      else log "  [失败] ${NAME} - 消融 ${A}，请查看 ${LOG}"; fi
    done
  done
else
  log "########## 模块 3/3：消融实验已跳过（RUN_ABLATION=0）##########"
fi

# ============================================================================
# 结果汇总（提取每个实验日志最后的 Mean AUC / Mean AUPR）
# ============================================================================
SUMMARY="${RESULT_DIR}/results_summary.txt"
: > "${SUMMARY}"
for f in "${RESULT_DIR}"/*.log; do
  case "$(basename "${f}")" in
    run_all_console.log) continue ;;
  esac
  line="$(grep -E "10 rounds for 5CV" "${f}" | tail -1)"
  if [ -n "${line}" ]; then
    echo "$(basename "${f}") : ${line}" >> "${SUMMARY}"
  fi
done

log ""
log "============================================================"
log "全部实验执行完成  $(date '+%F %T')"
log "------------------------------------------------------------"
cat "${SUMMARY}" | tee -a "${CONSOLE_LOG}"
log "------------------------------------------------------------"
log "进度日志：${CONSOLE_LOG}"
log "结果汇总：${SUMMARY}"
log "============================================================"
