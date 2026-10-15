#!/bin/bash
# ============================================================================
# HGDC 复现脚本（单卡串行）【仅 MTGCN × 3 网络】
#   模型：MTGCN
#   网络：GGNet / PathNet / PPNet（每个网络 5 折 × 10 次）
#
# 适配：NVIDIA RTX 4090，conda 环境 hgdc
# 说明：不修改任何基准文件，只调用官方入口 run_baselines_5CV_test.py；
#       GPU 负采样补丁通过临时 sitecustomize.py + PYTHONPATH 注入，仅对本脚本生效。
#
# 超参严格按官方 README：MTGCN --epochs 1000 --lr 0.01
# ============================================================================
set -uo pipefail
# ---------------------------- 路径与环境配置 ----------------------------
PROJECT_DIR="/root/private_data/users/wangxin/HGDC"
CONDA_BASE="/root/private_data/users/wangxin/miniconda3"
CONDA_ENV="hgdc"
DEVICE=0
# 日志目录
RESULT_DIR="${PROJECT_DIR}/experiment_results"
CONSOLE_LOG="${RESULT_DIR}/run_mtgcn_console.log"
mkdir -p "${RESULT_DIR}"
: > "${CONSOLE_LOG}"   # 清空本次进度日志
# ---------------------------- 三个生物网络 ----------------------------
NETWORKS=(
  "GGNet:./data/GGNet/dataset_GGNet_ten_5CV.pkl:111"
  "PathNet:./data/PathNet/dataset_PathNet_ten_5CV.pkl:24"
  "PPNet:./data/PPNet/dataset_PPNet_ten_5CV.pkl:50"
)
# ---------------------------- 进度日志函数 ----------------------------
log(){ echo "$@" | tee -a "${CONSOLE_LOG}"; }
# ---------------------------- 激活 conda 环境 ----------------------------
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}" || { echo "无法进入项目目录 ${PROJECT_DIR}"; exit 1; }
# 打印 torch / torch_geometric 版本，确认 torch.isin 是否存在（便于留痕）
python - <<'PYV'
import torch
print("torch =", torch.__version__, "| torch.isin =", hasattr(torch, "isin"),
      "| CUDA =", torch.cuda.is_available())
PYV
# ============================================================================
# MTGCN GPU 负采样补丁（不修改任何基准文件，仅在跑 MTGCN 时用 PYTHONPATH 注入）
#   语义与 PyG 官方 negative_sampling 完全等价：不输出正边/自环、不重复、
#   均匀无放回、数量一致；任何失败都回退官方 CPU 实现（慢但正确）。
# ============================================================================
PATCH_DIR="$(mktemp -d /tmp/hgdc_mtcn_patch.XXXXXX)"
cleanup(){ [ -d "${PATCH_DIR}" ] && rm -rf "${PATCH_DIR}"; }
trap cleanup EXIT
cat > "${PATCH_DIR}/sitecustomize.py" <<'PYEOF'
# -*- coding: utf-8 -*-
import torch
import torch_geometric.utils as _u

# 先保存官方原始实现，作为正确兜底
_original_negative_sampling = _u.negative_sampling
_warned = False


def _gpu_negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                           method="sparse", force_undirected=False, **kwargs):
    global _warned

    # 二分图 / 强制无向：官方编码不同，直接交回官方
    if isinstance(num_nodes, (tuple, list)) or force_undirected:
        return _original_negative_sampling(
            edge_index, num_nodes=num_nodes,
            num_neg_samples=num_neg_samples, method=method,
            force_undirected=force_undirected)

    try:
        device = edge_index.device
        ei = edge_index.long()
        if num_nodes is None:
            n = int(ei.max().item()) + 1
        else:
            n = int(num_nodes)

        need = int(ei.size(1)) if num_neg_samples is None else int(num_neg_samples)
        if need <= 0:
            return ei.new_empty((2, 0))

        total = n * n
        # 禁边 = 正边 + 全部自环（即使输入不含自环也显式补上，与官方一致）
        row, col = ei[0], ei[1]
        forbidden = torch.unique(row * n + col)
        loops = torch.arange(n, device=device, dtype=torch.long) * (n + 1)
        forbidden = torch.unique(torch.cat([forbidden, loops]))
        valid_total = total - int(forbidden.numel())

        if valid_total < need:
            return _original_negative_sampling(
                edge_index, num_nodes=num_nodes,
                num_neg_samples=num_neg_samples, method=method,
                force_undirected=False)

        # available[code]=True 表示该编号是“合法且未选中”的负边；选中后置 False
        available = torch.ones(total, dtype=torch.bool, device=device)
        available[forbidden] = False
        p_valid = valid_total / float(total)

        picked = []
        got = 0
        for _ in range(20):
            remain = need - got
            k = int(remain / max(p_valid, 1e-6) * 1.2) + 1024
            cand = torch.randint(0, total, (k,), device=device)
            cand = cand[available[cand]]      # 只保留合法、未选过的
            cand = torch.unique(cand)         # 本轮去重（unique 返回升序）
            if cand.numel() > remain:
                # 关键：不能直接 [:remain]（会只留下小编号节点），先随机打乱再截取
                perm = torch.randperm(cand.numel(), device=device)[:remain]
                cand = cand[perm]
            if cand.numel() > 0:
                available[cand] = False
                picked.append(cand)
                got += int(cand.numel())
            if got >= need:
                break

        if got < need:
            raise RuntimeError("GPU sampling short: got %d / need %d" % (got, need))

        neg_code = torch.cat(picked)
        src = neg_code // n                 # 非负整数 // == floor，兼容所有 torch 版本
        dst = neg_code % n
        return torch.stack([src, dst], dim=0)

    except Exception as e:
        # 红线：任何失败都不能产出脏数据，退回官方 CPU 实现（慢但正确）
        if not _warned:
            print("[neg_sampling] GPU fast path failed(%r), "
                  "fall back to official negative_sampling (slower but correct)." % (e,),
                  flush=True)
            _warned = True
        return _original_negative_sampling(
            edge_index, num_nodes=num_nodes,
            num_neg_samples=num_neg_samples, method=method,
            force_undirected=force_undirected)


_u.negative_sampling = _gpu_negative_sampling
PYEOF
# ---------------------------- 启动提示 ----------------------------
log "============================================================"
log "MTGCN 复现启动（仅 MTGCN × 3 网络）  $(date '+%F %T')"
log "环境: ${CONDA_ENV}   GPU: ${DEVICE}   结果目录: ${RESULT_DIR}"
log "============================================================"
# ============================================================================
# 仅运行 MTGCN：epochs=1000, lr=0.01（官方 README 设定）
# ============================================================================
EPOCHS=1000
LR=0.01
for net in "${NETWORKS[@]}"; do
  IFS=':' read -r NAME DATA AVGDEG <<< "${net}"
  LOG="${RESULT_DIR}/${NAME}_MTGCN_baseline.log"
  log ">>> MTGCN - ${NAME}（日志: ${LOG}）"
  PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}" python run_baselines_5CV_test.py \
    --graph_diffusion none --is_5_CV_test True \
    --dataset_file "${DATA}" --model "MTGCN" \
    --epochs "${EPOCHS}" --lr "${LR}" --device "${DEVICE}" > "${LOG}" 2>&1
  if [ $? -eq 0 ]; then log "  [完成] ${NAME} - MTGCN"; \
  else log "  [失败] ${NAME} - MTGCN，请查看 ${LOG}"; fi
done
# ============================================================================
# 结果汇总（提取每个日志最后的 Mean AUC / Mean AUPR）
# ============================================================================
SUMMARY="${RESULT_DIR}/results_summary_mtgcn.txt"
: > "${SUMMARY}"
for net in "${NETWORKS[@]}"; do
  IFS=':' read -r NAME DATA AVGDEG <<< "${net}"
  f="${RESULT_DIR}/${NAME}_MTGCN_baseline.log"
  line="$(grep -E "10 rounds for 5CV" "${f}" | tail -1)"
  if [ -n "${line}" ]; then
    echo "${NAME}_MTGCN_baseline.log : ${line}" >> "${SUMMARY}"
  fi
done
log ""
log "============================================================"
log "MTGCN 全部执行完成  $(date '+%F %T')"
log "------------------------------------------------------------"
cat "${SUMMARY}" | tee -a "${CONSOLE_LOG}"
log "------------------------------------------------------------"
log "进度日志：${CONSOLE_LOG}"
log "结果汇总：${SUMMARY}"
log "============================================================"
