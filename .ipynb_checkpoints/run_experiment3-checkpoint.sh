#!/bin/bash
# ==============================================================
# 实验三：HGDC 独立测试集（OncoKB / CancerMine）批量串行运行
# 组合：2 个独立名单 × 3 个网络，仅评估 HGDC，共 6 组
# 所有日志保存到 experiment_results/independent 文件夹。
# ==============================================================

# -------------------- 全局配置 --------------------
PROJECT_DIR="/root/private_data/users/wangxin/HGDC"
CONDA_PATH="/root/private_data/users/wangxin/miniconda3"
CONDA_ENV="hgdc"
DEVICE=0

RESULT_DIR="${PROJECT_DIR}/experiment_results/independent"

# HGDC 超参数
HGDC_EPOCHS=100
HGDC_LR=0.001
W_DECAY=0.00001
IN_CHANNELS=58
HIDDEN_CHANNELS=100
PPR_ALPHA=0.9
PPR_EPS=0.0001
HK_T=1

# -------------------- 网络配置 --------------------
# 格式：名称:数据pkl:边txt:avg_deg
NETWORKS=(
    "GGNet:./data/GGNet/dataset_GGNet_ten_5CV.pkl:./data/GGNet/GGNet.txt:111"
    "PathNet:./data/PathNet/dataset_PathNet_ten_5CV.pkl:./data/PathNet/PathNet.txt:24"
    "PPNet:./data/PPNet/dataset_PPNet_ten_5CV.pkl:./data/PPNet/PPNet.txt:50"
)

# -------------------- 独立名单配置 --------------------
# 格式：名称:名单文件
INDEPENDENT_SETS=(
    "OncoKB:./data/OncoKB_genes.txt"
    "CancerMine:./data/CancerMine_driver.txt"
)

# -------------------- 初始化环境 --------------------
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}
mkdir -p ${RESULT_DIR}

echo "=================================================="
echo "  实验三：HGDC 独立测试集（OncoKB / CancerMine）"
echo "  环境：${CONDA_ENV} | GPU：${DEVICE}"
echo "  日志目录：${RESULT_DIR}"
echo "=================================================="

# -------------------- HGDC 独立测试 --------------------
echo ""
echo "########## HGDC 独立测试 ##########"
for ind_info in "${INDEPENDENT_SETS[@]}"; do
    IFS=':' read -r IND_NAME IND_FILE <<< "${ind_info}"
    for net_info in "${NETWORKS[@]}"; do
        IFS=':' read -r NET_NAME DATASET NET_FILE AVG_DEG <<< "${net_info}"
        LOG_FILE="${RESULT_DIR}/${NET_NAME}_HGDC_${IND_NAME}.log"
        echo ">>> HGDC - ${NET_NAME} - ${IND_NAME}"
        python run_independent_test.py \
            --graph_diffusion "ppr" \
            --ppr_alpha ${PPR_ALPHA} \
            --ppr_eps ${PPR_EPS} \
            --net_avg_deg ${AVG_DEG} \
            --hk_t ${HK_T} \
            --is_5_CV_test True \
            --dataset_file "${DATASET}" \
            --net_file "${NET_FILE}" \
            --independent_set "${IND_NAME}" \
            --independent_file "${IND_FILE}" \
            --epochs ${HGDC_EPOCHS} \
            --lr ${HGDC_LR} \
            --w_decay ${W_DECAY} \
            --in_channels ${IN_CHANNELS} \
            --hidden_channels ${HIDDEN_CHANNELS} \
            --device ${DEVICE} > "${LOG_FILE}" 2>&1
        echo "  - 完成：${NET_NAME} HGDC ${IND_NAME}"
    done
done

# -------------------- 收尾 --------------------
echo ""
echo "=================================================="
echo "  HGDC 独立测试全部完成，日志：${RESULT_DIR}"
echo "=================================================="
echo "批量提取结果："
echo '   for f in '${RESULT_DIR}'/*.log; do echo -n "$(basename $f): "; grep -E "Mean AUC" "$f" | tail -1; done'
