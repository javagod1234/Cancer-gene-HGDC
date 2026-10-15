#!/bin/bash
# ==============================================================
# 实验五、六：HGDC 重复训练 100 次 → 全网络基因平均分数
# 在三个网络（GGNet / PathNet / PPNet）上串行运行 train_HGDC.py
# 每个网络输出一份“基因分数排名表”，供实验五、六分析使用。
# ==============================================================

# -------------------- 全局配置 --------------------
PROJECT_DIR="/root/private_data/users/wangxin/HGDC"
CONDA_PATH="/root/private_data/users/wangxin/miniconda3"
CONDA_ENV="hgdc"
DEVICE=0

# 分数文件统一存放目录
SCORE_DIR="${PROJECT_DIR}/predicted_scores"
LOG_DIR="${PROJECT_DIR}/experiment_results/exp56"

# 训练超参数（与论文一致）
TIMES=100        # 每个网络重复训练 100 次
EPOCHS=100
LR=0.001
W_DECAY=0.00001
IN_CHANNELS=58
HIDDEN_CHANNELS=100
PPR_ALPHA=0.9
PPR_EPS=0.0001
HK_T=1

# -------------------- 网络配置 --------------------
# 格式：名称:数据pkl(非CV版，含mask键):边txt:avg_deg
NETWORKS=(
    "GGNet:./data/GGNet/dataset_GGNet.pkl:./data/GGNet/GGNet.txt:111"
    "PathNet:./data/PathNet/dataset_PathNet.pkl:./data/PathNet/PathNet.txt:24"
    "PPNet:./data/PPNet/dataset_PPNet.pkl:./data/PPNet/PPNet.txt:50"
)

# -------------------- 初始化环境 --------------------
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR}
mkdir -p ${SCORE_DIR} ${LOG_DIR}

# 检查 numpy 版本：SciPy 要求 >=1.22.4 且 <2.0，统一用 1.26.4
# （1.21.6 偏低会报警告，2.x 偏高会导致 sklearn/scipy 二进制不兼容）
NEED_NUMPY=$(python -c "import numpy as np; v=[int(x) for x in np.__version__.split('.')[:2]]; print(0 if (v[0]==1 and v[1]>=22) else 1)")
if [ "${NEED_NUMPY}" = "1" ]; then
    echo "当前 numpy 不符合要求（需 >=1.22.4 且 <2.0），自动安装 numpy==1.26.4 ..."
    pip install "numpy==1.26.4" --quiet
fi

# 检查 numba：PyG 的 GDC 在 exact=True 时强制 import numba（精确 PPR 计算）。
# 旧 numba（如 0.55/0.56）与 numpy 1.26 ABI 不兼容，会报
# "SystemError: initialization of _internal failed"。需 numba>=0.59（配 llvmlite 0.42）。
NUMBA_OK=$(python -c "
try:
    import numba
    import numba.np.ufunc._internal
    print('1')
except Exception:
    print('0')
")
if [ "${NUMBA_OK}" != "1" ]; then
    echo "numba 与当前 numpy 不兼容，自动安装 numba==0.59.1 llvmlite==0.42.0 ..."
    pip install "numba==0.59.1" "llvmlite==0.42.0" --quiet
fi
python -c "import numpy,numba; print('numpy', numpy.__version__, '| numba', numba.__version__, '均可用')"

# 确认 CUDA 可用（不可用时 train_HGDC 会退回 CPU，这里给出提示）
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"

echo "=================================================="
echo "  实验五、六：HGDC 100 次重复训练"
echo "  环境：${CONDA_ENV} | GPU：${DEVICE}"
echo "  分数目录：${SCORE_DIR}"
echo "=================================================="

# -------------------- 逐个网络训练 --------------------
for net_info in "${NETWORKS[@]}"; do
    IFS=':' read -r NET_NAME DATASET NET_FILE AVG_DEG <<< "${net_info}"
    LOG_FILE="${LOG_DIR}/${NET_NAME}_train100.log"
    echo ">>> 训练 HGDC ×${TIMES} - ${NET_NAME}"

    # 注意：train_HGDC.py 不传 --is_5_CV_test，使用固定的全部有标签节点 mask
    python train_HGDC.py \
        --graph_diffusion "ppr" \
        --ppr_alpha ${PPR_ALPHA} \
        --ppr_eps ${PPR_EPS} \
        --net_avg_deg ${AVG_DEG} \
        --hk_t ${HK_T} \
        --dataset_file "${DATASET}" \
        --net_file "${NET_FILE}" \
        --times ${TIMES} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --w_decay ${W_DECAY} \
        --in_channels ${IN_CHANNELS} \
        --hidden_channels ${HIDDEN_CHANNELS} \
        --device ${DEVICE} > "${LOG_FILE}" 2>&1

    # train_HGDC.py 默认输出 predicted_socres_HGDC.txt，按网络重命名
    if [ -f "./predicted_socres_HGDC.txt" ]; then
        mv ./predicted_socres_HGDC.txt "${SCORE_DIR}/predicted_scores_${NET_NAME}.txt"
        echo "  - 完成，分数：${SCORE_DIR}/predicted_scores_${NET_NAME}.txt"
    else
        echo "  ! 警告：${NET_NAME} 未生成分数文件，请查看日志 ${LOG_FILE}"
    fi
done

# -------------------- 收尾 --------------------
echo ""
echo "=================================================="
echo "  三个网络 100 次训练全部完成"
echo "  分数文件：${SCORE_DIR}/predicted_scores_<网络>.txt"
echo "  下一步：运行 analyze_exp56.py 出实验五、六结果"
echo "=================================================="
