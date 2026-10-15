#!/bin/bash
# ==============================================================
# HGDC 实验八复现：PDG 中可操作基因的分布（Figure 3C）
# 自动激活 conda 环境、串行运行、日志统一保存
# 用法：
#   nohup ./run_exp8.sh > run_exp8.nohup.log 2>&1 &
#   tail -f run_exp8.nohup.log
# ==============================================================

# --------------------- 路径配置（按实际情况修改） ---------------------
PROJECT_DIR="/root/private_data/users/wangxin/HGDC"
CONDA_PATH="/root/private_data/users/wangxin/miniconda3"
CONDA_ENV="hgdc"

RESULT_DIR="${PROJECT_DIR}/experiment_results/exp8"
mkdir -p "${RESULT_DIR}"

echo "=================================================="
echo "  HGDC 实验八复现（Figure 3C）"
echo "  项目目录：${PROJECT_DIR}"
echo "  环境：${CONDA_ENV}"
echo "  开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# --------------------- 激活环境 ---------------------
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd ${PROJECT_DIR} || { echo "项目目录不存在：${PROJECT_DIR}"; exit 1; }

# --------------------- 检查依赖（xenaPython、matplotlib） ---------------------
python - <<'PY'
import importlib, subprocess, sys
for pkg, pip_name in [("xenaPython","xenaPython"), ("matplotlib","matplotlib"),
                      ("pandas","pandas"), ("numpy","numpy")]:
    if importlib.util.find_spec(pkg) is None:
        print("缺少依赖 %s，正在安装 ..." % pip_name)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
print("依赖检查完成")
PY

# --------------------- 串行运行实验八 ---------------------
python run_exp8_pdg.py 2>&1 | tee "${RESULT_DIR}/run_exp8.log"

echo "=================================================="
echo "  实验八执行完成，结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果目录：${RESULT_DIR}"
echo "  分布图：${RESULT_DIR}/exp8_Fig3C_actionable_distribution.png"
echo "  分布表：${RESULT_DIR}/exp8_actionable_distribution.csv"
echo "=================================================="
