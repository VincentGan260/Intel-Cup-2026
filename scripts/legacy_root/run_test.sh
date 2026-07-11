#!/bin/bash
# DK 2500 快速测试脚本

set -e

echo "=========================================="
echo "Intel-Cup-2026 DK 2500 快速测试"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查环境
echo -e "\n${YELLOW}[1/4] 检查环境...${NC}"

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 未安装"
    exit 1
fi

# 检查 OpenVINO
if python3 -c "import openvino" 2>/dev/null; then
    OV_VERSION=$(python3 -c "import openvino; print(openvino.__version__)")
    echo -e "${GREEN}✓${NC} OpenVINO: $OV_VERSION"
else
    echo -e "${RED}✗${NC} OpenVINO 未安装"
    echo "请运行: pip install openvino"
    exit 1
fi

# 检查 Ultralytics
if python3 -c "from ultralytics import YOLO" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Ultralytics 已安装"
else
    echo -e "${YELLOW}!${NC} Ultralytics 未安装，正在安装..."
    pip install ultralytics -q
fi

# 下载模型（如需要）
echo -e "\n${YELLOW}[2/4] 检查模型文件...${NC}"

if [ ! -f "yolo26n.pt" ]; then
    echo "下载 YOLO26n 模型..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
    echo -e "${GREEN}✓${NC} 模型已下载"
else
    echo -e "${GREEN}✓${NC} 模型已存在"
fi

# 检查数据集
echo -e "\n${YELLOW}[3/4] 检查测试数据...${NC}"

if [ -d "datasets/bdd100k_subset_500/images/val" ]; then
    IMG_COUNT=$(ls datasets/bdd100k_subset_500/images/val/*.jpg 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} 测试数据: $IMG_COUNT 张图片"
else
    echo -e "${YELLOW}!${NC} 测试数据集不存在"
    echo "请确保 datasets/bdd100k_subset_500/images/val/ 目录存在"
fi

# 运行测试
echo -e "\n${YELLOW}[4/4] 运行测试...${NC}"

# 创建输出目录
mkdir -p runs/comprehensive_eval runs/segmentation_eval runs/final_report

echo ""
read -p "请选择测试类型:
  1. 目标检测测试 (推荐首次运行)
  2. 语义分割测试
  3. 生成报告
  4. 全部运行
  5. 退出
请输入选项 [1-5]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}运行目标检测测试...${NC}"
        python3 scripts/comprehensive_eval.py
        ;;
    2)
        echo -e "\n${GREEN}运行语义分割测试...${NC}"
        python3 scripts/segmentation_eval.py
        ;;
    3)
        echo -e "\n${GREEN}生成报告...${NC}"
        python3 scripts/generate_report.py
        ;;
    4)
        echo -e "\n${GREEN}运行全部测试...${NC}"
        python3 scripts/comprehensive_eval.py
        python3 scripts/segmentation_eval.py
        python3 scripts/generate_report.py
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=========================================="
echo "测试完成!"
echo "==========================================${NC}"

echo ""
echo "结果文件:"
echo "  - 目标检测: runs/comprehensive_eval/summary.csv"
echo "  - 语义分割: runs/segmentation_eval/summary.csv"
echo "  - 最终报告: runs/final_report/test_report.md"

echo ""
read -p "是否打开结果目录? [y/N]: " open_result
if [ "$open_result" = "y" ] || [ "$open_result" = "Y" ]; then
    xdg-open runs/ 2>/dev/null || echo "请手动打开 runs/ 目录查看结果"
fi
