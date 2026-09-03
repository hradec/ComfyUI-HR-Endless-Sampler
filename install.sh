#!/bin/bash
# ============================================================
# HR Endless Sampler (mickeylan fork) 安装脚本
# ============================================================
# 适用：中文用户 / 12GB 显存用户
# 特色：支持 Qwen3.5/3.6/3.8 导演
# ============================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HR Endless Sampler (mickeylan fork)${NC}"
echo -e "${GREEN}中文用户 / 低显存用户优化版${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# 检测 ComfyUI 目录
detect_comfyui() {
    # 常见位置
    local paths=(
        "$HOME/ComfyUI"
        "$HOME/comfyui"
        "$(pwd)/ComfyUI"
        "/opt/ComfyUI"
    )
    
    for path in "${paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/main.py" ]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

# 查找 Python 解释器
detect_python() {
    local comfyui_dir="$1"
    
    # ComfyUI 自带的 Python
    if [ -f "$comfyui_dir/venv/bin/python" ]; then
        echo "$comfyui_dir/venv/bin/python"
        return 0
    fi
    if [ -f "$comfyui_dir/venv/bin/python3" ]; then
        echo "$comfyui_dir/venv/bin/python3"
        return 0
    fi
    if [ -f "$comfyui_dir/python_embeded/python.exe" ]; then
        echo "$comfyui_dir/python_embeded/python.exe"
        return 0
    fi
    
    # 系统 Python
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    if command -v python &> /dev/null; then
        echo "python"
        return 0
    fi
    
    return 1
}

# 主流程
main() {
    # 1. 查找 ComfyUI
    echo -e "${YELLOW}[1/4] 查找 ComfyUI 目录...${NC}"
    COMFYUI_DIR=$(detect_comfyui)
    
    if [ -z "$COMFYUI_DIR" ]; then
        echo -e "${RED}错误：未找到 ComfyUI 目录！${NC}"
        echo "请确保 ComfyUI 已安装，或将本插件放到 ComfyUI/custom_nodes/ 目录"
        exit 1
    fi
    echo -e "${GREEN}✓ 找到 ComfyUI: $COMFYUI_DIR${NC}"
    
    # 2. 确认插件目录
    PLUGIN_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-MiniMax-H3-Sampler-Unlimited"
    
    if [ ! -d "$PLUGIN_DIR" ]; then
        echo -e "${RED}错误：插件目录不存在！${NC}"
        echo "请将本插件放到: $PLUGIN_DIR"
        exit 1
    fi
    echo -e "${GREEN}✓ 找到插件: $PLUGIN_DIR${NC}"
    
    # 3. 查找 Python
    echo -e "${YELLOW}[2/4] 查找 Python 解释器...${NC}"
    PYTHON=$(detect_python "$COMFYUI_DIR")
    
    if [ -z "$PYTHON" ]; then
        echo -e "${RED}错误：未找到 Python 解释器！${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 使用 Python: $PYTHON${NC}"
    
    # 4. 安装依赖
    echo -e "${YELLOW}[3/4] 安装依赖...${NC}"
    echo
    
    # 检查当前版本
    echo "检查 llama-cpp-python 版本..."
    if $PYTHON -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null; then
        echo -e "${GREEN}✓ llama-cpp-python 已安装${NC}"
    else
        echo -e "${YELLOW}需要安装 llama-cpp-python...${NC}"
    fi
    
    echo
    echo "运行: pip install -r requirements.txt"
    cd "$PLUGIN_DIR"
    $PYTHON -m pip install -r requirements.txt
    
    echo
    echo -e "${YELLOW}[4/4] 验证安装...${NC}"
    
    # 编译检查
    echo "Python 编译检查..."
    $PYTHON -m compileall -q "$PLUGIN_DIR" 2>/dev/null && \
        echo -e "${GREEN}✓ 编译检查通过${NC}" || \
        echo -e "${YELLOW}⚠ 编译检查有警告（可能需要重启 ComfyUI）${NC}"
    
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}安装完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo "下一步："
    echo "1. 重启 ComfyUI"
    echo "2. 下载 Qwen 模型到 models/LLM/GGUF/"
    echo "3. 在节点中选择 director_backend = qwen3.8"
    echo
    echo "12GB 显存推荐配置："
    echo "  director_backend = qwen3.8"
    echo "  director_mtp = true"
    echo "  director_reasoning_effort = medium"
    echo "  chunk_frames = 56"
    echo "  video_continuation = 22"
    echo
}

main "$@"
