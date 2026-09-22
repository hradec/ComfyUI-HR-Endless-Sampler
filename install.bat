@echo off
REM ============================================================
REM HR Endless Sampler (mickeylan fork) 安装脚本 - Windows
REM ============================================================
REM 适用：中文用户 / 12GB 显存用户
REM 特色：支持 Qwen3.5/3.6/3.8 导演
REM ============================================================

setlocal enabledelayedexpansion

echo ========================================
echo HR Endless Sampler (mickeylan fork)
echo 中文用户 / 低显存用户优化版
echo ========================================
echo.

REM 查找 ComfyUI 目录
set "COMFYUI_DIR="
for %%p in (
    "%USERPROFILE%\ComfyUI"
    "%USERPROFILE%\comfyui"
    "C:\ComfyUI"
    "D:\ComfyUI"
) do (
    if exist "%%p\main.py" (
        set "COMFYUI_DIR=%%p"
        goto :found_comfyui
    )
)

:found_comfyui
if "%COMFYUI_DIR%"=="" (
    echo [错误] 未找到 ComfyUI 目录！
    echo 请确保 ComfyUI 已安装
    pause
    exit /b 1
)

echo [1/4] 找到 ComfyUI: %COMFYUI_DIR%

REM 确认插件目录
set "PLUGIN_DIR=%COMFYUI_DIR%\custom_nodes\ComfyUI-MiniMax-H3-Sampler-Unlimited"
if not exist "%PLUGIN_DIR%" (
    echo [错误] 插件目录不存在！
    echo 请将本插件放到: %PLUGIN_DIR%
    pause
    exit /b 1
)

echo [2/4] 找到插件: %PLUGIN_DIR%

REM 查找 Python
set "PYTHON="
if exist "%COMFYUI_DIR%\python_embeded\python.exe" (
    set "PYTHON=%COMFYUI_DIR%\python_embeded\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON=python"
    )
)

if "%PYTHON%"=="" (
    echo [错误] 未找到 Python 解释器！
    pause
    exit /b 1
)

echo [3/4] 使用 Python: %PYTHON%

REM 安装依赖
echo.
echo [4/4] 安装依赖...
cd /d "%PLUGIN_DIR%"
%PYTHON% -m pip install -r requirements.txt

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 重启 ComfyUI
echo 2. 下载 Qwen 模型到 models\LLM\GGUF\
echo 3. 在节点中选择 director_backend = qwen3.8
echo.
echo 12GB 显存推荐配置：
echo   director_backend = qwen3.8
echo   director_mtp = true
echo   director_reasoning_effort = medium
echo   chunk_frames = 56
echo   video_continuation = 22
echo.

pause
