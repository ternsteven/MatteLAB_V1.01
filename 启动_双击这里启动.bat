
@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
title BiRefNet - 启动器
color 07


REM --- 环境变量 ---
@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
title BiRefNet 启动器

echo.
echo ================================================================
echo                 【✨ BiRefNet 图片/视频背景替换工具 ✨】
echo ================================================================
echo.
echo     感谢原作者: 郑鹏 (ZhengPeng7)  github.com/ZhengPeng7/BiRefNet
echo     代码修改与功能添加: 小T_sune
echo.
echo     ⚠ 本软件仅供学习与研究使用，严禁商用！
echo ================================================================
echo.
echo     启动时间: %date% %time%
echo     操作系统: %OS%
echo ================================================================
echo.
echo 🚀 正在启动 BiRefNet，请稍候加载模型中...
echo ================================================================
echo.


REM 项目根目录
set "ROOT=%~dp0"

REM Python 选择：优先项目自带，其次系统 Python
set "PYTHON=%ROOT%python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

REM 让 Python 能 import src/*
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"

REM 模型缓存与输出目录（Transformers v5 建议使用 HF_HOME）
set "HF_HOME=%ROOT%models_local"
set "BIRE_OUTPUT_DIR=%ROOT%preds-BiRefNet"

REM 建议的环境选项（可按需注释掉）
set "PYTHONUTF8=1"
set "PYTHONWARNINGS=ignore"
set "GRADIO_ANALYTICS_ENABLED=FALSE"
set "KMP_DUPLICATE_LIB_OK=TRUE"

cd /d "%ROOT%"

echo 启动中，请稍候...
REM 使用 main.py 的“端口就绪后自动打开浏览器”机制
"%PYTHON%" "%ROOT%main.py"

echo.
echo =====================================
echo 进程已退出（若异常请查看上方日志）
echo =====================================
pause
endlocal




