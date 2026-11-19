@echo off
chcp 65001 >nul
title AIGC 检测 Demo
cd /d %~dp0

echo ============================================
echo   AIGC 文本检测器 - 本地 Demo
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 找不到 python 命令
    echo 请安装 Python 3.11+ 并勾选 "Add Python to PATH"
    echo 下载 https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 只检查 numpy: 四个档位都有纯 numpy 快路径, sklearn/scipy 是可选的,
REM 以前这里检查 sklearn 会逼所有人装 187MB 用不上的依赖。
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行需要安装 numpy，请稍候...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败，请手动运行:
        echo     pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
)

echo [1/2] 正在启动检测服务（首次加载模型约 0.5 秒）...
echo.
echo   服务地址: http://127.0.0.1:8000/
echo   关闭本窗口即停止服务
echo.

REM 后台等待服务就绪后打开浏览器
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:8000/"

python src\serve.py

echo.
echo 服务已停止。
pause
