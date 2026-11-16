@echo off
chcp 65001 >nul
title AIGC 检测 Demo
cd /d %~dp0
echo 启动检测服务...
start "" http://127.0.0.1:8000/
python src\serve.py
pause
