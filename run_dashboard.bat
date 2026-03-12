@echo off
chcp 65001 >nul
title BPU Experiment Dashboard
cd /d "%~dp0"
python -X utf8 src\tui\dashboard.py
pause
