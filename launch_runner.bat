@echo off
title BPU Experiment Runner
cd /d "C:\Users\tejas\OneDrive\Desktop\connectome_bpu"
echo Starting BPU experiments at %date% %time%
"C:\Users\tejas\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe" -X utf8 -u run_experiments.py
echo.
echo Runner exited at %date% %time% with code %ERRORLEVEL%
pause
