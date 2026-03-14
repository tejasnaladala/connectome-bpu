@echo off
cd /d "C:\Users\tejas\OneDrive\Desktop\connectome_bpu"
echo === SAVING CHECKPOINT ===
git add results/all_results.csv
git add docs/plans/2026-03-13-research-telescope-design.md
git add docs/research-telescope-pipeline.html
git add scripts/telescope_scanner.py
git add telescope/results/
git add telescope/digests/
git add check_progress.bat
git add run_analysis.bat
git status
echo === COMMITTING ===
git commit -m "checkpoint: 667/900 experiments (74.1%%), p=0.009, win rate 71.4%% - telescope pipeline design + interactive graph + scanner working"
echo === DONE ===
