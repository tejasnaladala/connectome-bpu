@echo off
cd /d "C:\Users\tejas\OneDrive\Desktop\connectome_bpu"
echo === STAGING ALL CHANGES ===
git add results/all_results.csv
git add scripts/
git add docs/
git add telescope/
git add paper/
git add src/
git add check_progress.bat
git add run_analysis.bat
git add quick_check.bat
git add deep_check.bat
git add save_checkpoint.bat
git add checkpoint.bat
git status
echo === COMMITTING ===
git commit -m "checkpoint: 694/900 (77.1%%) p=0.006 win_rate=72.2%% - all metrics improving"
echo === PUSHING ===
git push
echo === DONE ===
