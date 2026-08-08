#!/usr/bin/env bash
# Fix2: clean 2018 baseline + corrected maps/indicators + change + figures + paper.
cd "/c/Users/vivek/OneDrive/Desktop/CVVP paper 2" || exit 1
rm -f data/kerala/*_2017.tif data/kerala/*_2017.json
L=results/logs
run () { echo "STAGE $1 START"; bash -c "$2" > "$L/$1.log" 2>&1; echo "STAGE $1 END rc=$?"; }

run download_2018 "python code/data/download_kerala_s2.py --year 2018 --max_scenes 15"
run maps "python code/eval/make_maps_indicators.py"
run change "python code/eval/change_detection.py --baseline_year 2018"
run figures "python code/eval/make_figures.py"
run paper "python code/paper/build_paper.py"
echo "PIPELINE_DONE"
