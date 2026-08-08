#!/usr/bin/env bash
# Full Kerala experiment + eval + paper build. Stage markers on stdout; detail in per-stage logs.
cd "/c/Users/vivek/OneDrive/Desktop/CVVP paper 2" || exit 1
L=results/logs
run () { echo "STAGE $1 START"; bash -c "$2" > "$L/$1.log" 2>&1; echo "STAGE $1 END rc=$?"; }

run kerala_train "python code/training/train_kerala.py --mode incity loco full --models vit_s resnet50 effb0 --epochs 30 --bs 128"
run download_2017 "python code/data/download_kerala_s2.py --year 2017 --max_scenes 15"
run maps "python code/eval/make_maps_indicators.py"
run change "python code/eval/change_detection.py --baseline_year 2017"
run attention "python code/eval/attention_maps.py"
run figures "python code/eval/make_figures.py"
run paper "python code/paper/build_paper.py"
echo "PIPELINE_DONE"
