#!/usr/bin/env bash
# Fix3: regenerate change (growth arrays) + journal figures + attention + rebuild paper.
cd "/c/Users/vivek/OneDrive/Desktop/CVVP paper 2" || exit 1
L=results/logs
run () { echo "STAGE $1 START"; bash -c "$2" > "$L/$1.log" 2>&1; echo "STAGE $1 END rc=$?"; }

run change "python code/eval/change_detection.py --baseline_year 2018"
run figures_journal "python code/eval/figures_journal.py"
run attention "python code/eval/attention_maps.py"
run paper "python code/paper/build_paper.py"
echo "PIPELINE_DONE"
