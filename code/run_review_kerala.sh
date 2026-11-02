#!/usr/bin/env bash
# Kerala runs, most important first. Each step is resumable.
T="python -u code/training/train_kerala_r2.py"
$T --mode cv --models vit_s resnet50 effb0 segformer_b0
$T --mode full
$T --mode loco --models vit_s resnet50 effb0 segformer_b0
$T --mode cv --year 2024 --models vit_s resnet50 --no_rgb
$T --mode loco --models vit_s resnet50 --held kochi --weight sqrt
$T --mode cv --models vit_s --no_rgb --alt_labels
echo KERALA_DONE
