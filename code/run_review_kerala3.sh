#!/usr/bin/env bash
# Fill the gaps left when a GPU fault stopped the first pass; finished runs are skipped.
T="python -u code/training/train_kerala_r2.py"
$T --mode cv --models vit_s resnet50 effb0 segformer_b0 --weight sqrt
$T --mode cv --models vit_s resnet50 effb0 segformer_b0 --weight sqrt
$T --mode cv --year 2024 --models vit_s resnet50 --no_rgb --weight sqrt
$T --mode cv --models vit_s --no_rgb --alt_labels --weight sqrt
echo KERALA3_DONE
