#!/usr/bin/env bash
# Square-root class weights everywhere: the inverse weights collapse when a class is nearly absent.
T="python -u code/training/train_kerala_r2.py"
$T --mode cv --models vit_s resnet50 effb0 segformer_b0 --weight sqrt
$T --mode full --weight sqrt
python -u code/eval/maps_r2.py
$T --mode cv --year 2024 --models vit_s resnet50 --no_rgb --weight sqrt
$T --mode cv --models vit_s --no_rgb --alt_labels --weight sqrt
echo KERALA2_DONE
