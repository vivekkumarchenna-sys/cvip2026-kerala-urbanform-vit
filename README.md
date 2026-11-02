# Urban-Form Mapping of Kerala Cities with Vision Transformers

Reproducible code for vision-transformer classification of land cover in three
Kerala cities (Kochi, Thiruvananthapuram, Kozhikode) from open multispectral
Sentinel-2 imagery, accepted at CVIP 2026. Composites from December 2020 to
April 2021 are weakly labelled with ESA WorldCover 2021, so every score measures
agreement with that product, not field accuracy.

See **[DATA-ACCESS.md](DATA-ACCESS.md)** for the (open) data sources.

## Results

Square-root class weights, with
in-city scores pooled over the five test folds (7,772 patches).

| Experiment | Result |
|---|---|
| EuroSAT benchmark (one run per model) | ResNet-50 98.67%, EfficientNet-B0 98.02%, ViT-S 97.78% accuracy |
| Kerala in-city, five-fold spatial CV | ViT-S 97.5% accuracy, 89.4% macro-F1; SegFormer-B0 97.4%, 90.4%; ResNet-50 96.8%, 89.5%; EfficientNet-B0 94.6%, 80.3% |
| Ten bands against RGB | ViT-S 96.3%, 89.1% on RGB; ResNet-50 95.5%, 88.3% on RGB |
| Leave-one-city-out, mean | SegFormer-B0 96.6%, 91.2%; ViT-S 96.5%, 84.3%; ResNet-50 94.7%, 82.4%; EfficientNet-B0 90.2%, 78.2% |
| Inverse-frequency weights, Kochi held out | ViT-S accuracy falls from 95.7% to 46.4%, ResNet-50 from 91.4% to 27.9% |
| City maps against WorldCover | 74.2% of pixels agree in Kochi, 77.6% in Thiruvananthapuram, 89.0% in Kozhikode |
| Built-up change, 2018 to 2024 | Not resolved by classifying each year independently |

## Repository layout

```
code/
  data/       Sentinel-2 (STAC) + WorldCover download, patchify, dataset build
  models/     Model definitions (ViT / CNN, multi-band input)
  training/   Training loops (AMP, 8 GB-friendly)
  eval/       Benchmarks, maps, indicators, change detection, attention, figures
  paper/      tables_r2.py: pooled scores, Wilson intervals, McNemar tests
data/         Downloaded data (gitignored: raw/ processed/ kerala/)
results/      metrics/r2/: every per-fold, per-seed result behind the paper
```

## Environment

- Python 3.12, NVIDIA RTX 5060 (8 GB), CUDA 12.x
- torch 2.11+cu128, timm, transformers, albumentations, rasterio, pystac-client,
  odc-stac, geopandas, rioxarray, huggingface `datasets`

## Reproduce

```bash
python code/data/download_kerala_s2.py --year 2021      # composites matched to the labels
python code/data/download_kerala_s2.py --year 2024      # labelling sensitivity, change analysis
python code/data/download_kerala_s2.py --year 2018      # change analysis
python code/data/build_worldcover_labels.py             # ESA WorldCover labels
python code/data/build_patches_year.py --years 2021 2024
python code/data/wetland_mangrove_stats.py              # wetland and mangrove cover by city
python code/data/download_eurosat.py                    # EuroSAT cache
python code/training/train_eurosat.py                   # E1 benchmark
bash code/run_review_kerala.sh                          # inverse-weight runs (sensitivity rows)
bash code/run_review_kerala2.sh                         # square-root weights (primary), maps
python code/paper/tables_r2.py                          # tables and number macros
python code/eval/figures_r2.py                          # figures
```

Finished runs are skipped, so a stopped script can be run again. `code/run_review_kerala3.sh`
reruns the steps a GPU fault interrupted. The older scripts (`train_kerala.py`,
`run_kerala_pipeline.sh`) belong to an earlier version and do not produce the numbers
above.

## Data licensing / attribution

- Sentinel-2, Copernicus / ESA (open, free).
- ESA WorldCover 2021 v200, ESA (CC-BY 4.0).
- EuroSAT (Helber et al., 2019), MIT-licensed redistribution.
