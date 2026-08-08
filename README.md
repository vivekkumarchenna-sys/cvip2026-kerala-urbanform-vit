# Urban-Form Mapping of Kerala Cities with Vision Transformers

Reproducible code for vision-transformer classification of urban form and land
cover in three Kerala cities (Kochi, Thiruvananthapuram, Kozhikode) from open,
fully multispectral Sentinel-2 imagery, with planning indicators and a 2018-2024
urban-growth analysis.

See **[DATA_ACCESS.md](DATA_ACCESS.md)** for the (open) data sources.

## Headline results

| Experiment | Result |
|---|---|
| EuroSAT-MSI benchmark | ResNet-50 98.7%, ViT-S 97.8%, EfficientNet-B0 98.0% (near-saturated; all within ~1 pt) |
| **Kerala in-city (ViT-S, 10-band)** | **97.96% acc, 0.971 macro-F1**; best backbone (ResNet 0.851, EfficientNet 0.710 macro-F1) |
| Multispectral vs RGB (ViT, Kerala) | +1.8 pt acc; Cropland/Grass F1 0.82 -> 0.94 |
| Cross-city (leave-one-city-out) | ViT-S best mean macro-F1 0.712; Kochi (water-dominated) hardest to transfer to |
| Planning indicators (2024) | Kochi built 29% / green 27% / water 44%; Thiruvananthapuram 27 / 60 / 13; Kozhikode 4 / 80 / 16 |
| Built-up growth 2018-2024 | Kozhikode +20%, Thiruvananthapuram +5%, Kochi about stable |

## Repository layout

```
code/
  data/       Sentinel-2 (STAC) + WorldCover download, patchify, dataset build
  models/     Model definitions (ViT / CNN, multi-band input)
  training/   Training loops (AMP, 8 GB-friendly)
  eval/       Benchmarks, maps, indicators, change detection, attention, figures
data/         Downloaded data (gitignored: raw/ processed/ kerala/)
references/   BibTeX + reference-verification report
results/      figures/ + metrics/  (checkpoints/ logs/ gitignored)
```

## Environment

- Python 3.12, NVIDIA RTX 5060 (8 GB), CUDA 12.x
- torch 2.11+cu128, timm, transformers, albumentations, rasterio, pystac-client,
  odc-stac, geopandas, rioxarray, huggingface `datasets`

## Reproduce

```bash
python code/data/download_kerala_s2.py --year 2024      # Sentinel-2 composites, 3 cities
python code/data/download_kerala_s2.py --year 2018      # baseline for change detection
python code/data/build_worldcover_labels.py             # ESA WorldCover labels
python code/data/build_patches.py                       # patchify -> labelled dataset
python code/data/download_eurosat.py                    # EuroSAT-MSI benchmark cache
python code/training/train_eurosat.py                   # E1 benchmark
python code/training/train_kerala.py                    # E2/E3 in-city + leave-one-city-out
python code/eval/make_maps_indicators.py                # E4 maps + indicators
python code/eval/change_detection.py --baseline_year 2018
python code/eval/figures_journal.py                     # all figures
```

## Data licensing / attribution

- Sentinel-2, Copernicus / ESA (open, free).
- ESA WorldCover 2021 v200, ESA (CC-BY 4.0).
- EuroSAT (Helber et al., 2019), MIT-licensed redistribution.
