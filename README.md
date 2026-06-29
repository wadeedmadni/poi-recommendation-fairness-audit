# Socioeconomic Fairness in Next-POI Recommendation — Code & Reproducibility

Code and analysis pipeline for the paper **"Fragile by Protocol: Socioeconomic Fairness Gaps in Next-POI Recommendation Depend on How — and Where — You Measure Them"** (M. W. Madni, N. Irfan).

We audit whether next-POI recommenders serve users from lower-income neighbourhoods worse, and show the answer is **contingent on evaluation protocol and dataset**: an apparent, statistically significant gap in New York City (Foursquare) changes sign with the model paradigm, is hidden by under-powered evaluation, and does not replicate in Austin (Gowalla).

## Repository structure

```
code/        analysis scripts (run in order; see below)
figures/     generated figures (fig1–fig5)
results/     text outputs of each analysis step
data/        (not included — see "Getting the data")
```

## Getting the data (public; not redistributed here)

1. **Foursquare NYC check-ins** (Yang et al.): https://sites.google.com/site/yangdingqi/home/foursquare-dataset → use `dataset_TSMC2014_NYC.txt`.
2. **Gowalla check-ins** (SNAP): https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz (we filter to the Austin bounding box).
3. **U.S. Census ACS B19013** (median household income by tract): https://data.census.gov — table B19013, 5-year, census tracts for the five NYC counties and for Travis County, TX.
4. **TIGER/Line tract shapefiles**: `tl_2022_36_tract` (New York) and `tl_2022_48_tract` (Texas) from https://www2.census.gov/geo/tiger/TIGER2022/TRACT/.

Place these under `data/` as referenced at the top of each script.

## Pipeline (run order)

| Script | Purpose |
|---|---|
| `01_spatial_join.py` | Map check-ins → census tract → income (NYC) |
| `02_user_home_ses.py` | Infer each user's home tract; assign low/mid/high SES |
| `03_baseline_audit.py` | Per-group accuracy for popularity / Markov / user-history |
| `04_mobility_diversity.py` | Mechanism: mobility regularity by SES (+ permutation test) |
| `05_significance.py` | Significance of the single-LOO gaps (shows under-power) |
| `06_eval_multitarget.py` | Higher-power multi-target evaluation |
| `07_mitigation.py` | Personalization-blend mitigation + accuracy–fairness trade-off |
| `09_provider_exposure.py` | Provider-side exposure vs demand by venue income |
| `10_coldstart.py` | Cold-start analysis (leave-users-out) |
| `11_harness.py` | Unified factorial harness (run per split: PU / LUO) |
| `12_matrix.py` | Protocol-sensitivity matrix + heatmap (NYC) |
| `13_fpmc.py` | FPMC learned model (BPR-SGD, NumPy) |
| `14_austin_prep.py` / `15_harness_generic.py` / `16_compare_cities.py` / `17_fig_compare.py` | Austin replication + NYC-vs-Austin comparison |
| `08_figures.py` | Paper figures |

## Requirements

Python 3.10+, with `pandas`, `numpy`, `geopandas`, `matplotlib`. Install:

```
pip install pandas numpy geopandas matplotlib
```

## Reproducibility notes

Random seeds are fixed; significance uses 4,000–5,000-sample permutation tests and 2,000-sample bootstrap CIs, all at the **user level** (one score per user) to avoid pseudoreplication. Results may differ negligibly across machines due to floating point.

## Citation

If you use this code, please cite the paper (preprint link to be added). 

## License

Code released under the MIT License. Datasets remain under their original licenses (Foursquare/Yang et al.; Gowalla/SNAP; U.S. Census public domain).
