# Bayesian Neural Network Prior Study

An end-to-end pipeline for exploring the impact of weight priors in Bayesian neural network on small regression datasets. The workflow covers dataset discovery, meta-feature extraction (via PyMFE), Metropolis–Hastings inference with swappable priors (Gaussian, Laplace, Student‑t), and post-hoc analysis tying posterior metrics back to dataset characteristics.

---

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
The pyproject file includes all dependencies.

---

## 2. Pipeline Overview

| Stage | Example Command | Outputs | Notes |
| --- | --- | --- | --- |
| Discover datasets | `python -m bnn_initializations_project.cli.discover --max 30` | `artifacts/manifest.parquet`, `artifacts/datasets/*.npz` | Pulls numeric-target regression tasks from OpenML according to `configs/datasets.yaml`. |
| Extract meta-features | `python -m bnn_initializations_project.cli.extract_meta --jobs 4` | `artifacts/meta_features.parquet` | Combines PyMFE features with custom descriptors (nonlinearity, skew/kurtosis, etc.). |
| Train + evaluate priors | `python -m bnn_initializations_project.cli.train_eval --prior isotropic_gaussian --jobs 2` (repeat for `laplace`, `student_t --nu 5`) | `artifacts/results.parquet`, optional predictive samples | Uses config in `configs/model.yaml` and `configs/run.yaml`; saves ID/OOD metrics per dataset × prior. |
| Aggregate + analyze | `python -m bnn_initializations_project.cli.aggregate_analyze` | `artifacts/analysis/*.parquet/csv/json` | Produces ΔNLL tables, meta-feature correlations, logistic summary for non-Gaussian wins. |

The pipeline is idempotent: rerun stages to append new datasets; pass `--force` to overwrite.

---

## 3. CLI Reference

### Dataset discovery
```
python -m bnn_initializations_project.cli.discover \
    --max 30 \
    --source openml \
    [--force]
```
Key filters live in `configs/datasets.yaml`:
- `min_rows`, `max_rows`, `max_features`
- `only_numeric_target`
- `source` currently only supports "openml"
- `drop_list` (OpenML IDs to skip)

### Meta-feature extraction
```
python -m bnn_initializations_project.cli.extract_meta \
    --jobs 4 \
    [--force]
```
Loads all datasets with `status="ok"` from the manifest; parallelized via joblib.

### Training & evaluation
```
python -m bnn_initializations_project.cli.train_eval \
    --prior {isotropic_gaussian|laplace|student_t} \
    [--nu 5.0] \
    --jobs 2 \
    [--force]
```
If no prior argument is given, the train_eval will run for all priors.

Important run settings (`configs/model.yaml`, `configs/run.yaml`):
- `layer_widths`, `activation`, `lik_sigma`, `init_scheme`
- `num_samples`, `num_burn_in`, `step_size`
- `id_split_ratio`, `ood_tail_fraction`, `max_ppc_samples`

### Aggregate analysis
```
python -m bnn_initializations_project.cli.aggregate_analyze
```
Outputs in `artifacts/analysis/` folder:
- `feature_metric_correlations.parquet`: correlations between meta-features and model metrics
- `meta_regression_coefficients.csv`: Coefficients for regression on NLL for all priors

---

## 4. Inspecting Results

- `artifacts/manifest.parquet`: dataset catalog with statuses/errors  
- `artifacts/meta_features.parquet`: PyMFE features per dataset  
- `artifacts/results.parquet`: metrics (`rmse`, `nll`, `cov90/95`, `iw90/95`, `accept_rate`) for ID/OOD splits (or leave `None` for no splitting)
- `artifacts/analysis/`: derived analysis products and visual summaries

Use `bnn_initializations_project.visualization` helpers for quick plots, see [example notebook](src/bnn_initializations_project/show_correlations.ipynb)

---

## 5. Package Layout

```
src/bnn_initializations_project
├── cli/                   # CLI entrypoints (discover, extract_meta, train_eval, aggregate_analyze)
├── configs/               # YAML configs (datasets, model, run)
├── dataio/                # Artifact storage helpers, manifest I/O, train/OOD splits
├── features/              # PyMFE wrapper + handcrafted meta-feature utilities
├── metrics/               # Predictive metrics + result row helpers
├── models/                # BNN model, priors, inference (Metropolis–Hastings)
└──pipelines/             # Orchestrators invoked by CLI modules
```

---

## 6. Misc

- **No datasets discovered:** Adjust filters in `configs/datasets.yaml` or increase `--max`.
- **Sampler issues (low `accept_rate`):** tweak `step_size` in `configs/run.yaml`.
- **Recomputing artifacts:** use `--force` or delete the specific parquet/csv files before rerunning.
