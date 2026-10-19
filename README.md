# OrganoResponse-ML

Machine learning and linear discriminant analysis for studying the antitumor
response of organic nanomaterials. The workflow combines publication-grouped
model evaluation with exploratory visualization of material descriptors.

| Script | Purpose |
| --- | --- |
| `response_models.py` | Model comparison with repeated nested publication-grouped cross-validation |
| `material_projection.py` | Exploratory linear discriminant analysis (LDA) and projection plots |

## Installation

Requires Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
```

Keep both Python scripts in the same directory: the LDA script imports shared
data-processing functions from `response_models.py`.

## Input data

Provide a CSV with one row per formulation. Place your dataset in `data/` and
run commands from the repository root. Research data are not included.

| Column | Description |
| --- | --- |
| `paper_id` | Source publication ID; required for grouped machine learning |
| `binary_labels` | Prespecified binary endpoint, encoded as 0 and 1 |
| `multiclass_labels` | Prespecified class labels for the default LDA analysis |
| `Tumor_Vol_Reduction_(%)` | Optional continuous outcome for coloring LDA plots |
| Predictor columns | Curated material, treatment and experimental descriptors |

Publication IDs and labels must be complete for grouped machine learning.
Missing predictor values are imputed within each training fold. Numeric
predictors are standardized; text predictors are one-hot encoded. Numeric
category codes retain their supplied encoding.

Known ID, label and outcome columns are excluded from the predictors. Review
the remaining columns for additional identifiers or outcome proxies. Use
`--features feature_a feature_b` to select predictors explicitly, or `--drop`
to exclude additional columns. Record the endpoint definition and use consistent
feature encodings and units.

## Machine learning

```bash
python response_models.py --data data/study.csv --out results/models
```

The default analysis compares random forest, support vector machine, nearest
neighbors, decision tree, Gaussian naive Bayes and AdaBoost. It uses five repeats
of five-fold outer evaluation with five-fold inner hyperparameter tuning.
All formulations from a publication remain together in both validation layers.
Preprocessing and tuning use only the corresponding training data.

For an initial run with fewer models and folds:

```bash
python response_models.py --data data/study.csv --out results/check --models forest bayes --folds 3 --inner-folds 2 --repeats 1
```

Available model names: `forest`, `svm`, `neighbors`, `tree`, `bayes`, `boost`,
`mlp` and `xgboost`. XGBoost requires a separate installation of `xgboost`
and its platform runtime. Run either script with `--help` for all options.

Outputs include:

- `fold_metrics.csv`: outer-fold ROC-AUC, average precision, balanced accuracy,
  F1, Matthews correlation coefficient and Cohen's kappa.
- `model_summary.csv`: mean and standard deviation across outer test folds.
- `heldout_predictions.csv`: predictions with row, publication, repeat and fold IDs.
- `selected_model.joblib`: the selected algorithm retuned and fitted on all
  development data, including its preprocessing pipeline.
- `run_info.json`: settings, predictor names, selected parameters and data checksum.

The classification threshold is fixed at 0.5. Fold standard deviations describe
split variability rather than independent-sample confidence intervals. Selecting
an algorithm using the outer scores can introduce selection optimism; assess
the selected system on independent publications before making final performance claims.

## Linear discriminant analysis

```bash
python material_projection.py --data data/study.csv --label multiclass_labels --out results/lda
```

LDA uses the supplied class labels without generating new endpoint thresholds.
At least three classes are needed for a two-dimensional projection; binary
labels provide one discriminant axis. To use binary labels, pass
`--label binary_labels`.

To color the projection by a continuous outcome:

```bash
python material_projection.py --data data/study.csv --color-by "Tumor_Vol_Reduction_(%)" --out results/lda_colored
```

Outputs are `discriminant_coordinates.csv` and projection plots in PNG and SVG
formats. LDA is fitted to the full input dataset for supervised exploration;
visual separation is not evidence of held-out predictive performance.

Both scripts require a new output directory to protect previous results.

## Reference

Methodological reference: Mendes, B. B., Zhang, Z., et al.
*A large-scale machine learning analysis of inorganic nanoparticles in preclinical
cancer research.* Nature Nanotechnology 19, 867–878 (2024).
[Paper](https://doi.org/10.1038/s41565-024-01673-7) ·
[Reference repository](https://github.com/RekerLab/NanoAnalysis).
This implementation reorganizes organic-material analysis workflows around
publication-grouped validation and exploratory LDA.
