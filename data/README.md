# Data

Large input and processed files are distributed through the companion data/model archive rather than Git.

Archive URL:

<https://doi.org/10.5281/zenodo.21337993>

## DDA inputs

The DDA preprocessing script expects pairs of files in one directory:

```text
<instrument>_peptide_auc.csv
<instrument>_good_samples.csv
```

The abundance CSV contains peptide rows and sample columns. The good-sample CSV contains one sample identifier per row.

## DIA inputs

The DIA preprocessing script expects:

```text
peptide_quantities.tsv
sample_quality_labels.xlsx
```

The peptide table must contain `PEP.StrippedSequence` and sample-level quantity columns. The metadata file must contain `sample`, `method`, and `quality` columns.

## Standardized processed data

Both workflows produce:

```text
data/processed/<instrument>/missingness_features.csv
data/processed/<instrument>/sample_labels.csv
```

The feature matrix contains one row per sample and two ordered feature blocks named `clean_missing__<peptide>` and `raw_missing__<peptide>`.

The combined labels used in the reported analysis are retained in `data/sample_labels.csv` because they are small and required to interpret the published results.
