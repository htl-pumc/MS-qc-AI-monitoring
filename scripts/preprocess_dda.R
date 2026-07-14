#!/usr/bin/env Rscript

# Build the two-channel missingness matrices used by the DDA models.

cli_args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% cli_args) {
  cat(
    "Usage: Rscript scripts/preprocess_dda.R ",
    "[--input-dir data/raw/dda] [--output-dir data/processed]\n",
    sep = ""
  )
  quit(status = 0)
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  script_arg <- args[grepl("^--file=", args)]
  if (length(script_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", script_arg[1]))))
  }
  getwd()
}

get_option <- function(name, default = NULL) {
  position <- match(name, cli_args)
  if (is.na(position)) {
    return(default)
  }
  if (position == length(cli_args)) {
    stop("Missing value for ", name)
  }
  cli_args[position + 1]
}

script_dir <- get_script_dir()
project_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
input_dir <- get_option("--input-dir", file.path(project_dir, "data", "raw", "dda"))
output_dir <- get_option("--output-dir", file.path(project_dir, "data", "processed"))

if (!dir.exists(input_dir)) {
  stop("DDA input directory does not exist: ", input_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

mark_row_outliers <- function(matrix, multiplier = 1.5) {
  cleaned <- t(apply(matrix, 1, function(values) {
    first_quartile <- quantile(values, 0.25, na.rm = TRUE)
    third_quartile <- quantile(values, 0.75, na.rm = TRUE)
    interval <- third_quartile - first_quartile
    lower <- first_quartile - multiplier * interval
    upper <- third_quartile + multiplier * interval
    values[values < lower | values > upper] <- NA
    values
  }))
  as.data.frame(cleaned, check.names = FALSE)
}

write_missingness_data <- function(instrument, abundance_file, good_file) {
  abundance <- read.csv(
    abundance_file,
    row.names = 1,
    check.names = FALSE
  )
  abundance <- log2(abundance + 1)
  colnames(abundance) <- gsub("_1_AUC", "", colnames(abundance), fixed = TRUE)

  good_samples <- read.csv(good_file, check.names = FALSE)[[1]]
  good_samples <- gsub("_iFOT", "", good_samples, fixed = TRUE)
  unknown_good <- setdiff(good_samples, colnames(abundance))
  if (length(unknown_good) > 0) {
    stop(instrument, " has good samples that are absent from the abundance matrix.")
  }

  cleaned <- mark_row_outliers(abundance, multiplier = 1.5)
  retained <- apply(cleaned, 1, function(values) {
    good_values <- values[names(values) %in% good_samples]
    mean(!is.na(good_values)) > 0.90 && sum(good_values, na.rm = TRUE) != 0
  })
  if (!any(retained)) {
    stop("No peptide features were retained for ", instrument)
  }

  peptide_names <- rownames(cleaned)[retained]
  clean_missing <- is.na(cleaned[retained, , drop = FALSE]) * 1L
  raw_missing <- is.na(abundance[retained, , drop = FALSE]) * 1L
  features <- cbind(t(clean_missing), t(raw_missing))
  colnames(features) <- c(
    paste0("clean_missing__", peptide_names),
    paste0("raw_missing__", peptide_names)
  )

  instrument_dir <- file.path(output_dir, instrument)
  dir.create(instrument_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(
    features,
    file.path(instrument_dir, "missingness_features.csv"),
    quote = TRUE
  )

  labels <- data.frame(
    sample_id = colnames(abundance),
    instrument = instrument,
    quantification = "DDA",
    quality_label = ifelse(colnames(abundance) %in% good_samples, "good", "bad")
  )
  write.csv(
    labels,
    file.path(instrument_dir, "sample_labels.csv"),
    row.names = FALSE,
    quote = TRUE
  )
  message(
    instrument,
    ": samples=", nrow(features),
    ", peptides=", length(peptide_names),
    ", input_features=", ncol(features)
  )
}

abundance_files <- sort(list.files(
  input_dir,
  pattern = "_peptide_auc\\.csv$",
  full.names = TRUE
))
if (length(abundance_files) == 0) {
  stop("No *_peptide_auc.csv files were found in ", input_dir)
}

for (abundance_file in abundance_files) {
  instrument <- sub("_peptide_auc\\.csv$", "", basename(abundance_file))
  good_file <- file.path(input_dir, paste0(instrument, "_good_samples.csv"))
  if (!file.exists(good_file)) {
    stop("Missing good-sample file for ", instrument, ": ", good_file)
  }
  write_missingness_data(instrument, abundance_file, good_file)
}
