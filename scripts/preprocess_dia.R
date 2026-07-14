#!/usr/bin/env Rscript

# Build the two-channel missingness matrix used by the DIA model.

suppressPackageStartupMessages(library(readxl))

cli_args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% cli_args) {
  cat(
    "Usage: Rscript scripts/preprocess_dia.R ",
    "[--input peptide_quantities.tsv] [--labels sample_quality_labels.xlsx] ",
    "[--output-dir data/processed] [--metadata-method M3] [--instrument m3]\n",
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
input_file <- get_option(
  "--input",
  file.path(project_dir, "data", "raw", "dia", "peptide_quantities.tsv")
)
label_file <- get_option(
  "--labels",
  file.path(project_dir, "data", "raw", "dia", "sample_quality_labels.xlsx")
)
output_dir <- get_option("--output-dir", file.path(project_dir, "data", "processed"))
metadata_method <- get_option("--metadata-method", "M3")
instrument <- get_option("--instrument", "m3")

if (!file.exists(input_file)) {
  stop("DIA peptide table does not exist: ", input_file)
}
if (!file.exists(label_file)) {
  stop("DIA label file does not exist: ", label_file)
}

metadata <- read_excel(label_file)
required_metadata <- c("sample", "method", "quality")
if (!all(required_metadata %in% colnames(metadata))) {
  stop("DIA label file must contain sample, method, and quality columns.")
}
metadata <- metadata[metadata$method == metadata_method, , drop = FALSE]
if (nrow(metadata) == 0) {
  stop("No metadata rows were found for method ", metadata_method)
}

peptide_table <- read.delim(input_file, check.names = FALSE)
if (!("PEP.StrippedSequence" %in% colnames(peptide_table))) {
  stop("DIA peptide table is missing PEP.StrippedSequence.")
}
if (ncol(peptide_table) < 8) {
  stop("DIA peptide table does not contain expected quantity columns.")
}

# The first occurrence is retained to reproduce the published preprocessing.
peptide_table <- peptide_table[!duplicated(peptide_table$PEP.StrippedSequence), ]
peptide_names <- peptide_table$PEP.StrippedSequence
abundance <- peptide_table[, 8:ncol(peptide_table), drop = FALSE]
rownames(abundance) <- peptide_names
selected_samples <- colnames(abundance)[colnames(abundance) %in% metadata$sample]
if (length(selected_samples) == 0) {
  stop("No DIA quantity columns match the label metadata.")
}
abundance <- abundance[, selected_samples, drop = FALSE]

good_samples <- metadata$sample[tolower(metadata$quality) == "good"]
good_samples <- intersect(good_samples, colnames(abundance))
if (length(good_samples) == 0) {
  stop("No good DIA reference samples were found.")
}

abundance <- log2(abundance + 1)
abundance[!is.finite(as.matrix(abundance))] <- NA
cleaned <- t(apply(abundance, 1, function(values) {
  first_quartile <- quantile(values, 0.25, na.rm = TRUE)
  third_quartile <- quantile(values, 0.75, na.rm = TRUE)
  interval <- third_quartile - first_quartile
  lower <- first_quartile - interval
  upper <- third_quartile + interval
  values[values < lower | values > upper] <- NA
  values
}))
cleaned <- as.data.frame(cleaned, check.names = FALSE)

retained <- apply(cleaned, 1, function(values) {
  good_values <- values[names(values) %in% good_samples]
  mean(!is.na(good_values)) > 0.90 && sum(good_values, na.rm = TRUE) != 0
})
if (!any(retained)) {
  stop("No DIA peptide features were retained.")
}

retained_names <- rownames(cleaned)[retained]
clean_missing <- is.na(cleaned[retained, , drop = FALSE]) * 1L
raw_missing <- is.na(abundance[retained, , drop = FALSE]) * 1L
features <- cbind(t(clean_missing), t(raw_missing))
colnames(features) <- c(
  paste0("clean_missing__", retained_names),
  paste0("raw_missing__", retained_names)
)

instrument_dir <- file.path(output_dir, instrument)
dir.create(instrument_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(
  features,
  file.path(instrument_dir, "missingness_features.csv"),
  quote = TRUE
)

quality_map <- setNames(tolower(metadata$quality), metadata$sample)
labels <- data.frame(
  sample_id = colnames(abundance),
  instrument = instrument,
  quantification = "DIA",
  quality_label = ifelse(
    unname(quality_map[colnames(abundance)]) == "good",
    "good",
    "bad"
  )
)
if (any(is.na(labels$quality_label))) {
  stop("At least one DIA sample is missing a quality label.")
}
write.csv(
  labels,
  file.path(instrument_dir, "sample_labels.csv"),
  row.names = FALSE,
  quote = TRUE
)
message(
  instrument,
  ": samples=", nrow(features),
  ", peptides=", length(retained_names),
  ", input_features=", ncol(features)
)
