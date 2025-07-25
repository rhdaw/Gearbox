if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("flowAI")

require(flowAI)
library(future.apply)

# Set up parallel processing (change workers as needed)
plan(multisession, workers = 24)

# Get FCS file list
fcs_dir <- "/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/original_fcs_files"
fcs_files <- list.files(fcs_dir, pattern = "\\.fcs$", full.names = TRUE)

# Define the processing function
process_fcs_file <- function(fcs_file) {
  tryCatch({
    cat("✅ Processing:", fcs_file, "\n")
    flow_auto_qc(fcs_file)
    return(NULL)
  }, error = function(e) {
    cat("❌ Skipping:", fcs_file, "| Error:", e$message, "\n")
    return(fcs_file)  # Return the failed file path
  })
}

# Run in parallel
skipped_files <- future_lapply(fcs_files, process_fcs_file)

# Report skipped files
skipped_files <- unlist(skipped_files)
skipped_files <- skipped_files[!is.na(skipped_files)]
if (length(skipped_files) > 0) {
  cat("⚠️ Skipped files:\n")
  print(skipped_files)
}