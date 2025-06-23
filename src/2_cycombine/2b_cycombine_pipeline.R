# Install cyCombine package from GitHub
devtools::install_github("biosurf/cyCombine")

library(cyCombine)
library(tidyverse)
print('Everything loaded in')

# Directory with FCS files
data_dir <- "/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai"

# Extract markers from panel
panel_file <- paste0("/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv") # Can also be .xlsx
metadata_file <- paste0("/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/stroke_impact_metadata_all_batches.csv") # Can also be .xlsx
print('Meta data loaded')

# Extract markers of interest
markers <- read.csv(panel_file) %>% 
  filter(Type != "None") %>% 
  pull(Antigen)

# Prepare a tibble from directory of FCS files
uncorrected <- prepare_data(
                      data_dir = data_dir,
                      metadata = metadata_file,
                      filename_col = "Filename",
                      batch_ids = "batch",
                      condition = "condition",
                      sample_ids = "Patient_id",
                      markers = markers,
                      down_sample = FALSE,
                      .keep = FALSE,
                      clean_colnames = FALSE,
                      panel = panel_file,
                      panel_channel = "Channel",
                      panel_antigen = "Antigen",
                      transform = TRUE)

print('Uncorrected Tibble made')

# Store result in dir
saveRDS(uncorrected, file = file.path(data_dir, "uncorrected.RDS"))
print('Uncorrected RDS Made')

seed_num <- sample(1:100, 1) #Setting a random seed number

corrected <- batch_correct(
  df = uncorrected,
  covar = "condition",
  markers = markers,
  norm_method = "scale", # "rank" is recommended when combining data with heavy batch effects
  rlen = 10, # Higher values are recommended if 10 does not appear to perform well
  seed = seed_num # Recommended to use your own random seed
)

saveRDS(corrected, file.path(data_dir, "corrected.RDS"))

print('Corrected Tibble made')

# Re-run clustering on corrected data
labels <- corrected %>% 
  create_som(markers = markers,
             rlen = 10)
uncorrected$label <- corrected$label <- labels

# If it errors out here it is usually because there are NAs within one of the columns. FlowSOM won't be generated if NAs are in the markers, but if there are NAs in anything else find them with: 
#metadata_csv_look<- read.csv('/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/stroke_impact_metadata_all_batches.csv')
#sapply(uncorrected, function(column) any(is.na(column)))

# Evaluate EMD
emd <- evaluate_emd(uncorrected, corrected, cell_col = "label")

# Reduction
emd$reduction

# Violin plot
emd$violin

# Scatter plot
emd$scatter

# Evaluate MAD
mad <- evaluate_mad(uncorrected, corrected, cell_col = "label")

# Score
mad$score


## Saving the batch corrected .fcs files <-untested.

# --- Prerequisites (ensure these are loaded/defined) ---
library(flowCore) # For write.FCS and flowFrame
library(tidyverse) # For dplyr functions if used for subsetting

# IMPORTANT: Ensure this 'markers' vector exactly matches the column names
# in 'corrected' that represent the expression data you want in the FCS file.
# From your list, it seems these are the ones from "CD45" to "HLADR".

# Your data_dir (as defined in your script)
# data_dir <- "/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai"

# --- FCS Writing Logic ---

# Output directory for corrected FCS files
corrected_fcs_dir <- file.path(dirname(data_dir), "corrected_fcs_files_from_R") # Place it alongside post_flowai
if (!dir.exists(corrected_fcs_dir)) {
  dir.create(corrected_fcs_dir, recursive = TRUE)
  print(paste("Created directory:", corrected_fcs_dir))
} else {
  print(paste("Output directory already exists:", corrected_fcs_dir))
}

# Determine the column to split by.
# 'sample' seems like a good candidate if it uniquely identifies original FCS files.
# If your original FCS files were named like "sample1.fcs", "sample2.fcs",
# and 'sample' column contains "sample1", "sample2", etc.
# Or, if you have an original filename column you brought through, use that.
# Let's assume 'sample' is the correct one.
split_col_name <- "sample" # <<<< USER: VERIFY THIS IS THE CORRECT COLUMN TO SPLIT BY

if (!split_col_name %in% colnames(corrected)) {
  stop(paste("The specified split column '", split_col_name, "' does not exist in the 'corrected' tibble.",
             "Common choices are 'sample', 'Filename', or 'Patient_id'. Please verify."))
}

# Get unique sample identifiers from the chosen column
sample_identifiers <- unique(corrected[[split_col_name]])
print(paste("Found", length(sample_identifiers), "unique samples to process based on column:", split_col_name))

# --- Loop through each sample and write an FCS file ---
for (current_sample_id in sample_identifiers) {
  
  print(paste("Processing sample:", current_sample_id))
  
  # Subset data for the current sample
  # Using base R for clarity here, dplyr::filter also works
  sample_data_df <- corrected[corrected[[split_col_name]] == current_sample_id, ]
  
  # Extract the expression matrix for the specified markers
  # Ensure all markers are present in sample_data_df (they should be, as they are in 'corrected')
  if (!all(markers %in% colnames(sample_data_df))) {
    missing_markers <- markers[!markers %in% colnames(sample_data_df)]
    warning(paste("Sample", current_sample_id, "is missing markers:", paste(missing_markers, collapse=", "), "- Skipping FCS write for this sample."))
    next # Skip to the next sample
  }
  expression_matrix <- as.matrix(sample_data_df[, markers])
  
  # --- Create a flowFrame ---
  # This is a crucial step. The `parameters` slot of a flowFrame needs to be
  # set up correctly, especially the 'name' ($PnN) and 'desc' ($PnS) for each channel.
  # Ideally, you'd get this structure from one of your original FCS files.
  
  # Option 1: Basic flowFrame (may not be ideal for all downstream software)
  # ff <- flowFrame(exprs = expression_matrix)
  
  # Option 2: More complete flowFrame with parameter metadata
  # We need to create the `parameters` AnnotatedDataFrame
  # The channel names ($PnN) might be different from your marker antigen names ($PnS)
  # For now, let's assume your 'markers' vector contains the names you want as $PnS (desc)
  # And we'll generate generic $PnN names.
  
  num_channels <- ncol(expression_matrix)
  pd <- data.frame(
    name = colnames(expression_matrix), # These will be used as $PnS (descriptions/antigens)
    desc = colnames(expression_matrix), # Or you can use your 'markers' vector if names were cleaned
    range = apply(expression_matrix, 2, function(x) diff(range(x, na.rm = TRUE))), # Placeholder
    minRange = apply(expression_matrix, 2, min, na.rm = TRUE),                     # Placeholder
    maxRange = apply(expression_matrix, 2, max, na.rm = TRUE)                      # Placeholder
  )
  # FCS standard often uses $PnN for channel name and $PnS for stain/antigen
  # If your 'markers' are antigen names, they should go into 'desc'.
  # The 'name' column in 'pd' should be the channel name (e.g., "FITC-A", "PE-Texas Red-A")
  # If you don't have original channel names, you can use the marker names for both,
  # or try to map them if you have a panel file.
  
  # Let's assume for now that colnames(expression_matrix) are the desired 'desc' (stain/antigen)
  # And we will use these as 'name' as well if channel names are not readily available.
  # This might need adjustment based on how `prepare_data` handled channel vs. antigen names.
  
  # Correctly format for AnnotatedDataFrame: rownames are $PnN, $PnS, etc.
  # We need to ensure the 'name' and 'desc' in the parameters match what downstream software expects.
  # The `name` field in the `parameters` data.frame should be the channel identifier (e.g., "FSC-A", "FL1-A").
  # The `desc` field should be the stain/marker name (e.g., "CD3", "CD45RA").
  
  # Let's assume your 'markers' vector contains the "desc" (antigen) names.
  # We need to find the corresponding "name" (channel) names if they were stored
  # or if your `prepare_data` used panel_channel and panel_antigen.
  # If `clean_colnames = FALSE` in `prepare_data`, `colnames(expression_matrix)` might be channel names.
  # If `clean_colnames = TRUE` (default), they are likely cleaned antigen names.
  
  # For this example, let's assume colnames(expression_matrix) are the antigen names (desc)
  # and we'll use them as channel names (name) as a fallback.
  # A more robust solution would be to use your panel_file to map antigens back to channel names.
  
  params_df <- data.frame(
    row.names = sprintf("$P%s", 1:num_channels), # Standard way to name rows
    name = colnames(expression_matrix),          # Placeholder: Use actual channel names if available
    desc = colnames(expression_matrix),          # These are your marker/antigen names
    range = apply(expression_matrix, 2, function(x) as.numeric(diff(range(x, na.rm=TRUE)))),
    minRange = apply(expression_matrix, 2, function(x) as.numeric(min(x, na.rm=TRUE))),
    maxRange = apply(expression_matrix, 2, function(x) as.numeric(max(x, na.rm=TRUE)))
  )
  
  # The actual column names in the expression_matrix must match params_df$name
  # If colnames(expression_matrix) are already the desired channel names, this is simpler.
  # If they are antigen names, and you want different channel names, you'd need to rename
  # the columns of expression_matrix to match params_df$name.
  
  # Let's assume colnames(expression_matrix) are what we want for the FCS file channels for now.
  # This is a common simplification if original channel names aren't easily propagated.
  
  parameters_ad <- Biobase::AnnotatedDataFrame(data = params_df)
  
  # Ensure expression_matrix column names match the 'name' field in parameters_ad if they differ
  # For this example, we assume they are already consistent or that colnames(expression_matrix) are the desired final channel names.
  
  ff <- new("flowFrame",
            exprs = expression_matrix,
            parameters = parameters_ad,
            description = list(GUID = paste0(current_sample_id,"_corrected.fcs"), # Example keyword
                               FIL = paste0(current_sample_id,"_corrected.fcs"),
                               `$CYT` = "CyCombine via R",
                               `$TOT` = nrow(expression_matrix),
                               `$PAR` = ncol(expression_matrix)
                               # Add other keywords as needed
            ))
  
  
  # --- Define output filename ---
  # Sanitize current_sample_id for use in filename (remove special chars, spaces)
  safe_sample_id_for_filename <- gsub("[^A-Za-z0-9_.-]", "_", as.character(current_sample_id))
  output_filename <- file.path(corrected_fcs_dir, paste0(safe_sample_id_for_filename, "_corrected.fcs"))
  
  # --- Write FCS file ---
  tryCatch({
    write.FCS(ff, filename = output_filename)
    print(paste("Successfully written:", output_filename))
  }, error = function(e) {
    print(paste("ERROR writing FCS for sample", current_sample_id, ":", e$message))
    # You might want to print more details from 'e' or the flowFrame 'ff' for debugging
    # print(str(ff))
    # print(head(Biobase::pData(Biobase::parameters(ff))))
  })
}

print("--- FCS writing process finished. ---")
