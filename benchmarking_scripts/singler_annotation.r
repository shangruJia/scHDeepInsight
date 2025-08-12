# If using integrated data (e.g., after Seurat integration)
library(SingleR)
library(Seurat)

# Load data
query <- readRDS("./path/to/query.rds")
ref <- readRDS("./path/to/reference_atlas.rds")

# Run SingleR
pred.singler <- SingleR(test = expr,
                        ref = ref,
                        labels = ref$cell_type, 
                        de.method = "wilcox",
                        BPPARAM = BiocParallel::MulticoreParam(4))  # parallel processing

# Add results
query$singler_main <- pred.singler$labels
query$singler_pruned <- pred.singler$pruned.labels  # NA for low confidence

# Save results
saveRDS(query, "./path/to/singler_annotated_seurat.rds")

# Export predictions with metadata
pred_df <- data.frame(
  cell_id = colnames(query),
  singler_label = pred.singler$labels,
  max_score = apply(pred.singler$scores, 1, max),
  delta_score = pred.singler$delta.next  # difference to next best match
)
write.csv(pred_df, "./path/to/singler_results.csv", row.names = FALSE)