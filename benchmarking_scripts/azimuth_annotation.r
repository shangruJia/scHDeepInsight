# Azimuth cell annotation workflow

library(Seurat)
library(Azimuth)
library(SeuratData)

# Load query data
query <- readRDS("./path/to/query.rds")

# Preprocess if not already done
query <- NormalizeData(query)
query <- FindVariableFeatures(query)
query <- ScaleData(query)
query <- RunPCA(query)
query <- RunUMAP(query, dims = 1:30)

# Run Azimuth annotation
# Option 1: PBMC reference (most common)
query <- RunAzimuth(query, reference = "pbmcref")

# Option 2: Other available references
# query <- RunAzimuth(query, reference = "lungref")      # Lung
# query <- RunAzimuth(query, reference = "kidneyref")    # Kidney
# query <- RunAzimuth(query, reference = "pancreasref")  # Pancreas
# query <- RunAzimuth(query, reference = "heartref")     # Heart
# query <- RunAzimuth(query, reference = "motorctxref")  # Motor cortex

# Extract annotations
query$azimuth_l2 <- query$predicted.celltype.l2

# Check annotation distribution
print(table(query$azimuth_l2))

# Visualize annotations
DimPlot(query, group.by = "azimuth_l2", label = TRUE, repel = TRUE)

# Filter low confidence predictions if needed
query$azimuth_l2_filtered <- ifelse(query$predicted.celltype.score > 0.75, 
                                   query$azimuth_l2, 
                                   "Low_confidence")

# Save annotated object
saveRDS(query, "./path/to/azimuth_annotated_seurat.rds")

# Export L2 predictions
azimuth_l2_results <- data.frame(
 cell_id = colnames(query),
 celltype_l2 = query$azimuth_l2,
 prediction_score = query$predicted.celltype.score
)
write.csv(azimuth_l2_results, "./path/to/azimuth_predictions.csv", row.names = FALSE)