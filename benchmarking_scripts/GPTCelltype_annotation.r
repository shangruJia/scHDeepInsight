# GPTCelltype cell annotation workflow

library(GPTCelltype)
library(openai)
library(Seurat)

# Set OpenAI API key
Sys.setenv(OPENAI_API_KEY = 'your_openai_api_key')

# Load data
raw_data <- Read10X("./path/to/data")
seurat_obj <- CreateSeuratObject(counts = raw_data, min.cells = 3, min.features = 200)

# Standard preprocessing
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)

# Determine PCs to use
ElbowPlot(seurat_obj)

# Clustering
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:15)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.8)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:15)

# Find markers for all clusters
markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25)

# Run GPTCelltype annotation
# Specify tissue type for better results
res <- gptcelltype(markers, tissuename = 'human PBMC', model = 'gpt-4')

# Add GPT annotations to Seurat object
seurat_obj@meta.data$celltype_gpt <- as.factor(res[as.character(Idents(seurat_obj))])

# Visualize results
DimPlot(seurat_obj, group.by = "celltype_gpt", label = TRUE)

# Save annotated object
saveRDS(seurat_obj, "./path/to/gptcelltype_annotated_seurat.rds")

# Export GPT cell type predictions
gpt_results <- data.frame(
 cell_id = rownames(seurat_obj@meta.data),
 celltype_gpt = seurat_obj@meta.data$celltype_gpt
)
write.csv(gpt_results, "./path/to/gptcelltype_results.csv", row.names = FALSE)