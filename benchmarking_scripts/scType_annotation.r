# sctype cell annotation workflow

# Load required packages
lapply(c("dplyr", "Seurat", "HGNChelper"), library, character.only = T)

# Download sctype functions
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

# Load and create Seurat object
raw_data <- Read10X("./path/to/data")
seurat_obj <- CreateSeuratObject(counts = raw_data, min.cells = 3, min.features = 200)

# Basic QC
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > 200 & percent.mt < 20)

# Standard Seurat workflow
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)

# Check how many PCs to use
ElbowPlot(seurat_obj)

# Clustering and UMAP
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:15)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.8)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:15)
DimPlot(seurat_obj, reduction = "umap", label = TRUE)

# Prepare sctype gene sets
tissue <- "Immune system"  # change based on your tissue
db_ <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"
gs_list <- gene_sets_prepare(db_, tissue)

# Get scaled data matrix
scRNAseqData_scaled <- as.matrix(GetAssayData(seurat_obj, slot = "scale.data", assay = "RNA"))

# Run sctype scoring
es.max <- sctype_score(scRNAseqData = scRNAseqData_scaled, 
                      scaled = TRUE, 
                      gs = gs_list$gs_positive, 
                      gs2 = gs_list$gs_negative)

# Summarize scores by cluster
cL_results <- do.call("rbind", lapply(unique(seurat_obj@meta.data$seurat_clusters), function(cl){
   es.max.cl = sort(rowSums(es.max[, rownames(seurat_obj@meta.data[seurat_obj@meta.data$seurat_clusters==cl, ])]), 
                    decreasing = TRUE)
   head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl), 10)
}))

sctype_scores <- cL_results %>% 
   group_by(cluster) %>% 
   top_n(n = 1, wt = scores)

# Check results
print(sctype_scores[,1:3])

# Annotate clusters
seurat_obj@meta.data$cell_type <- ""
for(j in unique(sctype_scores$cluster)){
   cl_type <- sctype_scores[sctype_scores$cluster==j, ]$type
   seurat_obj@meta.data$cell_type[seurat_obj@meta.data$seurat_clusters == j] <- cl_type
}

# Visualize annotated cells
DimPlot(seurat_obj, reduction = "umap", group.by = 'cell_type', label = TRUE, repel = TRUE)

# Save annotation results
saveRDS(seurat_obj, "annotated_seurat.rds")
write.csv(sctype_scores, "sctype_results.csv", row.names = FALSE)