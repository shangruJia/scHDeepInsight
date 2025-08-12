# Garnett cell annotation with custom trained classifier

library(Seurat)
library(garnett)
library(monocle)
library(org.Hs.eg.db)

# Convert Seurat to CDS
convertSeuratToCDS <- function(seurat_obj) {
   data <- GetAssayData(seurat_obj, slot = "counts")
   pd <- new("AnnotatedDataFrame", data = seurat_obj@meta.data)
   fd <- data.frame(gene_short_name = rownames(seurat_obj),
                    row.names = rownames(seurat_obj))
   fd <- new("AnnotatedDataFrame", data = fd)
   
   cds <- newCellDataSet(as(data, "sparseMatrix"),
                         phenoData = pd,
                         featureData = fd,
                         lowerDetectionLimit = 0.5,
                         expressionFamily = negbinomial.size())
   return(cds)
}

# Load reference data for training
reference <- readRDS("./path/to/reference.rds")
ref_cds <- convertSeuratToCDS(reference)
ref_cds <- estimateSizeFactors(ref_cds)

# Create marker file from reference
markers <- FindAllMarkers(reference, only.pos = TRUE, min.pct = 0.25)
top_markers <- markers %>% 
   group_by(cluster) %>% 
   top_n(10, avg_log2FC)

sink("./path/to/markers.txt")
for (ct in unique(reference$cell_type)) {
   genes <- top_markers[top_markers$cluster == ct, "gene"]
   cat(paste0(">", ct, "\n"))
   cat(paste0("expressed: ", paste(genes, collapse = ", "), "\n\n"))
}
sink()

# Train classifier
classifier <- train_cell_classifier(
   cds = ref_cds,
   marker_file = "./path/to/markers.txt",
   db = org.Hs.eg.db,
   cds_gene_id_type = "SYMBOL",
   num_unknown = 50
)

# Load query data
raw_data <- Read10X("./path/to/data")
seurat_obj <- CreateSeuratObject(counts = raw_data)

# Convert and prepare query data
cds <- convertSeuratToCDS(seurat_obj)
cds <- estimateSizeFactors(cds)

# Run classification
cds <- classify_cells(cds, classifier,
                    db = org.Hs.eg.db,
                    cluster_extend = TRUE,
                    cds_gene_id_type = "SYMBOL")

# Extract and save results
predictions <- pData(cds)[, c("cell_type", "cluster_ext_type")]
write.csv(predictions, "./path/to/garnett_predictions.csv")