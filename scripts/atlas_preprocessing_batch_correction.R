# Load required libraries for Seurat-based integration and visualization
library(Seurat)
library(SeuratDisk)
library(ggplot2)
library(hdf5r)
library(sctransform)
library(Matrix)
library(tibble)
library(dplyr)
library(Matrix)
library(STACAS)
library(SignatuR)
library(dplyr)
library(ggplot2)
library(scIntegrationMetrics)

# Load preprocessed reference Seurat object
atlas <- readRDS("./immune/immune_atlas_qc.rds")
atlas
atlas@meta.data$detailed_type <- replace(atlas@meta.data$detailed_type, atlas@meta.data$detailed_type== "NK_CD56+", "NK_CD16-")
atlas@meta.data$detailed_type <- replace(atlas@meta.data$detailed_type, atlas@meta.data$detailed_type== "Erythrocyte_Erythrocyte", "Erythrocyte")
atlas@meta.data["nCount_RNA"] <- colSums(x = atlas, slot = "counts")  
atlas@meta.data["nFeature_RNA"] <- colSums(x = GetAssayData(object = atlas, slot = "counts") > 0)
atlas <- PercentageFeatureSet(atlas, pattern = "^MT-", col.name = "percent.mt")
head(atlas@meta.data)
as_tibble(
  atlas[[]],
  rownames="Barcodes"
) -> qc.metrics
theme_set(theme_bw(base_size = 22))
qc.metrics %>%
  arrange(percent.mt) %>%
  ggplot(aes(nCount_RNA,nFeature_RNA)) + 
  theme(text = element_text(size = 15)) +
  geom_point(aes(color=percent.mt),size=2) + 
  scale_color_gradientn(colors=c("purple","blue","green","yellow","red"), limits=c(0,20)) +
  guides(colour = guide_coloursteps(show.limits = TRUE, barheight=20))+
  ggtitle("Metrics before QC") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 400) +
  geom_hline(yintercept = 5000) +
  scale_x_log10() + scale_y_log10() +
  theme(plot.margin = unit(c(2,2,2,2), "cm"))
head(atlas@meta.data)
genes_of_interest <- read.csv("./sch_immune/genes.csv", stringsAsFactors = FALSE)
genes_of_interest <- genes_of_interest$genes  
atlas <- subset(atlas, features = genes_of_interest)
atlas
atlas <- SCTransform(atlas, vst.flavor= "v2", do.correct.umi = TRUE, 
                     vars.to.regress = c("percent.mt
                                         "),
                     residual.type = "pearson", variable.features.n = 5000, 
                     ncells=5000, verbose = TRUE)
atlas
str(atlas[["SCT"]])
atlas[["SCT"]]
atlas[["SCT"]]@SCTModel.list$counts
saveRDS(
  object = atlas[["SCT"]]@SCTModel.list$counts, 
  file   = "./sch_immune/sct_model.rds"
)
atlas[["SCT"]]@counts
counts_data <- LayerData(atlas[["RNA"]], layer = "counts")
counts_data
atlas
atlas[["RNA"]] <- NULL
DefaultAssay(atlas) <- "SCT"
atlas <- RunPCA(atlas, 
                assay = "SCT",
                npcs = 50, 
                verbose = FALSE)
atlas <- RunUMAP(atlas,
                 dims = 1:50,  
                 verbose = FALSE)
atlas@meta.data
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas, 
             reduction = "umap",
             group.by = "primary_type",  
             label = TRUE,  
             label.size = 4,
             repel = TRUE) + 
     ggtitle("Cell Types") +
     theme_minimal()
p
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas, 
             reduction = "umap",
             group.by = "detailed_type",
             label = TRUE,
             label.size = 4,
             repel = TRUE) + 
     theme_minimal() +
     theme(legend.position = "none") +
     ggtitle("Cell Types")
p
selected_types <- c("CD4+T")  
atlas_subset1 <- subset(atlas, subset = primary_type %in% selected_types)
selected_types <- c("CD8+T")  
atlas_subset2 <- subset(atlas, subset = primary_type %in% selected_types)
selected_types <- c("CD4+T", "CD8+T")  
atlas_subset3 <- subset(atlas, subset = primary_type %in% selected_types)
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas_subset1, 
             reduction = "umap",
             group.by = "detailed_type",
             label = TRUE,
             label.size = 4,
             repel = TRUE) + 
     theme_minimal() +
     ggtitle("Cell Types")
p
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas_subset2, 
             reduction = "umap",
             group.by = "detailed_type",
             label = TRUE,
             label.size = 4,
             repel = TRUE) + 
     theme_minimal() +
     ggtitle("Cell Types")
p
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas_subset3, 
             reduction = "umap",
             group.by = "detailed_type",
             label = TRUE,
             label.size = 4,
             repel = TRUE) + 
     theme_minimal() +
     ggtitle("Cell Types")
p
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(atlas_subset2, 
             reduction = "pca",
             group.by = "detailed_type",
             label = TRUE,
             label.size = 4,
             repel = TRUE) + 
     theme_minimal() +
     ggtitle("Cell Types")
p
ElbowPlot(atlas_subset2, ndims = 50)
VizDimLoadings(atlas_subset2, dims = 1:5, reduction = "pca")
object <- atlas_subset2
pca_data <- Embeddings(object, reduction = "pca")
pca_data <- as.data.frame(pca_data)
pca_data$detailed_type <- object$detailed_type
library(ggplot2)
ggplot(pca_data, aes(x = detailed_type, y = PC_1, fill = detailed_type)) +
  geom_boxplot() + theme_minimal()
FeaturePlot(object, features = "percent.mt", reduction = "pca")
FeaturePlot(object, features = "percent.mt", reduction = "umap")


ref_obj <- readRDS("./sch_immune/reference_subset.rds")
ref_obj
raw_data = Read10X("./query/lee/matrix_files")
metadata = read.csv("./query/lee/metadata.csv")
query_obj = CreateSeuratObject(counts = raw_data, meta.data = metadata)
query_obj <- NormalizeData(query_obj)
shared_genes <- intersect(rownames(ref_obj), rownames(query_obj))
query_obj <- query_obj[shared_genes, ]
ref_obj <- ref_obj[shared_genes, ]
batch_vector <- c(rep("reference", ncol(ref_obj)), rep("query", ncol(query_obj)))
project_query_stacas <- function(query_obj, ref_obj, ndims = 30) {
  ref_assay <- DefaultAssay(ref_obj)
  query_assay <- DefaultAssay(query_obj)
  shared_genes <- intersect(rownames(ref_obj), rownames(query_obj))
  VariableFeatures(ref_obj) <- shared_genes
  VariableFeatures(query_obj) <- shared_genes
  proj_anchors <- FindAnchors.STACAS(
    object.list = list(ref_obj, query_obj),
    assay = c(ref_assay, query_assay),
    anchor.features = shared_genes,  # Use all commone genes
    dims = 1:ndims,
    k.anchor = 5,
    anchor.coverage = 1,
    correction.scale = 100,
    alpha = 0.5,
    verbose = FALSE
  )
  integration_tree <- matrix(c(-1, -2), nrow = 1, ncol = 2)
  projected <- IntegrateData.STACAS(
    proj_anchors,
    k.weight = 100,
    dims = 1:ndims,
    sample.tree = integration_tree,
    features.to.integrate = shared_genes,  # Integrate all commonn genes
    verbose = FALSE
  )
  query_cells <- colnames(query_obj)
  projected_query <- subset(projected, cells = query_cells)
  projected_query@meta.data <- query_obj@meta.data
  return(projected_query)
}
start_time <- proc.time()
projected_query <- project_query_stacas(
  query_obj = query_obj, 
  ref_obj = ref_obj,
  ndims = 30
)
total_time <- proc.time() - start_time
print("--------- Time Summary ---------")
print(paste("Total execution time:", total_time[3], "seconds"))
print(paste("                    ", total_time[3]/60, "minutes"))
corrected_expression <- GetAssayData(projected_query, assay = "integrated", slot = "data")
corrected_expression
projected_query
query_raw <- GetAssayData(projected_query, assay = "RNA", slot = "counts")
query_df <- corrected_expression
projected_query <- SetAssayData(projected_query, new.data = as.matrix(query_df))
cat("Saving the results...\n")
projected_query@assays[["RNA"]] <- NULL
SaveH5Seurat(projected_query, "./sch_immune/batch_corrected_query.h5seurat", assay="integrated")
Convert("./sch_immune/batch_corrected_query.h5seurat", dest = "h5ad")
projected_query@meta.data
projected_query <- ScaleData(projected_query)
projected_query <- RunPCA(projected_query)
projected_query <- RunUMAP(projected_query, dims = 1:30)
projected_query
png("./sch_immune/umap_cell_types.png", width = 6000, height = 2000, res = 300)
# Visualize UMAP embedding with primary cell type labels
DimPlot(projected_query, 
        reduction = "umap",
        group.by = "cell_type",
        label = TRUE,
        label.size = 4)
dev.off()
SaveH5Seurat(projected_query, 
            "./sch_immune/batch_corrected_query.h5seurat", 
             assay="integrated")
Convert("./sch_immune/batch_corrected_query.h5seurat", 
        dest = "h5ad")
# Save integrated expression matrix in Matrix Market format
writeMM(corrected_expression, file = "./sch_immune/query_matrix.mtx")
write.table(rownames(corrected_expression), file = "./sch_immune/query_features.tsv", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(colnames(corrected_expression), file = "./sch_immune/query_barcodes.tsv", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)


seed = 1234
# Set the random seed for reproducibility
set.seed(seed)
# Load preprocessed reference Seurat object
atlas <- readRDS("./immune/immune_atlas_qc_sct.rds")
atlas
atlas$part <- paste(as.character(atlas$dataset), as.character(atlas$assay), sep="_")
atlas@meta.data$detailed_type <- replace(atlas@meta.data$detailed_type, atlas@meta.data$detailed_type== "NK_CD56+", "NK_CD16-")
atlas@meta.data$detailed_type <- replace(atlas@meta.data$detailed_type, atlas@meta.data$detailed_type== "Erythrocyte_Erythrocyte", "Erythrocyte")
atlas[["RNA"]] <- NULL
# Replace RNA assay with SCT assay data for further processing
atlas[["RNA"]] <- atlas[["SCT"]]
DefaultAssay(atlas) <- "RNA"
atlas[["SCT"]] <- NULL
atlas <- SetAssayData(object = atlas, assay = "RNA", layer = "scale.data", new.data = matrix(0, 0, 0))
atlas
# Define a utility function for logging with timestamps
log_with_time <- function(msg, verbose = TRUE) {
    if(verbose) {
        timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
        print(paste0("[", timestamp, "] ", msg))
    }
}
# Function to report current memory usage
report_memory <- function() {
    mem <- gc(full = TRUE)
    used_mb <- sum(mem[,2])
    return(paste0("Memory usage: ", round(used_mb, 2), " MB"))
}
# Function to report progress percentage
calculate_progress <- function(current, total) {
    return(paste0(round(current/total * 100, 2), "%"))
}
# Split the reference object into two balanced batches based on cell types
batch_split <- function(atlas, checkpoint_dir = "checkpoints", verbose = TRUE) {
    dir.create(checkpoint_dir, showWarnings = FALSE)
    batch1_file <- file.path(checkpoint_dir, "batch1.rds")
    batch2_file <- file.path(checkpoint_dir, "batch2.rds")
    if(file.exists(batch1_file) && file.exists(batch2_file)) {
# Define a utility function for logging with timestamps
        log_with_time("Loading existing batch splits...", verbose)
        return(list(
            batch1 = readRDS(batch1_file),
            batch2 = readRDS(batch2_file)
        ))
    }
    cell_types <- unique(atlas$primary_type)
    batch1_cells <- c()
    batch2_cells <- c()
    for(ct in cell_types) {
        cells_of_type <- colnames(atlas)[atlas$primary_type == ct]
        n_cells <- length(cells_of_type)
        cells_of_type <- sample(cells_of_type)
        mid <- ceiling(n_cells/2)
        batch1_cells <- c(batch1_cells, cells_of_type[1:mid])
        batch2_cells <- c(batch2_cells, cells_of_type[(mid+1):n_cells])
        if(verbose) {
# Define a utility function for logging with timestamps
            log_with_time(sprintf("Split %s: %d cells to batch1, %d cells to batch2", 
                                ct, mid, n_cells-mid), verbose)
        }
    }
# Define a utility function for logging with timestamps
    log_with_time("Creating and saving batch1...", verbose)
    batch1 <- atlas[, batch1_cells]
    saveRDS(batch1, batch1_file)
# Define a utility function for logging with timestamps
    log_with_time("Creating and saving batch2...", verbose)
    batch2 <- atlas[, batch2_cells]
    saveRDS(batch2, batch2_file)
    if(verbose) {
# Define a utility function for logging with timestamps
        log_with_time("Batch 1 composition:", verbose)
        print(table(batch1$primary_type))
# Define a utility function for logging with timestamps
        log_with_time("Batch 2 composition:", verbose)
        print(table(batch2$primary_type))
    }
    rm(batch1, batch2)
    gc()
    return(list(
        batch1 = readRDS(batch1_file),
        batch2 = readRDS(batch2_file)
    ))
}
# Select variable features across all parts of the dataset for integration
SelectFeaturesForAll <- function(atlas, checkpoint_dir = "checkpoints", verbose = TRUE) {
    features_file <- file.path(checkpoint_dir, "variable_features.rds")
    if(file.exists(features_file)) {
# Define a utility function for logging with timestamps
        log_with_time("Loading existing variable features", verbose)
        return(readRDS(features_file))
    }
# Define a utility function for logging with timestamps
    log_with_time("Splitting dataset for feature selection...", verbose)
    obj_list <- SplitObject(atlas, split.by = "part")
# Define a utility function for logging with timestamps
    log_with_time("Selecting integration features...", verbose)
    variable_features <- SelectIntegrationFeatures(obj_list, nfeatures = 5000)
# Define a utility function for logging with timestamps
    log_with_time("Saving selected features...", verbose)
    saveRDS(variable_features, features_file)
    rm(obj_list)
    gc()
    return(variable_features)
}
# Function to run STACAS integration on a given batch
IntegrateBatch <- function(batch, batch_name, checkpoint_dir, genes.blocklist = NULL, 
                          variable_features = NULL, verbose = TRUE) {
    integrated_file <- file.path(checkpoint_dir, paste0(batch_name, "_integrated.rds"))
    if(file.exists(integrated_file)) {
# Define a utility function for logging with timestamps
        log_with_time(paste("Loading existing integration for", batch_name), verbose)
        return(readRDS(integrated_file))
    }
# Define a utility function for logging with timestamps
    log_with_time(paste("Splitting", batch_name, "by part..."), verbose)
    obj_list <- SplitObject(batch, split.by = "part")
    rm(batch)
    gc()
# Define a utility function for logging with timestamps
    log_with_time(paste("Running STACAS integration for", batch_name, "..."), verbose)
    tryCatch({
        integrated <- Run.STACAS(
            obj_list,
            genesBlockList = genes.blocklist,
            dims = 1:20,
            anchor.features = variable_features,
            k.anchor = 40,
            k.score = 60,
            k.weight = 30,
            cell.labels = "primary_type",
            verbose = verbose
        )
# Define a utility function for logging with timestamps
        log_with_time(paste("Saving", batch_name, "integration result..."), verbose)
        saveRDS(integrated, integrated_file)
        rm(obj_list)
        gc()
        return(integrated)
    }, error = function(e) {
        error_info <- list(
            error_message = e$message,
# Function to report current memory usage
            memory_state = report_memory(),
            timestamp = Sys.time(),
            batch = batch_name
        )
        saveRDS(error_info, file.path(checkpoint_dir, paste0(batch_name, "_error.rds")))
        stop(paste("Integration failed for", batch_name, ":", e$message))
    })
}
# Set working directory for output and checkpoints
setwd("/Usersdata/shangru/docker/merge_atlas")  # 设置工作目录
checkpoint_dir <- "/Usersdata/shangru/docker/merge_atlas/checkpoints"  # 检查点目录名
save.path <- file.path(checkpoint_dir, "integrated_final.rds")
# Load SignatuR human signature sets
hs.sign <- GetSignature(SignatuR$Hs)
my.genes.blocklist <- c(GetSignature(SignatuR$Hs$Blocklists),
                        GetSignature(SignatuR$Hs$Compartments))
# Select variable features across all parts of the dataset for integration
variable_features <- SelectFeaturesForAll(
    atlas = atlas,
    checkpoint_dir = checkpoint_dir,
    verbose = TRUE
)
# Perform batch splitting of the reference object
batches <- batch_split(atlas, checkpoint_dir, verbose = TRUE)
rm(atlas)
gc()
# Define a utility function for logging with timestamps
log_with_time("Processing batch 1...")
# Function to run STACAS integration on a given batch
batch1_integrated <- IntegrateBatch(
    batch = batches$batch1,
    batch_name = "batch1",
    checkpoint_dir = checkpoint_dir,
    genes.blocklist = my.genes.blocklist,
    variable_features = variable_features,
    verbose = TRUE
)
rm(batch1_integrated)
gc()
# Define a utility function for logging with timestamps
log_with_time("Processing batch 2...")
# Function to run STACAS integration on a given batch
batch2_integrated <- IntegrateBatch(
    batch = batches$batch2,
    batch_name = "batch2",
    checkpoint_dir = checkpoint_dir,
    genes.blocklist = my.genes.blocklist,
    variable_features = variable_features,
    verbose = TRUE
)
batch2_integrated
batch1_integrated <- readRDS("/Usersdata/shangru/docker/merge_atlas/checkpoints/batch1_integrated.rds")
batch1_integrated@assays$RNA <- NULL
batch1_integrated[["pca"]] <- NULL
batch1_integrated
# Merge two integrated batches into one final Seurat object
merged <- merge(batch1_integrated, batch2_integrated)
sparse_mat <- merged@assays$integrated@data
dir.create("./ref_matrix")
# Save integrated expression matrix in Matrix Market format
writeMM(sparse_mat, file="./ref_matrix/matrix.mtx")
writeLines(rownames(sparse_mat), "./ref_matrix/features.tsv")
writeLines(colnames(sparse_mat), "./ref_matrix/barcodes.tsv")
# Save merged integrated Seurat object
saveRDS(merged, "./merged_integrated.rds")
merged
merged
# Set variable features to row names
VariableFeatures(merged) <- rownames(merged)
head(rownames(merged))
merged <-merged %>% ScaleData() %>%
  RunPCA(npcs=20) %>% RunUMAP(dims=1:20)
merged <-merged %>% ScaleData() %>%
  RunPCA(npcs=20) %>% RunUMAP(dims=1:20)
# Visualize UMAP embedding with primary cell type labels
p <- DimPlot(merged, group.by = "primary_type", label=T, label.size = 2) +
  NoLegend() + theme(aspect.ratio = 1) + ggtitle("Cell labels after integration") 
head(merged@assays$integrated@data[1:5, 1:5])
batch2_integrated@assays$RNA <- NULL
batch2_integrated[["pca"]] <- NULL
batch2_integrated[["umap"]] <- NULL
batch2_integrated
head(batch1_integrated@assays$integrated@data[1:5, 1:5])
head(batch2_integrated@assays$integrated@data[1:5, 1:5])
batch2_integrated <- batch2_integrated %>% RunUMAP(dims=1:20)
p2_ss
# Compute integration metrics such as CiLISI and celltype_ASW
integrationMetrics <- list()
# Compute integration metrics such as CiLISI and celltype_ASW
integrationMetrics[["ssSTACAS"]] <- getIntegrationMetrics(object=batch2_integrated,
                                                      metrics = c("CiLISI","celltype_ASW"),
                                                      meta.label = "primary_type",
                                                      meta.batch = "part")
# Compute integration metrics such as CiLISI and celltype_ASW
integrationMetrics