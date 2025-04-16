# Load necessary libraries
suppressPackageStartupMessages({
  library(Seurat)
  library(STACAS)
  library(Matrix)
  library(SeuratDisk)
})

process_and_project_data <- function(output_prefix, ref_file) {
  # Check for necessary libraries
  required_packages <- c("Seurat", "SeuratDisk", "STACAS", "Matrix")
  
  missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
  
  if (length(missing_packages) > 0) {
    stop(paste("The following required packages are missing. Please install them before running this script:", 
               paste(missing_packages, collapse = ", ")))
  }
  
  # Load the reference
  cat("Loading reference data...\n")
  ref_obj <- readRDS(ref_file)
  
  # Read the matrix files
  cat("Reading matrix files...\n")
  raw_data <- Read10X(file.path(output_prefix, "matrix_files"))
  
  # Use read.csv to read metadata
  metadata <- read.csv(file.path(output_prefix, "metadata.csv"))
  
  # Create Seurat object
  cat("Creating Seurat object...\n")
  query_obj <- CreateSeuratObject(counts = raw_data, meta.data = metadata)
  
  # Normalize data 
  cat("Normalizing data\n")
  query_obj <- NormalizeData(query_obj)
  
  # Automatic gene alignment
  cat("Aligning genes...\n")
  shared_genes <- intersect(rownames(ref_obj), rownames(query_obj))
  query_obj <- query_obj[shared_genes, ]
  ref_obj <- ref_obj[shared_genes, ]
  
  # Set all shared genes as variable features
  VariableFeatures(ref_obj) <- shared_genes
  VariableFeatures(query_obj) <- shared_genes
  
  # Create a batch vector
  batch_vector <- c(rep("reference", ncol(ref_obj)), rep("query", ncol(query_obj)))
  
  # Project query using STACAS
  cat("Finding anchors...\n")
  
  start_time <- proc.time()
  
  proj_anchors <- FindAnchors.STACAS(
    object.list = list(ref_obj, query_obj),
    assay = c(DefaultAssay(ref_obj), DefaultAssay(query_obj)),
    anchor.features = shared_genes,
    dims = 1:20,
    k.anchor = 40,
    k.score = 60,
    verbose = FALSE
  )

  # 1. Find anchors
#   proj_anchors <- FindAnchors.STACAS(
#     object.list = list(ref_obj, query_obj),
#     assay = c(DefaultAssay(ref_obj), DefaultAssay(query_obj)),
#     anchor.features = shared_genes,
#     dims = 1:30,
#     k.anchor = 5,
#     anchor.coverage = 1,
#     correction.scale = 100,
#     alpha = 0.5,
#     verbose = FALSE
#   )
  
  # 2. Set integration tree
  integration_tree <- matrix(c(-1, -2), nrow = 1, ncol = 2)
  
  # 3. Integrate data
  cat("Integrating data...\n")
  projected <- IntegrateData.STACAS(
    proj_anchors,
    k.weight = 30,
    dims = 1:20,
    sample.tree = integration_tree,
    features.to.integrate = shared_genes,
    verbose = FALSE
  )
  
  # 4. Extract only query cells
  cat("Extracting query cells...\n")
  query_cells <- colnames(query_obj)
  projected_query <- subset(projected, cells = query_cells)
  
  # 5. Restore query metadata
  projected_query@meta.data <- query_obj@meta.data
  
  # Calculate time
  total_time <- proc.time() - start_time
  cat("--------- Time Summary ---------\n")
  cat(paste("Total execution time:", total_time[3], "seconds\n"))
  cat(paste("                     ", total_time[3]/60, "minutes\n"))
  
  # Save as .h5seurat and convert to .h5ad
  cat("Saving the results...\n")
  projected_query@assays[["RNA"]] <- NULL
  
  SaveH5Seurat(projected_query, file.path(output_prefix, "batch_corrected_query.h5seurat"), assay="integrated")
  Convert(file.path(output_prefix, "batch_corrected_query.h5seurat"), dest = "h5ad")
  
  cat("Batch correction process completed successfully.\n")
  cat("The corrected file was saved as batch_corrected_query.h5ad in the output directory.\n")
}