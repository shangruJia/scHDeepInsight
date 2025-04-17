# scHDeepinsight

scHDeepinsight is a Python package for hierarchical annotation of immune cells in single-cell RNA sequencing (scRNA-seq) data. By combining **DeepInsight** transformation with a **hierarchical CNN model**, it provides accurate classification of immune cell types with both base-level and detailed subtype identification.

## Features

- **Batch Correction**: Aligns query data with the integrated reference dataset using STACAS integration
- **Image Transformation**: Converts gene expression matrices to image representations
- **Hierarchical Classification**: Two-level annotation providing both major cell types and detailed subtypes
- **Rare Cell Detection**: Identifies potential rare cell populations based on prediction confidence

## Installation

Install SCHdeepinsight using pip:

```bash
pip install SCHdeepinsight
```

Additionally, the package requires pyDeepInsight:

```bash
pip install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
```

## R Dependencies

For batch correction functionality, the following R packages are required:

```r
# Install required CRAN packages
install.packages(c("Seurat", "Matrix", "SeuratDisk"))

# Install packages from GitHub
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Install STACAS
remotes::install_github("carmonalab/STACAS")
```

## Usage

Here's how to use SCHdeepinsight to analyze your scRNA-seq data:

```python
from SCHdeepinsight import immune

# Initialize with output directory
classifier = immune("./output_dir")

# Complete pipeline: batch correction, image transformation, and prediction
results = classifier.run_pipeline(
    input_file="path/to/query.h5ad",
    ref_file="path/to/reference.rds", 
    batch_size=128
)

# Access prediction results
print(results.head())

# Save results
results.to_csv("./immune_cell_predictions.csv")
```

### Step-by-Step Approach

For more fine-grained control, you can run each step separately:

```python
from SCHdeepinsight import immune

# Initialize
classifier = immune("./output_dir")

# Step 1: Batch correction
corrected_file = classifier.batch_correction(
    input_file="path/to/query.h5ad",
    ref_file="path/to/reference.rds"
)

# Step 2: Transform gene expression to images
classifier.image_transform(corrected_file)

# Step 3: Predict cell types
results = classifier.predict(
    batch_size=128,
    rare_base_threshold=60,
    rare_detailed_threshold=10
)
```

## Prediction Results

The results DataFrame contains:

- `barcode`: Cell identifiers
- `predicted_base_type`: Major cell type classification
- `predicted_detailed_type`: Detailed subtype classification
- `base_type_probability`: Confidence score for base type prediction
- `detailed_type_probability`: Confidence score for detailed type prediction
- `is_potential_rare`: Boolean flag for potential rare cell types
