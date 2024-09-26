
# SCHdeepinsight

SCHdeepinsight is a Python package designed for processing and annotating single-cell RNA sequencing (scRNA-seq) data, specifically for immune cells. It leverages **DeepInsight** and **Convolutional Neural Networks (CNN)** to develop an **automated** model for annotating immune cells. By conducting an in-depth analysis of the **hierarchical structure** of immune cells, the model achieves highly efficient and accurate cell type annotation for single-cell RNA sequencing (scRNA-seq) data. The model is particularly effective in handling immune cells, demonstrating exceptional accuracy in identifying both common and potential rare cell types.

## Features

- **Preprocessing**: Normalizes and logarithmically transforms scRNA-seq data stored in `.h5ad` files.
- **Image Transformation**: Converts processed scRNA-seq data into images suitable for deep learning model input.
- **Cell Type Prediction**: Uses a pre-trained deep learning model to predict cell types, including base and detailed types.
- **Rare Cell Identification**: Identifies potential rare cell types by analyzing prediction probabilities.

## Installation

You can install SCHdeepinsight using `pip`. Note that the package has a dependency on `pyDeepInsight`, which is installed from GitHub.

```bash
python3 -m pip -q install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
```

## R Dependencies

Before running the batch correction process, ensure that the following R packages are installed. Some of these packages need to be installed directly from GitHub. The installation instructions include commands to install both CRAN packages and GitHub packages using the `remotes` package.

```r
# Install the remotes package if not already installed
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

# Load the remotes package
library(remotes)

# Install necessary packages from GitHub
remotes::install_github("carmonalab/STACAS")
remotes::install_github("carmonalab/ProjecTILs")
remotes::install_github("mojaveazure/seurat-disk")

# Install other required packages from CRAN
install.packages(c("Seurat", "Matrix"))
```

### Notes

- Ensure you have an active internet connection to download the packages.
- If you encounter any installation issues, ensure you have the necessary development tools for your operating system, as some packages may require compilation.

## Usage

Here's a brief overview of how to use SCHdeepinsight in your workflow:

1. **Preprocess the Data**: Use the `preprocess` method to normalize and log-transform your scRNA-seq data. This step prepares the data for further analysis.
2. **Batch Correction**: Perform batch correction using the `batch_correction` method if you need to correct for batch effects. This method uses an R script to project the query dataset onto a reference, ensuring that technical differences between batches do not interfere with downstream analysis. If batch correction is not required, you can skip this step.
3. **Image Transformation**: Convert the processed data into images using the `image_transform` method. This step is crucial for transforming the gene expression data into a format suitable for input into the deep learning model.
4. **Prediction**: Use the `predict` method to classify cell types. This step includes both base type and detailed subtype classification and identifies potential rare cell types based on probability thresholds.

## Example

Here’s an example of how to use the `Immune` class to preprocess, batch correct, transform images, and predict:

```python
# Import the Immune class
from immune import Immune

# Set the output prefix path
output_prefix = "output_directory"

# Create an instance of the Immune class
immune = Immune(output_prefix=output_prefix)

# Option 1: Batch correction (Recommended)
ref_file = "reference.h5ad"  # Path to the reference data file
batch_corrected_path = immune.batch_correction(input_file="input_query.h5ad", ref_file=ref_file)
print(f"Batch-corrected file saved at: {batch_corrected_path}")

# Option 2: Preprocess the data (Use this if batch correction is not needed)
# query_path = "input_query.h5ad"  # Path to the input data file
# preprocessed_path = immune.preprocess(query_path)
# print(f"Preprocessed file saved at: {preprocessed_path}")

# Image transformation
# By default, use the batch-corrected path if batch correction was performed
# If preprocessing was used instead, pass the preprocessed path to image_transform
image_path = immune.image_transform(query_path=batch_corrected_path)
print(f"Image data saved at: {image_path}")

# Prediction
predictions = immune.predict(batch_size=128, rare_base_threshold=60, rare_detailed_threshold=10)
print("Prediction results:")
print(predictions)
```

### Explanation

1. **Create an Immune Instance**:
   - Use the `output_prefix` parameter to specify the directory for output files.

2. **Option 1: Batch Correction (Recommended)**:
   - Perform batch correction on the input data using the `batch_correction` method. This method corrects the input data based on a reference dataset and saves the corrected data. The batch-corrected file is recommended for further analysis.

3. **Option 2: Data Preprocessing**:
   - If batch correction is not needed, you can use the `preprocess` method to normalize and log-transform the input `.h5ad` file. Only use this if batch correction is unnecessary.

4. **Image Transformation**:
   - This step converts the batch-corrected data into image format for prediction. By default, it uses the batch-corrected path. If you used preprocessing instead, replace `batch_corrected_path` with `preprocessed_path` in this step.

5. **Prediction**:
   - Use the `predict` method to make predictions on the transformed image data, outputting the prediction results, including cell types and potential rare cell markers.
