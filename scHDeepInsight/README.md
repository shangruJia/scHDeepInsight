# scHDeepInsight

scHDeepInsight is a Python package designed for processing and annotating single-cell RNA sequencing (scRNA-seq) data, specifically for immune cells. It leverages **DeepInsight** and **Convolutional Neural Networks (CNN)** to develop an **automated** model for annotating immune cells. By conducting an in-depth analysis of the **hierarchical structure** of immune cells, the model achieves highly efficient and accurate cell type annotation for single-cell RNA sequencing (scRNA-seq) data. The model is particularly effective in handling immune cells, demonstrating exceptional accuracy in identifying both common and potential rare cell types.

## Features

- **Preprocessing**: Normalizes and logarithmically transforms scRNA-seq data stored in `.h5ad` files.
- **Image Transformation**: Converts processed scRNA-seq data into images suitable for deep learning model input.
- **Cell Type Prediction**: Uses a pre-trained deep learning model to predict cell types, including base and detailed types.
- **Rare Cell Identification**: Identifies potential rare cell types by analyzing prediction probabilities.

## Installation

You can install scHDeepInsight using `pip`. Note that the package has a dependency on `pyDeepInsight`, which is installed from GitHub.

```bash
python3 -m pip -q install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
