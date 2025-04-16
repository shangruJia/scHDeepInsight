import os
from pathlib import Path
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.io as io
from scipy.sparse import issparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from huggingface_hub import hf_hub_download
import pickle
from sklearn import preprocessing
import cv2
import warnings
import rpy2.robjects as ro
import gzip

warnings.filterwarnings("ignore")

class HierarchicalNet(nn.Module):
    """
    Hierarchical CNN model for cell type classification.
    
    This model has a two-level classification scheme:
    1. Base classifier for major cell types
    2. Detailed classifiers for cell subtypes within each major type
    """
    def __init__(self, num_base_classes, num_detailed_classes, index):
        """
        Initialize the hierarchical neural network model.
        
        Args:
            num_base_classes (int): Number of base/major cell types
            num_detailed_classes (int): Total number of detailed cell subtypes
            index (list): List containing the number of subtypes for each base type
        """
        super().__init__()
        # Feature extractor backbone (EfficientNet-B5)
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5')
        backbone_out = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # Remove final FC layer to get features
        
        # Base classifier for major cell types
        self.base_classifier = nn.Sequential(
            nn.Linear(backbone_out, backbone_out // 2),
            nn.BatchNorm1d(backbone_out // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(backbone_out // 2, num_base_classes)
        )
        
        # Separate detailed classifiers for each base class
        self.detailed_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out, backbone_out // 2),
                nn.BatchNorm1d(backbone_out // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(backbone_out // 2, backbone_out // 4),
                nn.BatchNorm1d(backbone_out // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(backbone_out // 4, size)
            ) for size in index
        ])
        
        self.index = index
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            tuple: (base_logits, detailed_logits_list) where:
                - base_logits: Logits for base cell types
                - detailed_logits_list: List of logits for detailed subtypes, one tensor per base type
        """
        features = self.backbone(x)
        base_logits = self.base_classifier(features)
        
        # Get detailed logits for each base class
        detailed_logits_list = [classifier(features) for classifier in self.detailed_classifiers]
            
        return base_logits, detailed_logits_list


class ImageDataset(Dataset):
    """
    Dataset class for loading cell images for prediction.
    """
    def __init__(self, images):
        """
        Initialize the dataset.
        
        Args:
            images (np.ndarray): Array of images
        """
        self.images = images
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the image
            
        Returns:
            torch.Tensor: Transformed image
        """
        img = self.images[idx]
        img = np.squeeze(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transforms(img)
        return img
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.images)


class immune:
    def __init__(self, output_prefix):
        """
        Initialize the model
        
        Parameters:
        -----------
        output_prefix : str
            Directory for output files
        """
        self.output_prefix = Path(output_prefix)
        os.makedirs(self.output_prefix, exist_ok=True)
        self.matrix_files_dir = self.output_prefix / 'matrix_files'
        os.makedirs(self.matrix_files_dir, exist_ok=True)
        
        # Get package directory
        self.dir = Path(__file__).resolve().parent
        self.pretrained_dir = Path(__file__).resolve().parent / "pretrained_files_immune"
        
        # Load files from package
        self.gene_list = self._load_gene_list()
        self.img_transformer = self._load_img_transformer()
        self.scaler = self._load_scaler()
        self.repo_id = "JiaShangru/scHDeepInsight"
        self.index = [11, 8, 2, 2, 4, 2, 4, 4, 3, 2, 1, 3, 2, 1, 1]
        self.model = self._load_model()

    def _load_gene_list(self):
        """Load gene list from package"""
        gene_list_path = self.pretrained_dir / "pretrained_genes_immune.csv"
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list file not found at {gene_list_path}")
        return pd.read_csv(gene_list_path, index_col=0)["genes"].tolist()

    def _load_img_transformer(self):
        """Load image transformer from package"""
        transformer_path = self.pretrained_dir / "img_transformer_immune.obj"
        if not transformer_path.exists():
            raise FileNotFoundError(f"Image transformer file not found at {transformer_path}")
        with open(transformer_path, 'rb') as file:
            return pickle.load(file)
            
    def _load_scaler(self):
        """Load MinMaxScaler from package"""
        scaler_path = self.pretrained_dir / "minmax_scaler_immune.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"MinMaxScaler file not found at {scaler_path}")
        with open(scaler_path, 'rb') as file:
            return pickle.load(file)

    def _load_model(self):
        """Load model weights from HuggingFace"""
        model = HierarchicalNet(len(self.index), sum(self.index), self.index)
        
        # Download weights from HuggingFace
        try:
            weights_path = hf_hub_download(repo_id="JiaShangru/scHDeepInsight", 
                                         filename="checkpoint_model.pth")
        except Exception as e:
            raise RuntimeError(f"Failed to download model weights: {str(e)}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(model_state_dict)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model.to(device).eval()

    def _decode_labels(self, predictions, encoder_file, column_name):
        """Decode labels using encoders from package"""
        encoder_path = self.pretrained_dir / encoder_file
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
        with open(encoder_path, 'rb') as file:
            encoder = pickle.load(file)
        labels = encoder.inverse_transform(predictions)
        return pd.DataFrame(labels, columns=[column_name])

    def preprocess(self, query_path: str):
        """Normalize and log transform the input data"""
        print("Preprocessing data...")
        query = anndata.read_h5ad(query_path)
        sc.pp.normalize_per_cell(query)
        sc.pp.log1p(query)
        
        output_path = self.output_prefix / "query_preprocessed.h5ad"
        query.write(output_path)
        return output_path

    def write_matrix_files(self, query):
        """Write and compress matrix files for R processing"""
        print("Writing matrix files...")
        # Write barcodes
        barcodes_path = self.matrix_files_dir / 'barcodes.tsv'
        with open(barcodes_path, 'w') as f:
            for item in query.obs_names:
                f.write(f"{item}\n")
        os.system(f"gzip -f {barcodes_path}")

        # Write features
        features_path = self.matrix_files_dir / 'features.tsv'
        with open(features_path, 'w') as f:
            for item in [f"{x}\t{x}\tGene Expression" for x in query.var["feature_name"]]:
                f.write(f"{item}\n")
        os.system(f"gzip -f {features_path}")

        # Write matrix
        matrix_path = self.matrix_files_dir / 'matrix.mtx'
        io.mmwrite(matrix_path, query.X.T)
        os.system(f"gzip -f {matrix_path}")

        # Write metadata
        metadata_path = self.output_prefix / 'metadata.csv'
        query.obs.to_csv(metadata_path)

    def batch_correction(self, input_file, ref_file):
        """Perform batch correction using R"""
        print("Performing batch correction...")
        try:
            r = ro.r
            script_path = self.dir / "r_scripts" / "process_data.R"
            if not script_path.exists():
                raise FileNotFoundError(f"R script not found at {script_path}")
            
            # Read and preprocess data
            query = anndata.read_h5ad(input_file)
            if query.raw is not None:
                query.X = query.raw.X
            
            if "feature_name" not in query.var.columns:
                query.var["feature_name"] = query.var.index.tolist()

            # Write matrix files
            self.write_matrix_files(query)

            # Execute R script
            r.source(str(script_path))
            process_r = ro.globalenv['process_and_project_data']
            process_r(str(self.output_prefix), ref_file)

            return self.output_prefix / "batch_corrected_query.h5ad"
            
        except Exception as e:
            raise RuntimeError(f"Batch correction failed: {str(e)}")

    def image_transform(self, query_path: str):
        """Transform gene expression data to images"""
        print("Transforming data to images...")
        query = anndata.read_h5ad(query_path)
        
        # Ensure feature_name exists and set as index
        query.var["feature_name"] = query.var.get("feature_name", query.var.index.tolist())
        query.var.index = query.var["feature_name"].values

        # Convert to dataframe
        if issparse(query.X):
            query_rna = pd.DataFrame(query.X.toarray())
        else:
            query_rna = pd.DataFrame(query.X)
            
        query_rna.index = query.obs.index.tolist()
        query_rna.columns = query.var.index.tolist()
        
        # Fill missing genes
        def fill_missing_genes_efficient(pretrained_df, all_genes):
            # Create new empty dataframe
            complete_df = pd.DataFrame(0, 
                                     index=pretrained_df.index,  # Sample names as index
                                     columns=all_genes)  # All gene names as columns
            
            # Fill existing values
            complete_df[pretrained_df.columns] = pretrained_df
            
            return complete_df
        
        # Apply scaler and transform
        complete_df = fill_missing_genes_efficient(query_rna, self.gene_list)
        X_query_norm = pd.DataFrame(np.clip(self.scaler.transform(complete_df), 0, 1))
        
        # Transform to images
        queryimg = self.img_transformer.transform(X_query_norm.values)
        queryimg = cv2.normalize(queryimg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Save results
        barcode_path = self.output_prefix / "barcode.csv"
        image_path = self.output_prefix / "query.npy"
        
        self._save_barcode(complete_df, barcode_path)
        np.save(image_path, queryimg)
        
        return image_path

    def predict(self, batch_size: int = 128, rare_base_threshold=60, rare_detailed_threshold=10):
        """Predict cell types"""
        print("Making predictions...")
        barcode_path = self.output_prefix / "barcode.csv"
        image_path = self.output_prefix / "query.npy"

        # Load data
        dataset = ImageDataset(np.load(image_path))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=min(8, os.cpu_count()), pin_memory=True)
        device = next(self.model.parameters()).device

        base_preds = []
        detailed_preds = []
        base_probs = []
        detailed_probs = []
        
        total_processed = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(device)
                base_logits, detailed_logits_list = self.model(data)
                
                # Get base predictions
                base_probs_batch = F.softmax(base_logits, dim=1)
                base_prob_values, base_pred = torch.max(base_probs_batch, dim=1)
                
                # Initialize batch predictions lists
                batch_size = len(data)
                batch_detailed_preds = [None] * batch_size
                batch_detailed_probs = [None] * batch_size
                
                # Process predictions using masking approach
                base_pred_np = base_pred.cpu().numpy()
                
                # Get detailed predictions using mask approach
                for i, logits in enumerate(detailed_logits_list):
                    mask_indices = base_pred_np == i
                    if mask_indices.any():
                        detailed_probs_i = F.softmax(logits[mask_indices], dim=1)
                        probs, preds = torch.max(detailed_probs_i, dim=1)
                        
                        # Adjust prediction index based on cumulative index
                        preds = preds + sum(self.index[:i])
                        
                        # Store predictions and probabilities
                        for idx, (pred, prob) in zip(np.where(mask_indices)[0], 
                                                zip(preds.cpu().numpy(), probs.cpu().numpy())):
                            batch_detailed_preds[idx] = pred.item()
                            batch_detailed_probs[idx] = prob.item()
                
                # Append batch results
                base_preds.append(base_pred.cpu())
                detailed_preds.extend(batch_detailed_preds)
                base_probs.append(base_prob_values.cpu())
                detailed_probs.extend(batch_detailed_probs)
                
                total_processed += batch_size
                if batch_idx % 10 == 0:
                    print(f"Processed {total_processed}/{len(dataset)} images")

        # Process results
        print("Processing results...")
        base_preds = torch.cat(base_preds).numpy()
        base_probs = torch.cat(base_probs).numpy()
        
        # Create initial results DataFrame
        results = pd.DataFrame({
            'barcode': pd.read_csv(barcode_path)['barcode'],
            'predicted_base_type': self._decode_labels(base_preds, 
                                                    "label_encoder_immune_base.obj", 
                                                    "predicted_base_type")['predicted_base_type'],
            'predicted_detailed_type': self._decode_labels(detailed_preds, 
                                                        "label_encoder_immune_detailed.obj", 
                                                        "predicted_detailed_type")['predicted_detailed_type'],
            'base_type_probability': base_probs,
            'detailed_type_probability': detailed_probs
        })
        
        # Add rare cell identification
        print("Identifying potential rare cells...")
        is_potential_rare_series = results.groupby('predicted_base_type', group_keys=False).apply(
            lambda group: self._is_potential_rare(group, 
                                                rare_base_threshold=rare_base_threshold, 
                                                rare_detailed_threshold=rare_detailed_threshold)
        )
        results['is_potential_rare'] = is_potential_rare_series
        
        return results

    def _is_potential_rare(self, base_type_group, rare_base_threshold=60, rare_detailed_threshold=10):
        """
        Identify potential rare cells within each base type group
        
        Parameters:
        -----------
        base_type_group : pandas.DataFrame
            Group of cells with the same base type
        rare_base_threshold : int
            Percentile threshold for base type probability
        rare_detailed_threshold : int
            Percentile threshold for detailed type probability
        """
        base_prob_threshold = np.percentile(base_type_group['base_type_probability'], rare_base_threshold)
        detailed_prob_threshold = np.percentile(base_type_group['detailed_type_probability'], rare_detailed_threshold)
        
        return (base_type_group['base_type_probability'] > base_prob_threshold) & \
            (base_type_group['detailed_type_probability'] < detailed_prob_threshold)

    def _save_barcode(self, sample, barcode_path):
        # Save barcodes to CSV file
        barcode = pd.DataFrame(sample.index.tolist(), columns=["barcode"])
        barcode.to_csv(barcode_path, index=False)

    def run_pipeline(self, input_file, ref_file, batch_size=128):
        """
        Run the complete cell type classification pipeline
        
        Parameters:
        -----------
        input_file : str
            Path to input h5ad file
        ref_file : str
            Path to reference RDS file
        batch_size : int
            Batch size for prediction
            
        Returns:
        --------
        pd.DataFrame
            Prediction results
        """
        print(f"Starting pipeline with input: {input_file}")
        
        # Step 1: Perform batch correction
        print("Step 1: Batch correction")
        corrected_file = self.batch_correction(input_file, ref_file)
        
        # Step 2: Transform gene expression to images
        print("Step 2: Image transformation")
        self.image_transform(corrected_file)
        
        # Step 3: Predict cell types
        print("Step 3: Cell type prediction")
        results = self.predict(batch_size)
        
        print("Pipeline completed successfully!")
        return results