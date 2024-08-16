import os
from pathlib import Path
from pyDeepInsight import ImageTransformer
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from efficientnet_pytorch import EfficientNet
import pickle
from sklearn import preprocessing
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

class Immune:
    def __init__(self):
        self.pretrained_dir = Path(__file__).resolve().parent / "pretrained_files_immune"
        self.gene_list = self._load_gene_list()
        self.img_transformer = self._load_img_transformer()
        self.index = [8, 2, 11, 2, 2, 4, 3, 1, 5, 3, 3, 1, 2, 2, 1]
        self.model = self._load_model()

    def _load_gene_list(self):
        gene_list_path = self.pretrained_dir / "pretrained_genes_immune.csv"
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list file not found at {gene_list_path}")
        return pd.read_csv(gene_list_path, index_col=0).index.tolist()

    def _load_img_transformer(self):
        transformer_path = self.pretrained_dir / "img_transformer_immune.obj"
        if not transformer_path.exists():
            raise FileNotFoundError(f"Image transformer file not found at {transformer_path}")
        with open(transformer_path, 'rb') as file:
            return pickle.load(file)

    def _load_model(self):
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=50)
        model = nn.DataParallel(model)
        checkpoint_path = self.pretrained_dir / "checkpoint_model_immune.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=False)
        return model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).eval()

    def preprocess(self, query_path: str, output_path: str):
        """Performs normalization and log1p transformation on the input .h5ad file."""
        query = anndata.read_h5ad(query_path)
        sc.pp.normalize_per_cell(query)
        sc.pp.log1p(query)
        query.write(output_path)  # Save the preprocessed .h5ad file
        return query

    def image_transform(self, query_path: str, barcode_path: str, image_path: str):
        """Transforms the .h5ad file into a DataFrame and then into images."""
        query = anndata.read_h5ad(query_path)
        query.var["feature_name"] = query.var.get("feature_name", query.var.index.tolist())
        query.var.index = query.var["feature_name"].values

        remain_list = list(set(query.var.index) & set(self.gene_list))
        query = query[:, remain_list]

        sample = self._scale_and_fill(query)
        self._save_barcode(sample, barcode_path)
        self._save_image(sample, image_path)

    def predict(self, barcode_path: str, image_path: str, batch_size: int = 128):
        """Predicts cell types and identifies potential rare cells."""
        prefolder = os.path.join(os.getcwd(),'pretrained_files_immune')

        class MyTestSet(Dataset):
            def __init__(self, img):
                self.img = np.load(img)
                self.transforms = transforms.Compose([transforms.ToTensor(), ])
            def __getitem__(self, index):
                img = self.img[index, :, :, :]
                img = np.squeeze(img)
                img = Image.fromarray(np.uint8(img))
                img = self.transforms(img)
                return img
            def __len__(self):
                return self.img.shape[0]

        test_set = MyTestSet(image_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # Prioritize using GPUs to load the model.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        out_base, out_detailed, out_base_probs, out_detailed_probs = [], [], [], []

        for data in test_loader:
            query = data.to(device)
            pred = F.softmax(self.model(query), dim=1)

            base_tensor = self._sum_base_type_tensor(pred.data)
            base_probs, predicted_base_by_tree = torch.max(base_tensor, 1)

            output_sub = self._sub_predicted(pred.data, predicted_base_by_tree)
            detail_probs, predicted_detailed = torch.max(output_sub.data, 1)

            out_base.append(predicted_base_by_tree)
            out_detailed.append(predicted_detailed)
            out_base_probs.append(base_probs)
            out_detailed_probs.append(detail_probs)

        pred_base = torch.cat(out_base, dim=0)
        pred_detailed = torch.cat(out_detailed, dim=0)

        pred_base_probs = torch.cat(out_base_probs, dim=0).cpu().numpy()
        pred_detail_probs = torch.cat(out_detailed_probs, dim=0).cpu().numpy()

        file = open(Path(prefolder, "label_encoder_immune_base.obj"),'rb')
        le_base = pickle.load(file)
        file.close()
        pr_base = pred_base.cpu().numpy()
        pred_label_base = le_base.inverse_transform(pr_base)
        pred_label_base = pd.DataFrame(pred_label_base)
        pred_label_base = pred_label_base.rename(columns={0:"prediceted_base_type"})

        file = open(Path(prefolder, "label_encoder_immune_detailed.obj"),'rb')
        le_detailed = pickle.load(file)
        file.close()
        pr_detailed = pred_detailed.cpu().numpy()
        pred_label_detailed = le_detailed.inverse_transform(pr_detailed)
        pred_label_detailed = pd.DataFrame(pred_label_detailed)
        pred_label_detailed = pred_label_detailed.rename(columns={0:"prediceted_detailed_type"})

        # Create DataFrame with probabilities
        prob_analysis = pd.DataFrame({
            "prediceted_base_type": pr_base,
            "prediceted_detailed_type": pr_detailed,
            "prediceted_base_type_prob": pred_base_probs,
            "prediceted_detailed_type_prob": pred_detail_probs
        })

        # Determine potential rare cells
        prob_analysis['is_potential_rare'] = prob_analysis.groupby('prediceted_base_type', group_keys=False).apply(self._is_potential_rare)
        barcode = pd.read_csv(barcode_path)
        pred_label = pd.concat([barcode, pred_label_base, pred_label_detailed, prob_analysis[['prediceted_base_type_prob', 'prediceted_detailed_type_prob', 'is_potential_rare']]], axis=1)
        
        return pred_label

    def _scale_and_fill(self, query):
        sample = pd.DataFrame(query.X.toarray()).T
        sample = preprocessing.MinMaxScaler().fit_transform(sample)
        sample = pd.DataFrame(sample).T
        sample.index = query.obs.index.values
        sample.columns = query.var.index.values

        excluded_genes = list(set(self.gene_list) - set(sample.columns))
        blank_dataframe = pd.DataFrame(np.zeros((len(sample), len(excluded_genes))), 
                                       index=sample.index, columns=excluded_genes)
        sample = pd.concat([sample, blank_dataframe], axis=1)
        sample = sample[self.gene_list]
        return sample

    def _save_barcode(self, sample, barcode_path):
        barcode = pd.DataFrame(sample.index.tolist(), columns=["barcode"])
        barcode.to_csv(barcode_path, index=False)

    def _save_image(self, sample, image_path):
        query_img = (self.img_transformer.transform(sample) * 255).astype(np.uint8)
        np.save(image_path, query_img)

    def _sum_base_type_tensor(self, data):
        base_type_tensor = torch.sum(data[:, 0:self.index[0]], dim=1).expand(1, -1)
        for i in range(1, len(self.index)):
            k1 = sum(self.index[0:i])
            k2 = sum(self.index[0:i+1])
            base_type_tensor = torch.cat(
                (base_type_tensor, torch.sum(data[:, k1:k2], dim=1).expand(1, -1)), dim=0
            )
        return base_type_tensor.t()

    def _sub_predicted(self, output, predicted_base_type):
        sub_tensor = output.clone()
        for i in range(len(sub_tensor)):
            base_type = predicted_base_type[i]
            k1 = sum(self.index[0:base_type])
            k2 = sum(self.index[0:base_type + 1])
            sub_tensor[i, :k1] = 0
            sub_tensor[i, k2:] = 0
        return sub_tensor

    def _create_pred_label(self, barcode_path, out_base, out_detailed, out_base_probs, out_detailed_probs):
        pred_base = torch.cat(out_base).cpu().numpy()
        pred_detailed = torch.cat(out_detailed).cpu().numpy()
        pred_base_probs = torch.cat(out_base_probs).cpu().numpy()
        pred_detail_probs = torch.cat(out_detailed_probs).cpu().numpy()

        pred_label_base = self._decode_labels(pred_base, "label_encoder_immune_base.obj", "prediceted_base_type")
        pred_label_detailed = self._decode_labels(pred_detailed, "label_encoder_immune_detailed.obj", "prediceted_detailed_type")

        prob_analysis = pd.DataFrame({
            "prediceted_base_type": pred_base,
            "prediceted_detailed_type": pred_detailed,
            "prediceted_base_type_prob": pred_base_probs,
            "prediceted_detailed_type_prob": pred_detail_probs
        })

        barcode = pd.read_csv(barcode_path, index_col=0)
        return pd.concat([barcode, pred_label_base, pred_label_detailed, prob_analysis], axis=1)

    def _decode_labels(self, predictions, encoder_file, column_name):
        encoder_path = self.pretrained_dir / encoder_file
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
        with open(encoder_path, 'rb') as file:
            encoder = pickle.load(file)
        labels = encoder.inverse_transform(predictions)
        return pd.DataFrame(labels, columns=[column_name])

    def _is_potential_rare(self, base_type_group):
        base_prob_50th = np.percentile(base_type_group['prediceted_base_type_prob'], 50)
        detailed_prob_10th = np.percentile(base_type_group['prediceted_detailed_type_prob'], 10)
        return (base_type_group['prediceted_base_type_prob'] > base_prob_50th) & \
               (base_type_group['prediceted_detailed_type_prob'] < detailed_prob_10th)