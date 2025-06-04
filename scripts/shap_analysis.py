from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import shap
import torch
import traceback
import warnings

warnings.simplefilter('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def clear_gpu_memory():
    """Explicitly clean GPU memory and print current usage"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB / {torch.cuda.memory_reserved(i)/1024**3:.2f}GB")
def clean_type_name(name):
    """Remove special characters from type names"""
    return name.replace('/', '')
class MyDataset(Dataset):
    def __init__(self, img, label):
        self.img = np.load(img, allow_pickle=True)
        self.label = torch.tensor(np.load(label, allow_pickle=True))
        self.transforms = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        img = self.img[index, :, :, :] 
        img = np.squeeze(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transforms(img)
        label = self.label[index]
        label = np.squeeze(label)
        return img, label
    def __len__(self):
        return self.img.shape[0]
class HierarchicalNet(torch.nn.Module):
    def __init__(self, num_base_classes, num_detailed_classes, index):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5')
        backbone_out = self.backbone._fc.in_features
        self.backbone._fc = torch.nn.Identity()
        self.base_classifier = torch.nn.Sequential(
            torch.nn.Linear(backbone_out, backbone_out // 2),
            torch.nn.BatchNorm1d(backbone_out // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(backbone_out // 2, num_base_classes)
        )
        self.detailed_classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(backbone_out, backbone_out // 2),
                torch.nn.BatchNorm1d(backbone_out // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(backbone_out // 2, backbone_out // 4),
                torch.nn.BatchNorm1d(backbone_out // 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(backbone_out // 4, size)
            ) for size in index
        ]) 
        self.index = index
    def forward(self, x):
        features = self.backbone(x)
        base_logits = self.base_classifier(features)
        detailed_logits_list = []
        for classifier in self.detailed_classifiers:
            detailed_logits_list.append(classifier(features))
        return base_logits, detailed_logits_list
def find_base_type(detailed, index):
    """
    Convert detailed class labels to base type labels
    Args:
        detailed (torch.Tensor): Detailed class labels
        index (list): Number of classes in each base type
    Returns:
        torch.Tensor: Base type labels
    """
    base_type = detailed.clone()
    cumulative_index = [0] + [sum(index[:i+1]) for i in range(len(index))]
    for i in range(len(base_type)):
        for j in range(1, len(cumulative_index)):
            if base_type[i] < cumulative_index[j]:
                base_type[i] = j - 1
                break
    return base_type
class HierarchicalModelWrapper(torch.nn.Module):
    def __init__(self, model, prediction_level='base'):
        super().__init__()
        self.model = model
        self.prediction_level = prediction_level
    def forward(self, x):
        base_logits, detailed_logits_list = self.model(x)
        if self.prediction_level == 'base':
            return base_logits
        else:
            return detailed_logits_list[0]  # Simplified handling
def load_model_for_shap(model_path, base_types, classes, index):
    """
    Load trained model for SHAP analysis with improved multi-GPU support
    Args:
        model_path (str): Path to saved model checkpoint
        base_types (list): List of base types
        classes (int): Total number of detailed classes
        index (list): Number of classes in each base type
    Returns:
        torch.nn.Module: Loaded and prepared model
    """
    model = HierarchicalNet(len(base_types), classes, index)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['model']
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    available_gpus = torch.cuda.device_count()
    if available_gpus >= 2:
        print(f"Using {available_gpus} GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(available_gpus)))
        model = model.to('cuda:0')
    else:
        print(f"Only {available_gpus} GPU available")
        model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model
def get_coordinates_from_pixel_index(idx, width=224):
    """
    Convert flattened pixel index to 2D coordinates
    Args:
        idx (int): Flattened pixel index
        width (int): Image width
    Returns:
        tuple: (y, x) coordinates
    """
    y = idx // width
    x = idx % width
    return y, x
def ensure_shap_values_list(shap_values, base_classes_count=15):
    """
    Process SHAP values, keeping information for each class, averaging over channels
    Args:
        shap_values (numpy.ndarray or list): SHAP values to process
        base_classes_count (int): Maximum number of base classes to process
    Returns:
        list: Processed SHAP values for each class
    """
    print("SHAP values details:")
    print(f"Type: {type(shap_values)}")
    print(f"Dimensions: {shap_values.ndim if hasattr(shap_values, 'ndim') else 'N/A'}")
    print(f"Shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5:  # (samples, channels, height, width, classes)
            print("Processing 5D array")
            avg_over_samples_channels = np.mean(shap_values, axis=(0, 1))  # Result is (224, 224, 15)
            return [avg_over_samples_channels[:, :, i] for i in range(min(base_classes_count, avg_over_samples_channels.shape[2]))]
        elif shap_values.ndim == 4:  # Possibly (samples, height, width, classes) or (channels, height, width, classes)
            print("Processing 4D array")
            avg_over_first_dim = np.mean(shap_values, axis=0)
            if avg_over_first_dim.ndim == 3:
                return [avg_over_first_dim[:, :, i] for i in range(min(base_classes_count, avg_over_first_dim.shape[2]))]
            else:
                return [avg_over_first_dim]
        elif shap_values.ndim == 3:  # Possibly (height, width, classes)
            print("Processing 3D array")
            if shap_values.shape[2] >= base_classes_count:
                return [shap_values[:, :, i] for i in range(min(base_classes_count, shap_values.shape[2]))]
            else:
                return [np.mean(shap_values, axis=0)]
        elif shap_values.ndim == 2:  # (height, width)
            print("Processing 2D array")
            return [shap_values]
        elif shap_values.ndim == 1:  # (pixels)
            print("Processing 1D array - needs reshaping")
            reshaped = shap_values.reshape(224, 224)
            return [reshaped]
    if isinstance(shap_values, list):
        processed_list = []
        for item in shap_values:
            if isinstance(item, np.ndarray):
                if item.ndim == 1:
                    processed_list.append(item.reshape(224, 224))
                else:
                    processed_list.append(item)
            else:
                processed_list.append(item)
        return processed_list
    raise ValueError(f"Cannot convert SHAP values. Type: {type(shap_values)}, Shape: {getattr(shap_values, 'shape', 'N/A')}")
def combine_shap_values(shap_values_list):
    """
    Combine SHAP values from multiple batches
    Args:
        shap_values_list (list): List of SHAP values from different batches
    Returns:
        numpy.ndarray or list: Combined SHAP values
    """
    if not shap_values_list:
        return None
    if all(isinstance(sv, list) for sv in shap_values_list):
        combined = []
        n_classes = len(shap_values_list[0])
        for class_idx in range(n_classes):
            class_values = [batch[class_idx] for batch in shap_values_list if class_idx < len(batch)]
            if class_values:
                combined.append(np.mean(np.stack(class_values, axis=0), axis=0))
        return combined
    elif all(isinstance(sv, np.ndarray) for sv in shap_values_list):
        return np.mean(np.stack(shap_values_list, axis=0), axis=0)
    else:
        raise ValueError("Cannot combine SHAP values with mixed or unknown formats")
def analyze_important_pixels_by_cell_type(model, dataset, base_types, index, samples_per_type=3500, save_dir="./sch_immune/pixel_importance_analysis"):
    """
    Analyze important pixels by cell type with optimized batch processing
    Args:
        model (torch.nn.Module): Trained model
        dataset (torch.utils.data.Dataset): Validation dataset
        base_types (list): List of base cell types
        index (list): Number of classes in each base type
        samples_per_type (int): Maximum samples to process per type
        save_dir (str): Directory to save analysis results
    Returns:
        dict: Detailed analysis of important pixels for each cell type
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Loading data in batches...")
    batch_size = 1000  # Batch size for loading data
    all_images = []
    all_labels = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_images, batch_labels in dataloader:
        all_images.append(batch_images)
        all_labels.append(batch_labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Loaded {len(all_images)} samples")
    all_base_labels = find_base_type(all_labels, index)
    samples_by_type = {}
    for i in range(len(all_base_labels)):
        base_idx = all_base_labels[i].item()
        if base_idx < len(base_types):
            base_type = base_types[base_idx]
            if base_type not in samples_by_type:
                samples_by_type[base_type] = []
            samples_by_type[base_type].append(i)
    for base_type, indices in samples_by_type.items():
        print(f"Type '{base_type}' has {len(indices)} samples")
    important_pixels = {}
    for base_type in base_types:
        important_pixels[base_type] = {
            'avg_shap_map': np.zeros((224, 224)),
            'top_indices': [],
            'top_coords': [],
            'top_values': [],
            'bottom_indices': [],
            'bottom_coords': [],
            'bottom_values': []
        }
    n_background = 100   # Reduced number of background samples
    batch_size = 210    # Smaller batch size to save memory
    priority_types = []
    all_cell_types = list(samples_by_type.keys())
    remaining_types = [t for t in all_cell_types if t not in priority_types]
    ordered_cell_types = priority_types + remaining_types
    print("Cell types will be processed in this order:")
    for i, cell_type in enumerate(ordered_cell_types):
        print(f"{i+1}. {cell_type} ({len(samples_by_type[cell_type])} samples)")
    for base_type in ordered_cell_types:
        sample_indices = samples_by_type[base_type]
        print(f"Processing cell type: {base_type}, total samples: {len(sample_indices)}")
        base_type_idx = base_types.index(base_type)
        n_bg = min(n_background, len(sample_indices))
        background_indices = sample_indices[:n_bg]
        background = all_images[background_indices].to(device)
        print(f"  - Using {n_bg} samples as background")
        clear_gpu_memory()
        wrapped_model = HierarchicalModelWrapper(model, prediction_level='base')
        explainer = shap.GradientExplainer(wrapped_model, background)
        remaining_indices = sample_indices[n_bg:]
        if len(remaining_indices) > samples_per_type - n_bg:
            print(f"  - Sampling {samples_per_type - n_bg} from {len(remaining_indices)} remaining samples")
            np.random.shuffle(remaining_indices)
            remaining_indices = remaining_indices[:(samples_per_type - n_bg)]
        all_batch_shap_values = []
        if len(remaining_indices) > 0:
            num_batches = (len(remaining_indices) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {base_type} samples", total=num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(remaining_indices))
                batch_indices = remaining_indices[start_idx:end_idx]
                batch = all_images[batch_indices].to(device)
                print(f"  - Processing batch {batch_idx+1}/{num_batches}, samples: {len(batch)}")
                try:
                    batch_shap_values = explainer.shap_values(batch)
                    all_batch_shap_values.append(batch_shap_values)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"  - Out of memory in batch {batch_idx+1}, retrying with smaller batch")
                        smaller_batch_size = max(1, len(batch) // 2)
                        for mini_idx in range(0, len(batch), smaller_batch_size):
                            mini_end = min(mini_idx + smaller_batch_size, len(batch))
                            mini_batch = batch[mini_idx:mini_end]
                            clear_gpu_memory()
                            try:
                                mini_batch_shap_values = explainer.shap_values(mini_batch)
                                all_batch_shap_values.append(mini_batch_shap_values)
                            except Exception as e2:
                                print(f"  - Error in mini-batch: {e2}")
                                for single_idx in range(mini_idx, mini_end):
                                    try:
                                        clear_gpu_memory()
                                        single_sample = batch[single_idx:single_idx+1]
                                        single_shap_values = explainer.shap_values(single_sample)
                                        all_batch_shap_values.append(single_shap_values)
                                    except Exception as e3:
                                        print(f"  - Skipping sample {single_idx} due to error: {e3}")
                    else:
                        print(f"  - Error in batch {batch_idx+1}: {e}")
                del batch
                clear_gpu_memory()
            print(f"  - Combining results from {len(all_batch_shap_values)} batches")
            try:
                shap_values = combine_shap_values(all_batch_shap_values)
            except Exception as e:
                print(f"  - Error combining SHAP values: {e}")
                if all_batch_shap_values:
                    shap_values = ensure_shap_values_list(all_batch_shap_values[0])
                else:
                    shap_values = [np.zeros((224, 224)) for _ in range(len(base_types))]
        else:
            print("  - No remaining samples, using background samples for analysis")
            shap_values = explainer.shap_values(background)
        shap_values = ensure_shap_values_list(shap_values)
        if base_type_idx < len(shap_values):
            base_shap = shap_values[base_type_idx]
        else:
            print(f"Warning: base_type_idx ({base_type_idx}) exceeds shap_values range ({len(shap_values)})")
            base_shap = shap_values[0]
        if isinstance(base_shap, np.ndarray):
            if base_shap.ndim == 1:
                try:
                    base_shap = base_shap.reshape(224, 224)
                except ValueError:
                    print(f"Warning: Cannot reshape array of shape {base_shap.shape} to (224, 224)")
                    base_shap = np.zeros((224, 224))
            elif base_shap.ndim == 3:
                base_shap = np.mean(base_shap, axis=0)
            elif base_shap.ndim != 2:
                print(f"Warning: base_shap dimensions ({base_shap.ndim}) not expected 2D, shape: {base_shap.shape}")
                if base_shap.size > 0:
                    base_shap = np.mean(base_shap.reshape(-1, 224, 224), axis=0)
                else:
                    base_shap = np.zeros((224, 224))
        else:
            print(f"Warning: base_shap is not a numpy array")
            base_shap = np.zeros((224, 224))
        avg_shap = base_shap
        if avg_shap.ndim != 2:
            print(f"Warning: avg_shap dimensions ({avg_shap.ndim}) not 2D, attempting conversion")
            if avg_shap.size > 0:
                try:
                    avg_shap = avg_shap.reshape(224, 224)
                except ValueError:
                    print(f"Cannot reshape array of shape {avg_shap.shape} to (224, 224)")
                    avg_shap = np.zeros((224, 224))
            else:
                avg_shap = np.zeros((224, 224))
        flat_shap_abs = np.abs(avg_shap).flatten()
        top_indices = np.argsort(flat_shap_abs)[-50:][::-1]
        bottom_indices = np.argsort(flat_shap_abs)[:50]
        image_width = 224
        top_coords = [get_coordinates_from_pixel_index(idx, image_width) for idx in top_indices]
        top_values = avg_shap.flatten()[top_indices]
        bottom_coords = [get_coordinates_from_pixel_index(idx, image_width) for idx in bottom_indices]
        bottom_values = avg_shap.flatten()[bottom_indices]
        important_pixels[base_type] = {
            'top_indices': top_indices,
            'top_coords': top_coords,
            'top_values': top_values,
            'bottom_indices': bottom_indices,
            'bottom_coords': bottom_coords,
            'bottom_values': bottom_values,
            'avg_shap_map': avg_shap
        }
        with open(f"{save_dir}/{base_type}_shap_values.pkl", 'wb') as f:
            pickle.dump(important_pixels[base_type], f)
        clear_gpu_memory()
    plt.figure(figsize=(20, 15))
    for i, type_name in enumerate(base_types):
        plt.subplot(3, 5, i+1)
        shap_map = important_pixels[type_name]['avg_shap_map']
        vmax = np.max(np.abs(shap_map))
        plt.imshow(shap_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(type_name)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_base_types_shap_maps.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(20, 15))
    for i, type_name in enumerate(base_types):
        plt.subplot(3, 5, i+1)
        plt.imshow(np.abs(important_pixels[type_name]['avg_shap_map']), cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"{type_name} (Absolute)")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_base_types_shap_maps_absolute.png", dpi=300, bbox_inches='tight')
    plt.close()
    with open(f"{save_dir}/detailed_shap_values.pkl", 'wb') as f:
        pickle.dump(important_pixels, f)
    return important_pixels
if __name__ == "__main__":
    try:
        save_dir = "./sch_immune/shap_files/pixel_importance_analysis"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        p = pd.read_csv("./sch_immune/shap_files/trainy.csv", index_col=0)
        base_types = [clean_type_name(type) for type in p["base_type"].value_counts().index.tolist()]
        types = []
        index = []
        for cell_type in p["base_type"].value_counts().index.tolist():
            p1 = p[p["base_type"] == cell_type]
            subtype_counts = p1["subtype"].value_counts()
            valid_subtypes = [clean_type_name(subtype) for subtype in subtype_counts[subtype_counts > 0].index.tolist()]
            length = len(valid_subtypes)
            index.append(length)
            types.extend(valid_subtypes)
        print("Base types:", base_types)
        print("Base types count:", len(base_types))
        print("Detailed types count:", len(types))
        print("Index details:", index)
        val_dataset = MyDataset(
            "./sch_immune/shap_files/remaining_images.npy", 
            "./sch_immune/shap_files/remaining_labels.npy"
        )
        model_path = "/Usersdata/shangru/docker/sch_immune/train_files_cosine/checkpoint_model.pth"
        model = load_model_for_shap(model_path, base_types, len(types), index)
        important_pixels = analyze_important_pixels_by_cell_type(
            model, val_dataset, base_types, index, samples_per_type=3500,
            save_dir=save_dir
        )
        print(f"Analysis completed and saved to {save_dir}")
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()