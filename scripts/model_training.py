from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import warnings

warnings.simplefilter('ignore')
global batch_size
batch_size = 256
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p = pd.read_csv("/home/ubuntu/shangru/files/trainy.csv", index_col=0)
p["base_type"].value_counts()
base_types = p["base_type"].value_counts().index.tolist()
global index
types = []
index = []
for type in base_types:
    p1 = p[p["base_type"] == type]
    subtype_counts = p1["subtype"].value_counts()
    valid_subtypes = subtype_counts[subtype_counts > 0].index.tolist()
    length = len(valid_subtypes)
    index.append(length)
    types.extend(valid_subtypes)
print(types)
print(index)
if not os.path.exists("/home/ubuntu/shangru/train_files"):
    os.mkdir("/home/ubuntu/shangru/train_files")
le_base = preprocessing.LabelEncoder()
le_base.fit(base_types)
le_base.classes_ = np.array(base_types)
classes_base = len(np.unique(base_types))
with open("/home/ubuntu/shangru/train_files/label_encoder_base.obj","wb") as f:
   pickle.dump(le_base, f)
le = preprocessing.LabelEncoder()
le.fit(types)
le.classes_ = np.array(types)
classes = len(np.unique(types))
with open("/home/ubuntu/shangru/train_files/label_encoder.obj","wb") as f:
   pickle.dump(le, f)
q = pd.read_csv("/home/ubuntu/shangru/files/valy.csv", index_col=0)
y_train = pd.DataFrame(le.transform(p['subtype']))
y_val = pd.DataFrame(le.transform(q['subtype']))
np.save('/home/ubuntu/shangru/train_files/train_label.npy', y_train)
np.save('/home/ubuntu/shangru/train_files/val_label.npy', y_val)
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)
class EarlyStopping:
   def __init__(self, patience=10, verbose=False, path='/home/ubuntu/shangru/train_files/checkpoint_model_scHDeepInsight.pth'): 
       self.patience = patience    
       self.verbose = verbose      
       self.counter = 0            
       self.best_score = None      
       self.early_stop = False     
       self.val_acc_max = 0   
       self.path = path
   def __call__(self, val_acc, model, epoch, optimizer, scheduler):
       score = val_acc
       if self.best_score is None:
           self.best_score = score
           self.checkpoint(val_acc, model, epoch, optimizer, scheduler)
       elif score > self.best_score: 
           self.best_score = score
           self.checkpoint(val_acc, model, epoch, optimizer, scheduler)
           self.counter = 0 
       else:
           self.counter += 1  
           if self.counter >= self.patience:
               self.early_stop = True
       if self.verbose:
           print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
       return self.early_stop
   def checkpoint(self, val_acc, model, epoch, optimizer, scheduler):
       if self.verbose:
           print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
       torch.save({
           'epoch': epoch + 1,
           'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
           'optimizer': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
       }, self.path)
       self.val_acc_max = val_acc
class MyDataset(Dataset):
    def __init__(self, img, label):
        self.img = np.load(img, allow_pickle=True)
        self.label = torch.tensor(np.load(label, allow_pickle=True))
        self.transforms = transforms.Compose([transforms.ToTensor(), ])
    def __getitem__(self, index):
        img = self.img[index, :, :, :] 
        img = np.squeeze(img)
        img = Image.fromarray(np.uint8(img))
        img = self.transforms(img)
        label = self.label[index]
        label = np.squeeze(label)
        return img,label
    def __len__(self):
        return self.img.shape[0]
def mask(img, num):
    """Apply random masking to images."""
    curr_batch_size = len(img)
    if curr_batch_size == 0:
        return img
    device = img.device
    masked = img.clone()
    x = torch.randint(0, img.size(-2), (num,), device=device)
    y = torch.randint(0, img.size(-1), (num,), device=device)
    for i in range(curr_batch_size):
        for xi, yi in zip(x, y):
            masked[i, :, xi, yi] = 0
    return masked
def determine_batch_size():
    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        suggested_batch = int(gpu_mem * 0.4 / (224 * 224 * 3 * 4))  # 40% of GPU memory
        return min(max(32, suggested_batch), 256)  # Bound between 32 and 256
    except:
        return 128  # Default fallback
class HierarchicalNet(nn.Module):
    def __init__(self, num_base_classes, num_detailed_classes, index):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5')
        backbone_out = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.base_classifier = nn.Sequential(
            nn.Linear(backbone_out, backbone_out // 2),
            nn.BatchNorm1d(backbone_out // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(backbone_out // 2, num_base_classes)
        )
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
        features = self.backbone(x)
        base_logits = self.base_classifier(features)
        detailed_logits_list = []
        for classifier in self.detailed_classifiers:
            detailed_logits_list.append(classifier(features))
        return base_logits, detailed_logits_list
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, input, target):
        """
        Calculate Focal Loss
        Parameters:
        input: model output logits, shape [N, C]
        target: target class indices, shape [N]
        """
        ce_loss = F.cross_entropy(
            input, target, weight=self.weight, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class HierarchicalLoss:
    def __init__(self, num_classes, index, alpha_init=0.5, beta=0.9, 
                 use_weights=True, gamma=2.0, use_focal_loss=True, device=None):
        self.alpha = alpha_init
        self.beta = beta  # Smoothing coefficient
        self.index = index
        self.alpha_history = []  # Record alpha history
        self.gamma = gamma  # Focal Loss parameter
        self.use_focal_loss = use_focal_loss
        self.use_weights = use_weights
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_weights:
            self.base_weights, self.detailed_weights = self._compute_weights(num_classes, index)
            if self.base_weights is not None:
                self.base_weights = self.base_weights.to(self.device)
            self.detailed_weights = [w.to(self.device) if w is not None else None for w in self.detailed_weights]
        else:
            self.base_weights, self.detailed_weights = None, [None] * len(index)
        if use_focal_loss:
            self.base_ce = FocalLoss(gamma=gamma, weight=self.base_weights)
            self.detailed_ce = [FocalLoss(gamma=gamma) for _ in range(len(index))]
        else:
            self.base_ce = nn.CrossEntropyLoss(weight=self.base_weights)
            self.detailed_ce = [nn.CrossEntropyLoss() for _ in range(len(index))]
    def _compute_weights(self, num_classes, index):
        """
        Calculate weights for base types and detailed types using log transformation to moderate extreme weights
        """
        base_counts = torch.tensor(index, dtype=torch.float)
        base_weights = 1.0 / torch.log1p(base_counts)  # log1p = log(1+x)
        base_weights = base_weights / base_weights.sum()  # Normalize
        detailed_weights_list = []
        for i, count in enumerate(index):
            if count > 0:
                detailed_weights = torch.ones(count, dtype=torch.float)
                detailed_weights = detailed_weights / detailed_weights.sum()  # Normalize
                detailed_weights_list.append(detailed_weights)
            else:
                detailed_weights_list.append(None)
        return base_weights, detailed_weights_list
    def _compute_detailed_loss(self, detailed_logits_list, base_targets, detailed_targets):
        total_loss = 0
        for i, (logits, weight) in enumerate(zip(detailed_logits_list, self.detailed_weights)):
            mask = base_targets == i
            if mask.any():
                curr_targets = detailed_targets[mask]
                curr_targets = curr_targets - sum(self.index[:i])
                if self.use_focal_loss:
                    curr_loss = self.detailed_ce[i](logits[mask], curr_targets)
                else:
                    if weight is not None:
                        weight_tensor = weight.to(curr_targets.device)
                        curr_loss = F.cross_entropy(logits[mask], curr_targets, weight=weight_tensor)
                    else:
                        curr_loss = F.cross_entropy(logits[mask], curr_targets)
                total_loss += curr_loss
        return total_loss
    def forward(self, base_logits, detailed_logits_list, base_targets, detailed_targets, is_validation=False):
        base_loss = self.base_ce(base_logits, base_targets)
        detailed_loss = self._compute_detailed_loss(detailed_logits_list, base_targets, detailed_targets)
        if is_validation:
            return 0.5 * base_loss + 0.5 * detailed_loss, base_loss, detailed_loss
        else:
            return self.alpha * base_loss + (1 - self.alpha) * detailed_loss, base_loss, detailed_loss
    def update_alpha(self, base_loss, detailed_loss):
        """Update alpha using exponential smoothing"""
        alpha_new = base_loss / (base_loss + detailed_loss)
        self.alpha = self.beta * self.alpha + (1 - self.beta) * alpha_new
        self.alpha = min(max(self.alpha, 0.1), 0.9)
        self.alpha_history.append(self.alpha)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
def find_base_type(detailed, index):
    base_type = detailed.clone()
    cumulative_index = [0] + [sum(index[:i+1]) for i in range(len(index))]
    for i in range(len(base_type)):
        for j in range(1, len(cumulative_index)):
            if base_type[i] < cumulative_index[j]:
                base_type[i] = j - 1
                break
    return base_type
def sum_base_type_tensor(data, index):
    base_type_tensor = torch.sum(data[:, 0:sum(index[0:1])], dim=1).expand(1, -1)
    for i in range(1, len(index)):
        k1 = sum(index[0:i])
        k2 = sum(index[0:i+1])
        base_type_tensor = torch.cat((base_type_tensor, torch.sum(data[:, k1:k2], dim=1).expand(1, -1)), dim=0)
    return torch.t(base_type_tensor)
def sub_predicted(output, predicted_base_type, index):
    sub_tensor = output.clone()
    for i in range(len(sub_tensor)):
        base_type = predicted_base_type[i]
        k1 = sum(index[:base_type])
        k2 = sum(index[:base_type + 1])
        sub_tensor[i, :k1] = 0
        sub_tensor[i, k2:] = 0
    return sub_tensor
def create_data_loader(dataset, batch_size, is_train):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True
    )
def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_base_loss = 0
    total_detailed_loss = 0
    base_correct = 0
    detailed_correct = 0
    samples = 0
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device)
        detailed_targets = targets.to(device)
        base_targets = find_base_type(detailed_targets, index)
        base_logits, detailed_logits_list = model(data)
        loss, base_loss, detailed_loss = criterion(base_logits, detailed_logits_list, 
                                                 base_targets, detailed_targets, 
                                                 is_validation=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, base_preds = torch.max(base_logits, 1)
        base_correct += (base_preds == base_targets).sum().item()
        detailed_preds = []
        detailed_targets_list = []
        for i, logits in enumerate(detailed_logits_list):
            mask_indices = base_preds == i
            if mask_indices.any():
                _, pred = torch.max(logits[mask_indices], 1)
                curr_targets = detailed_targets[mask_indices] - sum(index[:i])
                detailed_preds.extend(pred.tolist())
                detailed_targets_list.extend(curr_targets.tolist())
        detailed_correct += sum(p == t for p, t in zip(detailed_preds, detailed_targets_list))
        samples += targets.size(0)
        total_loss += loss.item()
        total_base_loss += base_loss.item()
        total_detailed_loss += detailed_loss.item()
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss {loss.item():.4f}, Alpha {criterion.alpha:.4f}')
    metrics = {
        'loss': total_loss / len(loader),
        'base_loss': total_base_loss / len(loader),
        'detailed_loss': total_detailed_loss / len(loader),
        'base_acc': 100 * base_correct / samples,
        'detailed_acc': 100 * detailed_correct / samples,
        'alpha': criterion.alpha
    }
    return metrics
def validate(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    total_base_loss = 0
    total_detailed_loss = 0
    base_correct = 0
    detailed_correct = 0
    samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            detailed_targets = targets.to(device)
            base_targets = find_base_type(detailed_targets, index)
            base_logits, detailed_logits_list = model(data)
            loss, base_loss, detailed_loss = criterion(base_logits, detailed_logits_list, 
                                                     base_targets, detailed_targets,
                                                     is_validation=True)
            _, base_preds = torch.max(base_logits, 1)
            base_correct += (base_preds == base_targets).sum().item()
            detailed_preds = []
            detailed_targets_list = []
            for i, logits in enumerate(detailed_logits_list):
                mask_indices = base_preds == i
                if mask_indices.any():
                    _, pred = torch.max(logits[mask_indices], 1)
                    curr_targets = detailed_targets[mask_indices] - sum(index[:i])
                    detailed_preds.extend(pred.tolist())
                    detailed_targets_list.extend(curr_targets.tolist())
            detailed_correct += sum(p == t for p, t in zip(detailed_preds, detailed_targets_list))
            samples += targets.size(0)
            total_loss += loss.item()
            total_base_loss += base_loss.item()
            total_detailed_loss += detailed_loss.item()
    criterion.update_alpha(total_base_loss / len(loader), total_detailed_loss / len(loader))
    return {
        'loss': total_loss / len(loader),
        'base_loss': total_base_loss / len(loader),
        'detailed_loss': total_detailed_loss / len(loader),
        'base_acc': 100 * base_correct / samples,
        'detailed_acc': 100 * detailed_correct / samples,
        'alpha': criterion.alpha
    }
torch.manual_seed(42)
np.random.seed(42)
print("Loading datasets...")
train_dataset = MyDataset(
    "/home/ubuntu/shangru/files/train.npy",
    "/home/ubuntu/shangru/train_files/train_label.npy"
)
val_dataset = MyDataset(
    "/home/ubuntu/shangru/files/val.npy", 
    "/home/ubuntu/shangru/train_files/val_label.npy"
)
print("Initializing model...")
model = HierarchicalNet(len(base_types), classes, index)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
criterion = HierarchicalLoss(
    num_classes=classes, 
    index=index, 
    alpha_init=0.5, 
    beta=0.9,
    use_weights=False,  # Use class weights
    gamma=2.0,         # Focal Loss parameter
    use_focal_loss=True,  # Enable Focal Loss
    device=device
)
if criterion.base_weights is not None:
    criterion.base_weights = criterion.base_weights.to(device)
    if criterion.use_focal_loss:
        criterion.base_ce.weight = criterion.base_weights
for i, weights in enumerate(criterion.detailed_weights):
    if weights is not None:
        criterion.detailed_weights[i] = weights.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,               # Optimizer whose learning rate is adjusted
    mode='max',              # 'max': Monitor metric should increase; 'min': Monitor metric should decrease
    factor=0.8,              # Factor by which learning rate will be reduced: lr = lr * factor
    patience=20,             # Number of epochs with no improvement after which LR will be reduced
    verbose=True,            # If True, prints a message when LR is reduced
    threshold=0.005,         # Threshold for measuring improvement
    threshold_mode='rel',    # 'rel': Relative improvement; 'abs': Absolute improvement
    min_lr=1e-6              # Lower bound on the learning rate
)
batch_size = determine_batch_size()
train_loader = create_data_loader(train_dataset, batch_size, True)
val_loader = create_data_loader(val_dataset, batch_size, False)
early_stopper = EarlyStopping(
    patience=40,
    path='/home/ubuntu/shangru/train_files/checkpoint_model.pth'
)
start_epoch = 0
if os.path.exists(early_stopper.path):
    print("Loading checkpoint...")
    checkpoint = torch.load(early_stopper.path)
    state_dict = checkpoint['model']
    if not isinstance(model, nn.DataParallel) and all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    elif isinstance(model, nn.DataParallel) and not all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {start_epoch}")
log_file = open("/home/ubuntu/shangru/train_files/training_log.txt", "w")
history = {'train': [], 'val': [], 'alpha': [], 'early_stopping_counter': []}
try:
    for epoch in range(start_epoch, 150):
        print(f"Training with Focal Loss (gamma={criterion.gamma}) and log-adjusted weights: {criterion.use_weights}")
        print(f"Initial alpha: {criterion.alpha}, beta: {criterion.beta}")
        print(f'\nEpoch {epoch + 1}')
        train_metrics = train_epoch(train_loader, model, criterion, 
                                    optimizer, device)
        val_metrics = validate(val_loader, model, criterion, device)
        scheduler.step(val_metrics['detailed_acc'])
        history['alpha'].append(criterion.alpha)
        history['early_stopping_counter'].append(early_stopper.counter)
        for phase, metrics in [('train', train_metrics), ('val', val_metrics)]:
            history[phase].append(metrics)
            log_str = f'{phase.upper()}: Loss={metrics["loss"]:.4f}, '
            log_str += f'Base_Acc={metrics["base_acc"]:.2f}, '
            log_str += f'Detailed_Acc={metrics["detailed_acc"]:.2f}, '
            log_str += f'Alpha={metrics["alpha"]:.4f}'
            print(log_str)
            log_file.write(f'Epoch {epoch + 1}: {log_str}\n')
        es_str = f'Early stopping counter: {early_stopper.counter}'
        print(es_str)
        log_file.write(f'Epoch {epoch + 1}: {es_str}\n')
        log_file.flush()
        if early_stopper(val_metrics['detailed_acc'], model, epoch, optimizer, scheduler):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nError during training: {str(e)}")
finally:
    log_file.close()
def plot_all_metrics(history):
    epochs = range(1, len(history['train']) + 1)
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'Training Metrics (Focal Loss: {criterion.use_focal_loss}, Gamma: {criterion.gamma})', fontsize=15)
    plt.subplot(2, 2, 1)
    for phase in ['train', 'val']:
        plt.plot(epochs, [m['base_acc'] for m in history[phase]], 
                label=f'{phase.capitalize()} base')
        plt.plot(epochs, [m['detailed_acc'] for m in history[phase]], 
                label=f'{phase.capitalize()} Detailed')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.subplot(2, 2, 2)
    for phase in ['train', 'val']:
        plt.plot(epochs, [m['loss'] for m in history[phase]], 
                label=phase.capitalize())
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['alpha'], label='Alpha', color='green')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Alpha Value')
    plt.title('Loss Weight Alpha Values')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['early_stopping_counter'], label='Counter', color='red')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Counter Value')
    plt.title('Early Stopping Counter')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/ubuntu/shangru/train_files/training_curves.png', dpi=600, bbox_inches='tight')
    plt.close()
plot_all_metrics(history)