# scDeepInsight training workflow

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from sklearn import preprocessing
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Custom Dataset class
class ImageDataset(Dataset):
   def __init__(self, img_path, label_path=None):
       self.img = np.load(img_path, allow_pickle=True)
       self.label = None
       if label_path:
           self.label = torch.tensor(np.load(label_path))
       self.transforms = transforms.Compose([transforms.ToTensor()])
   
   def __getitem__(self, index):
       img = self.img[index, :, :, :]
       img = np.squeeze(img)
       img = Image.fromarray(np.uint8(img))
       img = self.transforms(img)
       
       if self.label is not None:
           label = self.label[index]
           label = np.squeeze(label)
           return img, label
       return img
   
   def __len__(self):
       return self.img.shape[0]

# Label smoothing loss
class LabelSmoothingCrossEntropy(nn.Module):
   def __init__(self, eps=0.1, reduction='mean'):
       super().__init__()
       self.eps = eps
       self.reduction = reduction
   
   def forward(self, output, target):
       c = output.size()[-1]
       log_preds = torch.nn.functional.log_softmax(output, dim=-1)
       if self.reduction == 'sum':
           loss = -log_preds.sum()
       else:
           loss = -log_preds.sum(dim=-1)
           if self.reduction == 'mean':
               loss = loss.mean()
       return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(
           log_preds, target, reduction=self.reduction)

# Early stopping
class EarlyStopping:
   def __init__(self, patience=10, verbose=False, path='checkpoint.pth'):
       self.patience = patience
       self.verbose = verbose
       self.counter = 0
       self.best_score = None
       self.early_stop = False
       self.val_acc_max = 0
       self.path = path
   
   def __call__(self, val_acc, model):
       score = val_acc
       if self.best_score is None:
           self.best_score = score
           self.checkpoint(val_acc, model)
       elif score < self.best_score:
           self.counter += 1
           if self.verbose:
               print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_score = score
           self.checkpoint(val_acc, model)
           self.counter = 0
   
   def checkpoint(self, val_acc, model):
       if self.verbose:
           print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model...')
       torch.save(model.state_dict(), self.path)
       self.val_acc_max = val_acc

# Training function
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
               device, epochs=100, early_patience=40, checkpoint_path='checkpoint.pth'):
   
   earlystopping = EarlyStopping(patience=early_patience, verbose=True, path=checkpoint_path)
   
   for epoch in range(epochs):
       print(f'\nEpoch: {epoch + 1}')
       
       # Training phase
       model.train()
       train_loss = 0.0
       train_correct = 0
       train_total = 0
       
       for batch_idx, (inputs, targets) in enumerate(train_loader):
           inputs, targets = inputs.to(device), targets.to(device)
           
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, targets)
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           _, predicted = torch.max(outputs.data, 1)
           train_total += targets.size(0)
           train_correct += predicted.eq(targets.data).cpu().sum()
           
           if batch_idx % 50 == 0:
               acc = 100. * float(train_correct) / float(train_total)
               avg_loss = train_loss / (batch_idx + 1)
               print(f'[{epoch+1}, {batch_idx}] Loss: {avg_loss:.3f} | Acc: {acc:.3f}%')
       
       # Validation phase
       model.eval()
       val_loss = 0.0
       val_correct = 0
       val_total = 0
       
       with torch.no_grad():
           for inputs, targets in val_loader:
               inputs, targets = inputs.to(device), targets.to(device)
               outputs = model(inputs)
               loss = criterion(outputs, targets)
               
               val_loss += loss.item()
               _, predicted = torch.max(outputs.data, 1)
               val_total += targets.size(0)
               val_correct += predicted.eq(targets.data).cpu().sum()
       
       val_acc = 100. * float(val_correct) / float(val_total)
       avg_val_loss = val_loss / len(val_loader)
       print(f'Validation - Loss: {avg_val_loss:.3f} | Acc: {val_acc:.3f}%')
       
       scheduler.step(val_acc)
       earlystopping(val_acc, model)
       
       if earlystopping.early_stop:
           print("Early Stopping!")
           break
   
   print("Training completed")

# Main training script
def train_scdeepinsight(train_img_path, train_label_path, val_img_path, val_label_path, 
                       output_dir, batch_size=128, epochs=100, lr=1e-4):
   
   # Create output directory
   os.makedirs(output_dir, exist_ok=True)
   
   # Load labels and create encoder
   train_labels = pd.read_csv(train_label_path)
   label_col = 'celltype'  # adjust column name as needed
   
   le = preprocessing.LabelEncoder()
   le.fit(train_labels[label_col])
   n_classes = len(le.classes_)
   
   # Save label encoder
   with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
       pickle.dump(le, f)
   
   # Encode labels
   y_train = le.transform(train_labels[label_col])
   val_labels = pd.read_csv(val_label_path)
   y_val = le.transform(val_labels[label_col])
   
   # Save encoded labels
   np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
   np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
   
   # Create datasets
   train_dataset = ImageDataset(train_img_path, os.path.join(output_dir, 'y_train.npy'))
   val_dataset = ImageDataset(val_img_path, os.path.join(output_dir, 'y_val.npy'))
   
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
   # Setup model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=n_classes)
   
   if torch.cuda.device_count() > 1:
       model = nn.DataParallel(model)
   model = model.to(device)
   
   # Setup training
   optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=1e-5)
   scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
   criterion = LabelSmoothingCrossEntropy(eps=1e-4)
   
   # Train model
   checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
   train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
               device, epochs=epochs, checkpoint_path=checkpoint_path)
   
   return model, le

# Prediction script
def predict_scdeepinsight(query_img_path, model_path, encoder_path, output_path, batch_size=128):
   
   # Load label encoder
   with open(encoder_path, 'rb') as f:
       le = pickle.load(f)
   n_classes = len(le.classes_)
   
   # Setup model
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=n_classes)
   
   if torch.cuda.device_count() > 1:
       model = nn.DataParallel(model)
   
   model = model.to(device)
   model.load_state_dict(torch.load(model_path))
   model.eval()
   
   # Create dataset and loader
   test_dataset = ImageDataset(query_img_path)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
   # Make predictions
   predictions = []
   with torch.no_grad():
       for inputs in test_loader:
           inputs = inputs.to(device)
           outputs = model(inputs)
           _, predicted = torch.max(outputs.data, 1)
           predictions.extend(predicted.cpu().numpy())
   
   # Decode predictions
   pred_labels = le.inverse_transform(predictions)
   
   # Save results
   pred_df = pd.DataFrame({'predicted_celltype': pred_labels})
   pred_df.to_csv(output_path, index=False)
   
   return pred_df

# Example usage
if __name__ == "__main__":
   # Training
   train_scdeepinsight(
       train_img_path="./path/to/train_images.npy",
       train_label_path="./path/to/train_labels.csv",
       val_img_path="./path/to/val_images.npy", 
       val_label_path="./path/to/val_labels.csv",
       output_dir="./path/to/output/",
       batch_size=128,
       epochs=100
   )
   
   # Prediction
   predict_scdeepinsight(
       query_img_path="./path/to/query_images.npy",
       model_path="./path/to/output/checkpoint.pth",
       encoder_path="./path/to/output/label_encoder.pkl",
       output_path="./path/to/predictions.csv"
   )