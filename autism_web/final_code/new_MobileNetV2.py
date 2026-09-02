import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Dataset: checking video integrity and loading frames

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.samples = []
        self.labels = []
        self.num_frames = num_frames
        self.transform = transform

        all_potential_samples = []
        all_potential_labels = []

        for label, class_name in enumerate(['non_stimming', 'stimming']):
            class_path = os.path.join(root_dir, class_name)
            print(f"Looking in {class_path}")
            for vid in glob(f"{class_path}/*.mp4"):
                all_potential_samples.append(vid)
                all_potential_labels.append(label)

        print("Validating video files...")
        for i, video_path in enumerate(tqdm(all_potential_samples, desc="Validating videos")):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file: {video_path}. Skipping.")
                cap.release()
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"Warning: Video {video_path} has 0 frames. Skipping.")
                cap.release()
                continue
            
            ret, _ = cap.read()
            if not ret:
                print(f"Warning: Could not read first frame from {video_path}. Skipping.")
                cap.release()
                continue
            
            cap.release()
            
            self.samples.append(video_path)
            self.labels.append(all_potential_labels[i])

    def __len__(self):
        return len(self.samples)

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None 

        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: 
                break
            if i in frame_idxs:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        cap.release()
        
        if len(frames) == 0:
            print(f"Warning: No frames were loaded from {path} after processing in load_video_frames. Check video integrity.")
            return None

        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))  
            
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        frames = self.load_video_frames(video_path)
        
        if frames is None:
            raise RuntimeError(f"Failed to load frames for video: {video_path} unexpectedly.")
            
        return frames, torch.tensor(label, dtype=torch.float32)

# 2. CNN + LSTM Model (CNN for spatial feature extraction, LSTM for temporal modeling)

# USE MOBILENETV2

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Load MobileNetV2 pretrained model
        base_model = models.mobilenet_v2(pretrained=True)
        
        # MobileNetV2 features are in the `features` attribute.
        # The classifier part is `base_model.classifier`.
        # We want the output of the convolutional layers before the final classification head.
        # The output of base_model.features is typically [B, 1280, 7, 7] for 224x224 input.
        # Then an AvgPool is applied to get [B, 1280, 1, 1], and then a view to [B, 1280].
        # So we take the `features` part and the `avgpool` (which is part of the classifier's first layer essentially)
        # However, a simpler way is to take all layers except the very last classifier layer.

        # Let's take the features layer and then apply adaptive average pooling as MobileNetV2 does internally.
        self.cnn = nn.Sequential(
            *list(base_model.children())[:-1], # Takes features and avgpool (or just features)
            nn.AdaptiveAvgPool2d((1, 1))       # Ensure output is 1x1
        )
        
        # The output size of MobileNetV2's feature extractor (after AdaptiveAvgPool2d) is 1280
        cnn_out_features = 1280 
        
        self.lstm = nn.LSTM(cnn_out_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            # The output of self.cnn will be [B*T, cnn_out_features, 1, 1] due to AdaptiveAvgPool2d
            feats = self.cnn(x).squeeze()  # Squeeze to [B*T, cnn_out_features]
        feats = feats.view(B, T, -1)
        _, (hn, _) = self.lstm(feats)
        hn_last_layer = hn[-1].squeeze(0) # Shape: [B, hidden_dim]
        out = self.fc(hn_last_layer)      # Shape: [B, 1]
        
        return torch.sigmoid(out).view(-1) # Reshape to a 1D tensor of size B


# 3. Training Loop and Evaluation

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = VideoDataset("UST_Data", transform=transform)
    
    print(f"[DEBUG] Total usable dataset size: {len(dataset)}")
    for class_name in ['stimming', 'non_stimming']:
        class_path = os.path.join("UST_Data", class_name)
        print(f"Original check for {class_path} ... found {len(glob(os.path.join(class_path, '*.mp4')))} .mp4 files")

    # Split into train, validation, and test sets
    total_size = len(dataset)
    train_val_size = int(0.9 * total_size) # Using 90% for train+val
    test_size = total_size - train_val_size # Remaining 10% for test

    train_val_ds, test_ds = torch.utils.data.random_split(dataset, [train_val_size, test_size])

    train_size = int(0.8 * len(train_val_ds)) # 80% of train_val_ds for training
    val_size = len(train_val_ds) - train_size # 20% of train_val_ds for validation
    train_ds, val_ds = torch.utils.data.random_split(train_val_ds, [train_size, val_size])


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2) # DataLoader for the test set

    model = CNNLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_accuracy = 0.0
    model_save_path = "best_stimming_detector_model_mobilenetv2.pth" # Changed model name

    for epoch in range(20):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            videos, labels = batch
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                preds = model(videos)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                preds = (preds > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Validation Accuracy: {best_val_accuracy:.2f}% to {model_save_path}")
    
    print("\n--- Training complete. Starting final evaluation ---")

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            videos, labels = videos.to(device), labels.to(device)
            preds = model(videos)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    binary_test_preds = (np.array(all_test_preds) > 0.5).astype(int)
    
    print("\n--- Test Set Evaluation Results ---")
    print(classification_report(all_test_labels, binary_test_preds, target_names=['non_stimming', 'stimming']))

    cm = confusion_matrix(all_test_labels, binary_test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['non_stimming', 'stimming'],
                yticklabels=['non_stimming', 'stimming'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()