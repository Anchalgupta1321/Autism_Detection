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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset: checking video integrity and loading frames

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.samples = []
        self.labels = []
        self.num_frames = num_frames
        self.transform = transform

        all_potential_samples = []
        all_potential_labels = []

        for label, class_name in enumerate(['non_ASD_behaviour', 'ASD_behaviour']):
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

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=64, dropout_prob=0.5):
        super().__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = base_model.features
        self.cnn_tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        cnn_out_features = 1280 
        self.lstm = nn.LSTM(cnn_out_features, hidden_dim, batch_first=True)
        
        # Dropout layer added before the final classification layer
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        cnn_feats = self.feature_extractor(x)
        cnn_feats = self.cnn_tail(cnn_feats)
        feats = cnn_feats.view(B, T, -1)
        _, (hn, _) = self.lstm(feats)
        hn_last_layer = hn[-1].squeeze(0)
        
        # Apply dropout before the final layer
        out = self.dropout(hn_last_layer)
        
        out = self.fc(out)
        return out.view(-1)

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ToTensor()
    ])

    dataset = VideoDataset("UST_Data", transform=transform)
    
    print(f"Total usable dataset size: {len(dataset)}")
    for class_name in ['ASD_behaviour', 'non_ASD_behaviour']:
        class_path = os.path.join("UST_Data", class_name)
        print(f"Original check for {class_path} found {len(glob(os.path.join(class_path, '*.mp4')))} .mp4 files")

    total_size = len(dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    print("\n--- Data Splitting ---")
    print(f"Splitting dataset of size {total_size} into:")
    print(f"  Train: {train_size} samples (~{train_size/total_size*100:.1f}%)")
    print(f"  Validation: {val_size} samples (~{val_size/total_size*100:.1f}%)")
    print(f"  Test: {test_size} samples (~{test_size/total_size*100:.1f}%)")
    print("\n")
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, 
                                                              [train_size, val_size, test_size],
                                                              generator=generator)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2)

    model = CNNLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_accuracy = 0.0
    model_save_path = "best_stimming_detector_model_mobilenetv2.pth"

    # Tracking metrics for visualization
    train_losses = []
    val_accuracies = []

    for epoch in range(18):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/18"):
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
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                preds_logits = model(videos)
                preds_probs = torch.sigmoid(preds_logits)
                preds_binary = (preds_probs > 0.5).float()
                correct += (preds_binary == labels).sum().item()
                total += labels.size(0)
        
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Append metrics after they are computed
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Validation Accuracy: {best_val_accuracy:.2f}% to {model_save_path}")

    # Plotting after training
    print("\n--- Training complete. Plotting metrics and starting final evaluation ---")

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, marker='o', color='orange', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Final Evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            videos, labels = videos.to(device), labels.to(device)
            preds_logits = model(videos)
            all_test_preds.extend(preds_logits.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    logits_tensor = torch.tensor(all_test_preds)
    probs = torch.sigmoid(logits_tensor).numpy()
    binary_preds = (probs > 0.5).astype(int)

    print("\n--- Test Set Evaluation Results ---")
    print(classification_report(all_test_labels, binary_preds, target_names=['non-ASD behaviour', 'ASD behaviour']))

    cm = confusion_matrix(all_test_labels, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['non-ASD behaviour', 'ASD behaviour'],
                yticklabels=['non-ASD behaviour', 'ASD behaviour'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    accuracy = accuracy_score(all_test_labels, binary_preds)
    precision = precision_score(all_test_labels, binary_preds)
    recall = recall_score(all_test_labels, binary_preds)
    f1 = f1_score(all_test_labels, binary_preds)
    roc_auc = roc_auc_score(all_test_labels, probs)

    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Precision  : {precision:.2f}")
    print(f"Recall     : {recall:.2f}")
    print(f"F1-score   : {f1:.2f}")
    print(f"ROC-AUC    : {roc_auc:.2f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")




# New function for single video inference
def predict_new_video(video_path, model_path, num_frames=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = CNNLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def load_frames_for_inference(path, num_frames, transform):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0 or not cap.isOpened():
            print(f"Error: Could not open or read frames from {path}.")
            cap.release()
            return None 

        frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: 
                break
            if i in frame_idxs:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if transform:
                    frame = transform(frame)
                frames.append(frame)

        cap.release()
        
        if len(frames) == 0:
            print(f"Error: No frames were loaded from {path}. Check video integrity.")
            return None

        if len(frames) < num_frames:
            frames += [frames[-1]] * (num_frames - len(frames))  
            
        return torch.stack(frames)


    print(f"Processing video: {video_path}")
    video_frames = load_frames_for_inference(video_path, num_frames, transform)

    if video_frames is None:
        return "Failed to process video."

    video_frames = video_frames.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_frames)
        probabilities = torch.sigmoid(logits)
        prediction_score = probabilities.item()
        binary_prediction = 1 if prediction_score > 0.5 else 0

    class_names = ['non-ASD behaviour', 'ASD behaviour']
    predicted_class_name = class_names[binary_prediction]

    print(f"\n--- Prediction Results for {os.path.basename(video_path)} ---")
    print(f"Probability of 'ASD behaviour': {prediction_score:.4f}")
    print(f"Predicted Class: {predicted_class_name}")

    return predicted_class_name, prediction_score


if __name__ == "__main__":
    train_and_evaluate()

    new_video_path = "/home/aastha/Documents/Work/test_vid2_HF.mp4"
    model_weights_path = "best_stimming_detector_model_mobilenetv2.pth" 

    print(f"\nAttempting to predict for a new video: {new_video_path}")
    if os.path.exists(model_weights_path):
        if os.path.exists(new_video_path):
            predicted_class, score = predict_new_video(new_video_path, model_weights_path)
            print(f"Prediction complete. The video is predicted as: {predicted_class} with score: {score:.4f}")
        else:
            print(f"Error: New video file not found at {new_video_path}. Please check the path.")
    else:
        print(f"Error: Trained model weights not found at {model_weights_path}. Please ensure training completed successfully.")