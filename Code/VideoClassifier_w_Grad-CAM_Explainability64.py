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

# XAI Imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

# 2. CNN + LSTM Model (Modified for Grad-CAM)

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

# 3. Training Loop and Evaluation

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
    
    print(f"[DEBUG] Total usable dataset size: {len(dataset)}")
    for class_name in ['stimming', 'non_stimming']:
        class_path = os.path.join("UST_Data", class_name)
        print(f"Original check for {class_path} ... found {len(glob(os.path.join(class_path, '*.mp4')))} .mp4 files")

    total_size = len(dataset)
    train_val_size = int(0.9 * total_size)
    test_size = total_size - train_val_size

    train_val_ds, test_ds = torch.utils.data.random_split(dataset, [train_val_size, test_size])

    train_size = int(0.8 * len(train_val_ds))
    val_size = len(train_val_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_val_ds, [train_size, val_size])


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2)

    model = CNNLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_accuracy = 0.0
    model_save_path = "best_stimming_detector_model_mobilenetv2.pth"

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with Validation Accuracy: {best_val_accuracy:.2f}% to {model_save_path}")
    
    print("\n--- Training complete. Starting final evaluation ---")

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            videos, labels = videos.to(device), labels.to(device)
            preds_logits = model(videos)
            all_test_preds.extend(preds_logits.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    binary_test_preds = (torch.sigmoid(torch.tensor(all_test_preds)) > 0.5).int().numpy()
    
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


# --- Helper function for video frame loading ---
def load_frames_for_inference(path, num_frames, transform):
    cap = cv2.VideoCapture(path)
    frames = []
    original_frames = [] # To store frames for visualization
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0 or not cap.isOpened():
        print(f"Error: Could not open or read frames from {path}.")
        cap.release()
        return None, None

    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: 
            break
        if i in frame_idxs:
            frame_resized = cv2.resize(frame, (224, 224))
            original_frames.append(frame_resized / 255.0) # Store normalized original for overlay
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            if transform:
                frame_rgb = transform(frame_rgb)
            frames.append(frame_rgb)

    cap.release()
    
    if len(frames) == 0:
        print(f"Error: No frames were loaded from {path}. Check video integrity.")
        return None, None

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
        original_frames += [original_frames[-1]] * (num_frames - len(original_frames))
        
    return torch.stack(frames), np.array(original_frames)


# --- Function for single video inference ---
def predict_new_video(video_path, model, device, num_frames=16):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    video_frames, original_frames = load_frames_for_inference(video_path, num_frames, transform)

    if video_frames is None:
        return "Failed to process video.", None, None, None

    video_frames = video_frames.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_frames)
        probabilities = torch.sigmoid(logits)
        prediction_score = probabilities.item()
        binary_prediction = 1 if prediction_score > 0.5 else 0

    class_names = ['non_stimming', 'stimming']
    predicted_class_name = class_names[binary_prediction]

    print(f"\n--- Prediction Results for {os.path.basename(video_path)} ---")
    print(f"Probability of 'stimming': {prediction_score:.4f}")
    print(f"Predicted Class: {predicted_class_name}")

    return predicted_class_name, prediction_score, video_frames, original_frames


# --- New function for Grad-CAM Visualization (FIXED) ---
def generate_gradcam_visualizations(model, target_layer, input_tensor, original_frames, output_path):
    # Grad-CAM needs a model that takes a single frame and outputs a scalar.
    # The CNNLSTM model expects a sequence, so we create a wrapper for the CNN part.
    class CNNForwardWrapper(nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.feature_extractor = feature_extractor
        
        def forward(self, x):
            # Get the feature map from the CNN part
            features = self.feature_extractor(x)
            # Return a list containing the scalar tensor to make it iterable
            # for the Grad-CAM library.
            return [torch.mean(features)]

    # We apply Grad-CAM only to the CNN feature extractor part of the model.
    # Ensure the wrapper is on the same device as the input tensor.
    cam_model = CNNForwardWrapper(model.feature_extractor).to(input_tensor.device)
    
    # The target layer for Grad-CAM is the one we specify, which is inside the feature_extractor.
    cam = GradCAM(model=cam_model, target_layers=[target_layer])
    
    # This custom target class tells Grad-CAM how to get the scalar value from the model's output list.
    class ScalarOutputTarget:
        def __call__(self, model_output):
            # FIX: The model_output is the scalar tensor itself, not a list.
            # The grad-cam library unpacks the list for us.
            return model_output

    targets = [ScalarOutputTarget()]

    cam_frames = []
    print("Generating Grad-CAM visualizations...")
    for i in tqdm(range(input_tensor.shape[1]), desc="Processing frames for Grad-CAM"):
        # Get a single frame tensor: [1, C, H, W]
        frame_tensor = input_tensor[:, i, :, :, :]
        
        # Generate the CAM for the frame, using our custom target.
        grayscale_cam = cam(input_tensor=frame_tensor, targets=targets)
        
        # Take the first (and only) CAM in the batch
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay the CAM on the original, un-normalized frame
        visualization = show_cam_on_image(original_frames[i], grayscale_cam, use_rgb=True)
        
        # Convert RGB to BGR for OpenCV
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cam_frames.append(visualization_bgr)

    # Save the frames with overlays as a new video file
    if not cam_frames:
        print("Warning: No Grad-CAM frames were generated.")
        return
        
    height, width, layers = cam_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))

    for frame in cam_frames:
        out.write(frame)
    out.release()
    print(f"Grad-CAM video saved to {output_path}")


if __name__ == "__main__":
    # --- Run Training and Evaluation (as before) ---
    # You can comment this out if you have an already trained model
    # train_and_evaluate()

    # --- Setup for Prediction and XAI ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weights_path = "best_stimming_detector_model_mobilenetv2.pth"
    
    # Make sure to change this to the video you want to analyze
    new_video_path = "/home/aastha/Documents/Work/test_vid_NT3.mp4" 
    grad_cam_output_path = "grad_cam_output.mp4"

    print(f"\n--- Loading Model for Prediction and XAI ---")
    if not os.path.exists(model_weights_path):
        print(f"Error: Trained model weights not found at {model_weights_path}.")
        print("Please run the training first or provide the correct path.")
    elif not os.path.exists(new_video_path):
        print(f"Error: New video file not found at {new_video_path}. Please check the path.")
    else:
        # --- Load Model ---
        model = CNNLSTM().to(device)
        
        # --- FIX for state_dict key mismatch ---
        # Load the state dict from the saved model, which has a different architecture
        # Use weights_only=True for security
        state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)

        # Create a new state_dict to hold the re-mapped keys
        new_state_dict = model.state_dict()

        # Manually map the weights from the old model structure to the new one
        for key in state_dict:
            if key.startswith('cnn.0.'):
                # This maps weights from the old `cnn` Sequential to the new `feature_extractor`
                # e.g., 'cnn.0.1.conv...' -> 'feature_extractor.1.conv...'
                new_key = key.replace('cnn.0.', 'feature_extractor.', 1)
                if new_key in new_state_dict:
                    new_state_dict[new_key] = state_dict[key]
            elif key in new_state_dict:
                # This handles the lstm and fc layers which have not changed
                new_state_dict[key] = state_dict[key]

        # Load the re-mapped state_dict into our new model architecture
        model.load_state_dict(new_state_dict)
        model.eval()

        # --- Make Prediction ---
        predicted_class, score, video_tensor, original_frames = predict_new_video(new_video_path, model, device)
        print(f"Prediction complete. The video is predicted as: {predicted_class} with score: {score:.4f}")

        # --- Generate XAI Visualization ---
        if video_tensor is not None:
            # The target layer is the last convolutional block in MobileNetV2's features
            target_layer = model.feature_extractor[-1]
            generate_gradcam_visualizations(model, target_layer, video_tensor, original_frames, grad_cam_output_path)
