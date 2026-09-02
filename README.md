# Early Detection of Autism in Toddlers

An end-to-end multi-modal screening and detection system for Autism Spectrum Disorder (ASD) in toddlers, integrating **Deep Learning (CNN-LSTM)**, **Explainable AI (Grad-CAM)**, and a **clinical questionnaire web portal**.

---

## 📌 Project Overview

Early detection of Autism Spectrum Disorder (ASD) in toddlers is vital for timely intervention and significantly improved developmental outcomes. This project provides:

1. **Automated Stimming Detection:** A hybrid deep neural network combining a pre-trained **MobileNetV2** (spatial feature extractor) and an **LSTM** (temporal sequence classifier) to detect atypical repetitive motor behaviors ("stimming"), including:
   - Hand flapping
   - Body spinning
   - Head banging
2. **Explainable AI (Grad-CAM):** Gradient-weighted Class Activation Mapping to provide visual explainability and transparency into the features influencing the model's classifications.
3. **Clinical Screening Questionnaire:** A multi-step structured questionnaire based on M-CHAT red flag criteria across 5 behavioral categories:
   - Social Interaction & Communication
   - Joint Attention & Social Referencing
   - Repetitive Behaviors & Sensory Responses
   - Play & Peer Interaction
   - Developmental Milestones
4. **Interactive Web Portal:** A Flask-based web application allowing caregivers and clinicians to complete the questionnaire, upload video recordings for real-time analysis, view detailed diagnostic insights, and receive summary reports via email.

---

## 📂 Repository Structure

```text
├── ASD_Paper.docx                   # Research paper manuscript
├── UST_ASD_PPT.pdf                  # Presentation slides on project & findings
├── Code/
│   ├── VideoClassifier_w_Grad-CAM_Explainability64.py  # Model training, evaluation & Grad-CAM pipeline
│   ├── test_vid.py                  # Video testing and inference script
│   ├── viz.py                       # Data visualization utilities
│   ├── best_stimming_detector_model_mobilenetv2.pth   # Pre-trained CNN-LSTM model weights
│   ├── video_data/                  # Video dataset
│   │   ├── ASD_behaviour/           # Stimming / ASD behavior sample videos
│   │   └── non_ASD_behaviour/       # Typical toddler movement video samples
│   └── videos_for_testing/          # Benchmark test videos
└── autism_web/
    ├── app.py                       # Flask web application server
    ├── requirements.txt             # Web application Python dependencies
    ├── .env.example                 # Template for email & session environment variables
    ├── templates/                   # Jinja2 HTML templates
    │   ├── index.html               # Welcome & landing page
    │   ├── Step1.html - Step5.html  # Multi-step behavioral questionnaire
    │   ├── video_upload.html        # Video upload interface
    │   ├── results.html             # Video inference results & visualization
    │   ├── final_results.html       # Combined diagnostic summary report
    │   └── email_template.html      # Email notification template
    ├── final_code/
    │   ├── best_stimming_detector_model_mobilenetv2.pth # Production model weights
    │   └── new_MobileNetV2.py       # MobileNetV2 feature extractor module
    ├── flask_session_data/          # Runtime server session store
    └── uploads/                     # Uploaded video directory
```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Computer Vision:** OpenCV, Pillow
- **Explainability (XAI):** PyTorch Grad-CAM
- **Web Framework:** Flask, Flask-Session, Flask-Mail
- **Data Analysis & Visualization:** NumPy, Matplotlib, Seaborn, scikit-learn, tqdm

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Anchalgupta1321/Autism_Detection.git
cd Autism_Detection
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd autism_web
pip install -r requirements.txt
```

*(Optional: For training and Grad-CAM visualization in `Code/`, also install `scikit-learn`, `seaborn`, `matplotlib`, `tqdm`, and `grad-cam`)*:

```bash
pip install scikit-learn seaborn matplotlib tqdm grad-cam
```

---

## 💻 Running the Web Application

1. Configure environment variables (optional, for email reports):
   ```bash
   cp .env.example .env
   # Edit .env with your SMTP credentials if automated email reporting is needed
   ```

2. Start the Flask server:
   ```bash
   cd autism_web
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

---

## 🔬 Model Architecture & Training

The stimming classification model uses a two-stage spatio-temporal architecture:

```
[Input Video Frames: B × T × C × H × W]
               │
               ▼
┌───────────────────────────────┐
│ MobileNetV2 (Pre-trained CNN) │ ──> Spatial Feature Vector (1280-dim per frame)
└───────────────────────────────┘
               │
               ▼
┌───────────────────────────────┐
│      LSTM Recurrent Layer     │ ──> Captures temporal dynamics across frames
└───────────────────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ Fully Connected Layer + Sigmoid│ ──> Probability score: P(Stimming / ASD Behavior)
└───────────────────────────────┘
```

- **Explainability:** Grad-CAM targets the final convolutional layer of MobileNetV2, computing gradients with respect to predicted class scores to project heatmaps onto the toddler's body and motion areas.

---

## 📄 License & Citation

This project is developed for research and educational purposes. Please refer to `ASD_Paper.docx` for the research methodology, dataset acquisition, and full experimental results.
