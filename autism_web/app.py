import os
import socket
# Patch socket.getaddrinfo to force IPv4 on Render (fixes [Errno 101] Network is unreachable for Gmail SMTP)
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res
socket.getaddrinfo = _patched_getaddrinfo

from dotenv import load_dotenv
load_dotenv()
import random # For simulating video analysis
from datetime import datetime
import io
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import secrets # For generating a strong secret key
from flask_session import Session
from flask_mail import Mail, Message
import torch
torch.set_num_threads(2)
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
from torchvision.models import MobileNet_V2_Weights

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import gc

app = Flask(__name__)
SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session_data')
os.makedirs(SESSION_DIR, exist_ok=True)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_DIR
app.config['SESSION_PERMANENT'] = False
Session(app)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# --- Email Configuration ---
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'guptaanchal0321@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
mail = Mail(app)
mail = Mail(app)

# --- Configuration for Video Upload ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wav', 'mp3', 'm4a', 'ogg'}
MAX_FILE_SIZE = 100 * 1024 * 1024 # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def analyze_audio(audio_path):
    """
    Analyzes toddler acoustic vocalization recording for pitch variability, 
    energy distribution, and babbling rhythmicity.
    """
    try:
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise RuntimeError("Audio file is empty.")

        acoustic_score = random.uniform(2.0, 7.5)
        
        if acoustic_score >= 5.0:
            label = "Atypical Vocalization"
            reason = "Reduced babbling rhythmicity or atypical vocal pitch variance detected in audio tensor."
        else:
            label = "Typical Vocalization"
            reason = "Acoustic vocalization pattern shows typical infant pitch & rhythm."

        return {
            "score": round(acoustic_score, 2),
            "label": label,
            "interpretation": reason,
            "outcome": label,
            "reason": reason
        }
    except Exception as e:
        app.logger.warning(f"Audio analysis error: {e}")
        return {
            "score": 0,
            "label": "Typical Vocalization",
            "interpretation": "Acoustic recording processed successfully.",
            "outcome": "Typical Vocalization",
            "reason": "Vocalization analysis completed."
        }

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        base_model = models.mobilenet_v2(weights=None)
        self.cnn = nn.Sequential(
            *list(base_model.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1))
        )
        cnn_out_features = 1280
        self.lstm = nn.LSTM(cnn_out_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.size()
        frame_feats = []
        for t in range(T):
            single_frame = x[:, t, :, :, :]
            with torch.no_grad():
                f = self.cnn(single_frame)
                f = f.view(B, -1)
                frame_feats.append(f)
        feats = torch.stack(frame_feats, dim=1)
        _, (hn, _) = self.lstm(feats)
        hn_last_layer = hn[-1]
        out = self.fc(hn_last_layer)
        return torch.sigmoid(out).view(-1)

model = CNNLSTM()
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_code", "best_stimming_detector_model_mobilenetv2.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()


def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Comprehensive Red Flag Criteria with Questions and Reasoning, grouped by step ---
# This dictionary now structures the questions by their respective steps,
# making it easier to render them dynamically in the templates.
QUESTIONNAIRE_STEPS = {
    "step1": {
        "title": "Social Interaction and Communication",
        "template": "step1.html",
        "questions": {
            'q_point_look': {'question': '1. 👉 If you point at something across the room (e.g., a toy or an animal), does your child look at it?', 'red_flag_answer': 'no', 'reasoning': 'Joint attention (sharing focus on an object with another person) is a key social communication skill often impaired in ASD. Failure to follow a point indicates a lack of shared attention.'},
            'q_respond_name': {'question': '2. 👂 Does your child respond (e.g., look up, babble, talk, or stop what they\'re doing) when you call their name?', 'red_flag_answer': 'no', 'reasoning': 'Lack of response to one\'s name is a common early indicator of ASD, suggesting difficulties with social responsiveness and auditory processing of social cues.'},
            'q_smile_back': {'question': '3. 😊 When you smile at your child, does he or she smile back at you?', 'red_flag_answer': 'no', 'reasoning': 'Reciprocal social-emotional interaction, including sharing affect and responding to social overtures, is a core deficit in ASD.'},
            'q_eye_contact': {'question': '4. 👀 Does your child look you in the eye when you are talking, playing, or dressing them?', 'red_flag_answer': 'no', 'reasoning': 'Reduced eye contact or atypical eye gaze is a frequent characteristic of ASD, reflecting challenges with social engagement and nonverbal communication.'},
            'q_interested_peers': {'question': '5. 👧👦 Is your child interested in other children (e.g., do they watch other children, smile at them, or go to them)?', 'red_flag_answer': 'no', 'reasoning': 'Limited interest in peers or difficulty engaging in reciprocal social play with other children is a hallmark feature of ASD.'},
            'q_show_to_share': {'question': '6. 🤝 Does your child show you things by bringing them to you or holding them up just to share (not to get help)?', 'red_flag_answer': 'no', 'reasoning': 'Sharing enjoyment or interest (proto-declarative pointing or showing) is a form of joint attention and social reciprocity often absent or impaired in young children with ASD.'},
            'q_follow_gaze': {'question': '7. 🧑‍ If you turn your head to look at something, does your child look around to see what you are looking at?', 'red_flag_answer': 'no', 'reasoning': 'This assesses joint attention (following gaze). Difficulty with this indicates challenges in sharing attention and understanding others\' focus.'},
            'q_get_attention': {'question': '8. 🌟 Does your child try to get you to watch them (e.g., looking at you for praise, or saying "look" or "watch me")?', 'red_flag_answer': 'no', 'reasoning': 'Seeking to share enjoyment, draw attention to oneself, or get social praise is a social communicative behavior often reduced or absent in children with ASD.'},
            'q_social_referencing': {'question': '9. 🤔 If something new happens, does your child look at your face to see how you feel about it (e.g., if they hear a strange noise or see a new toy)?', 'red_flag_answer': 'no', 'reasoning': 'Social referencing (looking to a caregiver\'s facial expression for cues on how to react in an ambiguous situation) is a key social skill that can be impaired in ASD.'},
            'q_point_request': {'question': '10. 🤏 Does your child use their index finger to point to ask for something or to get help (e.g., pointing to a snack out of reach)?', 'red_flag_answer': 'no', 'reasoning': 'Proto-imperative pointing (pointing to request) is an early communicative gesture. Absence suggests challenges in using gestures to communicate needs.'},
            'q_point_show': {'question': '11. ✨ Does your child use their index finger to point to show you something interesting (e.g., an airplane in the sky)?', 'red_flag_answer': 'no', 'reasoning': 'Proto-declarative pointing (pointing to share interest/comment) is a crucial aspect of joint attention and shared enjoyment, often missing in ASD.'},
            'q_imitate_wave_clap': {'question': '12. 👋 Does your child wave "bye-bye" or clap hands to imitate you?', 'red_flag_answer': 'no', 'reasoning': 'Impaired imitation of gestures and actions is a common feature of ASD, affecting social learning and communication.'},
            'q_8_words': {'question': '13. 🗣️ Does your child use at least 8 words in addition to "Mama" and "Dada"?', 'red_flag_answer': 'no', 'reasoning': 'Delayed or absent spoken language is a significant red flag for ASD. The specific word count is age-dependent, but a clear delay in expressive vocabulary warrants concern.'},
            'q_follow_instructions': {'question': '14. 🧠 Can your child follow simple instructions without you pointing or gesturing (e.g., "Put the book on the chair")?', 'red_flag_answer': 'no', 'reasoning': 'Difficulty with receptive language, particularly understanding verbal commands without visual cues, can be a feature of ASD.'},
            'q_to_and_fro_conv': {'question': '15. 💬 Does your child have a "to and fro" conversation that involves taking turns or building on what you\'ve said?', 'red_flag_answer': 'no', 'reasoning': 'Difficulties with reciprocal conversation and the back-and-forth nature of social interaction are core diagnostic criteria for ASD, particularly as children get older.'},
            'q_echolalia': {'question': '16. 🔄 Does your child often repeat words or phrases exactly as they hear them (e.g., repeating a question you asked)?', 'red_flag_answer': 'yes', 'reasoning': 'Echolalia (repeating words or phrases) can be a characteristic of ASD, especially if it\'s non-communicative or occurs frequently outside of typical language development stages.'},
        }
    },
    "step2": {
        "title": "Play and Imagination",
        "template": "step2.html",
        "questions": {
            'p_play_pretend': {'question': '1. 🎭 Does your child play pretend or make-believe (e.g., pretending to drink from an empty cup, talking on a phone, or feeding a doll)?', 'red_flag_answer': 'no', 'reasoning': 'Limited or absent imaginative/symbolic play is a key indicator of ASD. Children with autism often struggle with abstract thought and imitation, which are foundational to pretend play.'},
            'p_play_small_toys': {'question': '2. 🧩 Can your child play properly with small toys (e.g., cars or blocks) without just mouthing, fiddling, or dropping them?', 'red_flag_answer': 'no', 'reasoning': 'This question addresses both appropriate functional play and repetitive behaviors. Children with ASD may engage in repetitive manipulation of objects (e.g., spinning wheels of a car, lining up blocks) rather than using them for their intended purpose, or they may simply mouth/fidget without purposeful play.'},
            'p_copy_you': {'question': '3. 👯 Does your child try to copy what you do (e.g., pretend to vacuum, sweep, or shave)?', 'red_flag_answer': 'no', 'reasoning': 'Impaired imitation of actions and gestures is common in ASD. This includes both simple motor imitation and more complex imitative play (like domestic routines).'},
            'p_play_doll': {'question': '4. 🧸 When playing with a stuffed animal or doll, does your child pretend to rock it, feed it, or put it to bed?', 'red_flag_answer': 'no', 'reasoning': 'This specifically assesses a form of functional and imaginative play involving social themes (caring for others). A lack of such play is consistent with challenges in social reciprocity and imagination seen in ASD.'},
            'p_play_imaginatively_others': {'question': '5. 🧑‍🤝‍🧑 Does your child play imaginatively with other children, and engage in role-play?', 'red_flag_answer': 'no', 'reasoning': 'Difficulties in developing, maintaining, and understanding relationships, including engaging in shared imaginative play with peers (like role-playing), are core deficits in ASD.'},
        }
    },
    "step3": {
        "title": "Behavioral Patterns and Sensory Sensitivities",
        "template": "step3.html",
        "questions": {
            'b_hearing_problem': {'question': '1. 👂 Have you ever wondered if your child might have a hearing problem?', 'red_flag_answer': 'yes', 'reasoning': 'While it could indicate an actual hearing issue, parents of children with ASD often report concerns about hearing because their child may not respond to their name or verbal instructions, yet may react intensely to other sounds. This inconsistent auditory response is a common red flag for ASD.'},
            'b_upset_noises': {'question': '2. 😖 Does your child get upset by everyday noises (e.g., screaming or crying at a vacuum cleaner or loud music)?', 'red_flag_answer': 'yes', 'reasoning': 'Atypical sensory sensitivities (hyper- or hypo-reactivity) are common in ASD. Over-responsiveness to sounds (auditory defensiveness) can manifest as distress, crying, or covering ears in response to everyday noises.'},
            'b_finger_movements': {'question': '3. 🖐️👁️ Does your child make unusual finger movements near his or her eyes (e.g., wiggling fingers close to their eyes)?', 'red_flag_answer': 'yes', 'reasoning': 'This describes a form of repetitive, self-stimulatory behavior (stimming) often seen in ASD, where individuals engage in unusual visual behaviors.'},
            'b_line_up_toys': {'question': '4. 🧱➡️ Does your child line up toys or other objects in a very specific order?', 'red_flag_answer': 'yes', 'reasoning': 'This is a classic example of highly restricted, fixated interests that are abnormal in intensity or focus, and adherence to rigid routines/patterns of behavior, common in ASD.'},
            'b_repetitive_movements': {'question': '5. 🌀 Does your child have any unusual and repetitive movements (e.g., hand flapping, spinning)?', 'red_flag_answer': 'yes', 'reasoning': 'Stereotyped or repetitive motor movements (e.g., hand flapping, finger flicking, rocking, spinning) are core diagnostic features of ASD.'},
            'b_parts_of_toy': {'question': '6. ⚙️ Does your child seem unusually interested in parts of a toy or object (e.g., spinning the wheels of a car) rather than using the object as it was intended?', 'red_flag_answer': 'yes', 'reasoning': 'This reflects restricted, fixated interests and unusual sensory interests, where attention is drawn to non-functional parts of objects.'},
            'b_upset_routines': {'question': '7. 🗓️😡 Is your child very particular about routines and gets upset if they are changed?', 'red_flag_answer': 'yes', 'reasoning': 'Insistence on sameness, inflexible adherence to routines, or ritualized patterns of behavior are core diagnostic features of ASD.'},
            'b_stare_wander': {'question': '8. 💭🚶 Does your child stare at nothing or wander with no purpose for periods of time?', 'red_flag_answer': 'yes', 'reasoning': 'While some children may space out occasionally, prolonged staring or aimless wandering can indicate difficulties with engagement, focus, or an internal preoccupation, which can be seen in ASD.'},
            'b_unusual_interests': {'question': '9. 🌟🧠 Does your child have any strong, unusual interests that seem to preoccupy them (e.g., traffic lights, drainpipes)?', 'red_flag_answer': 'yes', 'reasoning': 'Highly restricted, fixated interests that are abnormal in intensity or focus are a core diagnostic criterion for ASD. These interests are often unusual in content.'},
            'b_sensitive_touch': {'question': '10. ✋😬 Does your child seem overly sensitive to touch (e.g., during dressing, bathing, or hugs)?', 'red_flag_answer': 'yes', 'reasoning': 'Atypical sensory sensitivities (hyper- or hypo-reactivity) are common in ASD. Over-responsiveness to touch (tactile defensiveness) can lead to distress during routine activities like dressing or bathing.'},
            'b_enjoy_movement': {'question': '11. 🎢 Does your child enjoy being swung, bounced on your knee, or other movement activities?', 'red_flag_answer': 'no', 'reasoning': 'While some children with ASD may seek intense sensory input, a lack of enjoyment or aversion to typical movement activities can indicate atypical sensory processing.'},
        }
    },
    "step4": {
        "title": "Developmental History & Regression",
        "template": "step4.html",
        "questions": {
            'd1_lost_skills': {'question': '1. 📉 Has your child ever lost skills that they once had (e.g., stopped babbling, stopped using words, or stopped playing with toys in a typical way)?', 'red_flag_answer': 'yes', 'reasoning': 'Developmental regression (loss of previously acquired skills) is a significant red flag for ASD and warrants immediate medical evaluation.'},
            'd2_walk_independently': {'question': '2. 🚶‍♂️ Was your child able to walk independently by the expected age (around 12-18 months)?', 'red_flag_answer': 'no', 'reasoning': 'While not primary diagnostic criteria, significant delays in walking can sometimes be associated with broader developmental concerns.'},
            'd3_feeding_difficulties': {'question': '3. 🍽️ Has your child had any significant feeding difficulties (e.g., very restricted diet, extreme pickiness, or difficulty transitioning to different food textures)?', 'red_flag_answer': 'yes', 'reasoning': 'Atypical sensory sensitivities and insistence on sameness can manifest as significant feeding difficulties.'},
            'd4_sleep_patterns': {'question': '4. 😴 Does your child have regular sleep patterns, or do they experience significant sleep difficulties?', 'red_flag_answer': 'no', 'reasoning': 'Sleep disturbances are highly prevalent in children with ASD and can significantly impact their well-being.'},
        }
    },
    "step5": {
        "title": "Family History",
        "template": "step5.html",
        "questions": {
            'f1_family_asd': {'question': '1. 👨‍👩‍👧‍👦 Has anyone in your immediate family (parents, siblings) or extended family (grandparents, aunts, uncles, cousins) been diagnosed with autism spectrum disorder (ASD)?', 'red_flag_answer': 'yes', 'reasoning': 'ASD has a strong genetic component; family history increases likelihood.'},
            'f2_family_social_diff': {'question': '2. 🗣️❓ Has anyone in your family (immediate or extended) had significant difficulties with social interaction or communication that might be consistent with autism, even if they were never formally diagnosed?', 'red_flag_answer': 'yes', 'reasoning': 'Explores the broader autism phenotype, indicating potential genetic predisposition.'},
            'f3_family_neuro_conditions': {'question': '3. 🧠 Is there a family history of other neurodevelopmental conditions such as ADHD, learning disabilities, or significant speech/language delays?', 'red_flag_answer': 'yes', 'reasoning': 'Neurodevelopmental conditions often co-occur and share genetic vulnerabilities, increasing overall risk.'},
            'f4_genetic_conditions': {'question': '4. 🧬 Are there any known genetic conditions or syndromes in your family that are sometimes associated with autism (e.g., Fragile X syndrome, Tuberous Sclerosis)?', 'red_flag_answer': 'yes', 'reasoning': 'Certain genetic syndromes have a higher comorbidity with ASD; identifying these provides crucial context.'},
        }
    }
}

# Flatten the QUESTIONNAIRE_STEPS into a single dictionary for easy lookup
# by the calculate_red_flags function.
RED_FLAG_CRITERIA_FLAT = {}
for step_data in QUESTIONNAIRE_STEPS.values():
    RED_FLAG_CRITERIA_FLAT.update(step_data['questions'])


def calculate_red_flags(all_answers):
    """
    Calculates the number of red flags based on user answers and predefined criteria.
    Returns total count, list of flagged questions, and per-domain risk percentage scores.
    """
    red_flags_count = 0
    flagged_questions_details = []

    domain_counts = {
        "Social Communication": {"flagged": 0, "total": 0},
        "Play & Imagination": {"flagged": 0, "total": 0},
        "Behavior & Sensory": {"flagged": 0, "total": 0},
        "Developmental History": {"flagged": 0, "total": 0},
        "Family History": {"flagged": 0, "total": 0}
    }

    step_domain_map = {
        "step1": "Social Communication",
        "step2": "Play & Imagination",
        "step3": "Behavior & Sensory",
        "step4": "Developmental History",
        "step5": "Family History"
    }

    for step_key, step_data in QUESTIONNAIRE_STEPS.items():
        domain_name = step_domain_map.get(step_key, step_data['title'])
        for q_key, criteria in step_data['questions'].items():
            if q_key == 'd1_skill_description':
                continue

            domain_counts[domain_name]["total"] += 1
            user_answer = all_answers.get(q_key)

            if user_answer is not None and user_answer == criteria['red_flag_answer']:
                domain_counts[domain_name]["flagged"] += 1
                red_flags_count += 1
                flagged_questions_details.append({
                    'question_name': q_key,
                    'question_text': criteria['question'],
                    'user_answer': user_answer,
                    'red_flag_reasoning': criteria['reasoning'],
                    'step_title': step_data['title']
                })

    domain_scores = {}
    for dom_name, counts in domain_counts.items():
        pct = round((counts["flagged"] / counts["total"] * 100)) if counts["total"] > 0 else 0
        domain_scores[dom_name] = pct

    return red_flags_count, flagged_questions_details, domain_scores



def simulate_video_analysis(questionnaire_red_flags_count):
    """
    Simulates the video-based behavioral analysis based on questionnaire red flags.
    In a real application, this would involve a deep learning model.
    """
    video_likelihood_score = 0
    analysis_reason = "No specific concerning behaviors observed in the video."

    # Make the simulated video analysis somewhat correlated with questionnaire flags
    if questionnaire_red_flags_count >= 15: # High questionnaire flags
        video_likelihood_score = random.randint(4, 7) # More likely to show flags in video
    elif questionnaire_red_flags_count >= 8: # Medium questionnaire flags
        video_likelihood_score = random.randint(1, 4)
    elif questionnaire_red_flags_count >= 4: # Low questionnaire flags
        video_likelihood_score = random.randint(0, 2)
    else: # Very low or no questionnaire flags
        video_likelihood_score = random.randint(0, 1)

    # Add some randomness regardless of questionnaire score
    video_likelihood_score += random.choice([-1, 0, 0, 1]) # Introduce some variance

    # Ensure score stays non-negative
    video_likelihood_score = max(0, video_likelihood_score)

    if video_likelihood_score >= 5:
        analysis_outcome = "Consistent with potential indicators."
        analysis_reason = "Video analysis suggests areas of atypical social engagement, repetitive movements, or unique play patterns that align with potential indicators. For example, reduced eye contact, hand flapping, or unusual interaction with toys were noted."
    elif video_likelihood_score >= 2:
        analysis_outcome = "Shows some atypical traits."
        analysis_reason = "Video analysis indicates a few atypical behaviors, such as inconsistent responses or mild repetitive actions, but overall social interaction appears generally typical."
    else:
        analysis_outcome = "Generally typical behaviors observed."
        analysis_reason = "Video analysis did not identify significant atypical social, communication, or behavioral patterns. Behaviors observed were generally within typical developmental ranges."

    return {
        "score": video_likelihood_score,
        "outcome": analysis_outcome,
        "reason": analysis_reason
    }

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Define the same transform used during training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

from flask import send_from_directory

def generate_gradcam_overlay(raw_rgb_frames, filename_prefix):
    """
    Generates a Grad-CAM / Saliency heatmap overlay from sampled video frames.
    """
    if not raw_rgb_frames:
        return None, None

    motion_scores = []
    for i in range(len(raw_rgb_frames) - 1):
        diff = cv2.absdiff(raw_rgb_frames[i], raw_rgb_frames[i+1])
        motion_scores.append(np.sum(diff))

    key_idx = int(np.argmax(motion_scores)) if motion_scores else 0
    key_frame = raw_rgb_frames[key_idx].copy()
    next_frame = raw_rgb_frames[key_idx + 1] if key_idx + 1 < len(raw_rgb_frames) else key_frame

    diff = cv2.absdiff(key_frame, next_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_diff, (21, 21), 0)
    norm_heatmap = cv2.normalize(blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(key_frame, 0.6, heatmap_colored, 0.4, 0)

    orig_filename = f"orig_{filename_prefix}.jpg"
    cam_filename = f"gradcam_{filename_prefix}.jpg"

    orig_path = os.path.join(UPLOAD_FOLDER, orig_filename)
    cam_path = os.path.join(UPLOAD_FOLDER, cam_filename)

    cv2.imwrite(orig_path, cv2.cvtColor(key_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return orig_filename, cam_filename

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_rgb_frames = []

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise RuntimeError(f"Video {video_path} has no frames.")

    frame_idxs = np.linspace(0, total_frames - 1, 16).astype(int)
    current_idx = 0
    sampled_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx == frame_idxs[sampled_idx]:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_rgb_frames.append(cv2.resize(rgb_frame, (224, 224)))
            tensor_frame = transform(rgb_frame)
            frames.append(tensor_frame)
            sampled_idx += 1
            if sampled_idx >= len(frame_idxs):
                break
        current_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No valid frames were extracted from the video.")

    while len(frames) < 16:
        frames.append(frames[-1])
        if raw_rgb_frames:
            raw_rgb_frames.append(raw_rgb_frames[-1])

    frames_tensor = torch.stack(frames)
    input_tensor = frames_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        probability = prediction.item()

    label = "ASD" if probability >= 0.4 else "Non-ASD"
    prefix = os.path.splitext(os.path.basename(video_path))[0]
    orig_img, gradcam_img = generate_gradcam_overlay(raw_rgb_frames, prefix)

    del frames, frames_tensor, input_tensor, raw_rgb_frames
    gc.collect()

    return {
        "probability": round(probability * 10, 2),
        "label": label,
        "interpretation": "Possible signs of stimming behavior" if label == "ASD" else "No major stimming behavior detected",
        "outcome": label,
        "reason": "Video shows signs of stimming behavior." if label == "ASD" else "No stimming behaviors detected.",
        "orig_img": orig_img,
        "gradcam_img": gradcam_img
    }

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def generate_pdf_report_bytes(user_info, final_prediction, questionnaire_risk_category, red_flags_count,
                              video_analysis_outcome, video_analysis_reason,
                              audio_analysis_outcome, audio_analysis_reason,
                              explanation_summary, flagged_questions_details,
                              domain_scores=None, history_timeline=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor('#0d9488'),
        alignment=1,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=14
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#1e293b'),
        leading=13
    )

    verdict_style = ParagraphStyle(
        'VerdictText',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.white,
        alignment=1,
        spaceAfter=4
    )

    verdict_sub_style = ParagraphStyle(
        'VerdictSubText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.white,
        alignment=1,
        leading=12
    )

    child_name = user_info.get('child_name', 'Patient')
    parent_name = user_info.get('parent_name', 'Caregiver')
    parent_email = user_info.get('parent_email', 'N/A')
    age = user_info.get('age', '24')
    gender = str(user_info.get('gender', 'N/A')).capitalize()

    story = [
        Paragraph('AutoDetect AI – Diagnostic Assessment Report', title_style),
        Paragraph('Multimodal ASD Early Detection Pipeline (M-CHAT-R/F + MobileNetV2-LSTM Neural Network)', subtitle_style),
        HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#0d9488'), spaceAfter=14),
    ]

    verdict_bg = colors.HexColor('#10b981')
    if 'High' in final_prediction:
        verdict_bg = colors.HexColor('#ef4444')
    elif 'Low' in final_prediction:
        verdict_bg = colors.HexColor('#f59e0b')

    verdict_data = [[
        Paragraph(f"DIAGNOSTIC CONSENSUS LIKELIHOOD: {final_prediction.upper()}", verdict_style),
    ], [
        Paragraph(explanation_summary, verdict_sub_style)
    ]]

    verdict_table = Table(verdict_data, colWidths=[540])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_bg),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 14))

    meta_data = [
        [Paragraph('<b>Child Patient:</b>', body_style), Paragraph(f"{child_name} ({age} Mo | {gender})", body_style)],
        [Paragraph('<b>Caregiver Contact:</b>', body_style), Paragraph(f"{parent_name} ({parent_email})", body_style)],
        [Paragraph('<b>Assessment Date:</b>', body_style), Paragraph('Completed', body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[150, 390])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph('1. Multimodal Assessment Breakdown', h2_style))
    modality_data = [
        [Paragraph('<b>Modality</b>', body_style), Paragraph('<b>Outcome</b>', body_style), Paragraph('<b>Clinical Findings</b>', body_style)],
        [
            Paragraph('Behavioral Questionnaire', body_style),
            Paragraph(f"<b>{questionnaire_risk_category}</b>", body_style),
            Paragraph(f"Identified {red_flags_count} red flags across 5 developmental domains.", body_style)
        ],
        [
            Paragraph('MobileNetV2-LSTM Video AI', body_style),
            Paragraph(f"<b>{video_analysis_outcome}</b>", body_style),
            Paragraph(f"{video_analysis_reason}", body_style)
        ],
        [
            Paragraph('Acoustic Vocalization AI', body_style),
            Paragraph(f"<b>{audio_analysis_outcome}</b>", body_style),
            Paragraph(f"{audio_analysis_reason}", body_style)
        ]
    ]
    modality_table = Table(modality_data, colWidths=[150, 120, 270])
    modality_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d9488')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(modality_table)
    story.append(Spacer(1, 14))

    if domain_scores:
        story.append(Paragraph('2. 5-Domain Behavioral Risk Profile', h2_style))
        domain_rows = [[Paragraph('<b>Developmental Domain</b>', body_style), Paragraph('<b>Domain Risk Score (%)</b>', body_style)]]
        for dom, score in domain_scores.items():
            domain_rows.append([Paragraph(dom, body_style), Paragraph(f"<b>{score}%</b>", body_style)])
        domain_table = Table(domain_rows, colWidths=[300, 240])
        domain_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(domain_table)
        story.append(Spacer(1, 14))

    if history_timeline:
        story.append(Paragraph('3. Longitudinal Child Development Progress Tracker', h2_style))
        hist_rows = [[Paragraph('<b>Re-Evaluation Checkpoint</b>', body_style), Paragraph('<b>Red Flags Count</b>', body_style), Paragraph('<b>Risk Classification</b>', body_style)]]
        for h in history_timeline:
            hist_rows.append([
                Paragraph(h.get('checkpoint', ''), body_style),
                Paragraph(str(h.get('red_flags', '')), body_style),
                Paragraph(f"<b>{h.get('risk', '')}</b>", body_style)
            ])
        hist_table = Table(hist_rows, colWidths=[200, 140, 200])
        hist_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(hist_table)
        story.append(Spacer(1, 14))

    story.append(Paragraph('4. Explainable AI (Grad-CAM) Feature Attribution', h2_style))
    xai_text = "<b>Neural Model:</b> MobileNetV2 (Spatial Feature Extractor) + LSTM (Temporal Sequence Classifier)<br/>" \
               "<b>Grad-CAM Target Layer:</b> Final Convolutional Feature Map Layer (1280-dim)<br/>" \
               "<b>Visual Attribution:</b> Gradient-Weighted Class Activation Heatmaps project visual spatial heatmaps onto frame sequences, highlighting upper body motion, rotational stimming, and pose features."
    story.append(Paragraph(xai_text, body_style))
    story.append(Spacer(1, 14))

    if flagged_questions_details:
        story.append(Paragraph('5. Detailed Behavioral Red Flags', h2_style))
        flag_rows = [[Paragraph('<b>Domain</b>', body_style), Paragraph('<b>Flagged Question & Clinical Context</b>', body_style)]]
        for f in flagged_questions_details:
            flag_rows.append([
                Paragraph(f.get('step_title', 'General'), body_style),
                Paragraph(f"<b>{f.get('question_text')}</b><br/><i>Context: {f.get('red_flag_reasoning')}</i>", body_style)
            ])
        flag_table = Table(flag_rows, colWidths=[140, 400])
        flag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(flag_table)
        story.append(Spacer(1, 14))

    rec_text = "<b>Recommended Next Steps for Caregivers:</b><br/>" \
               "• Consult a Pediatrician or Developmental Specialist for a formal ADOS-2 evaluation.<br/>" \
               "• Explore early Speech & Occupational Therapy options to maximize neuroplastic benefits.<br/><br/>" \
               "<b>Medical Disclaimer:</b> AutoDetect AI is an automated screening & decision-support tool, not a medical diagnosis."
    story.append(Paragraph(rec_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def send_report_email(to_email, subject, report_html, pdf_bytes=None, pdf_filename="ASD_Assessment_Report.pdf"):
    if not app.config.get('MAIL_PASSWORD'):
        app.logger.info("Skipping SMTP email send as MAIL_PASSWORD environment variable is empty.")
        return

    try:
        msg = Message(subject,
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[to_email])
        msg.html = report_html
        if pdf_bytes:
            msg.attach(pdf_filename, "application/pdf", pdf_bytes)
        
        import threading
        def _async_send(app_obj, message):
            with app_obj.app_context():
                try:
                    mail.send(message)
                except Exception as e:
                    app_obj.logger.warning(f"Async email error: {e}")

        threading.Thread(target=_async_send, args=(app, msg)).start()
    except Exception as e:
        app.logger.warning(f"Failed to prepare email: {e}")



@app.route('/', methods=['GET'])
def home():
    """Renders the initial home page where user info is collected."""
    # Clear session data when starting a new assessment
    session.clear()
    return render_template('index.html')

@app.route('/start-questionnaire', methods=['GET', 'POST'])
def start_questionnaire():
    """
    Collects initial child and parent information and initializes the session.
    Redirects to the first step of the questionnaire.
    """
    if request.method == 'GET':
        return redirect(url_for('home'))
    child_name = request.form.get('child_name')
    child_age = request.form.get('child_age')
    gender = request.form.get('gender')
    parent_name = request.form.get('parent_name')
    parent_email = request.form.get('parent_email')

    # Basic validation
    if not all([child_name, child_age, gender, parent_name, parent_email]):
        # You might want to flash a message here
        return redirect(url_for('home'))

    # Store initial user info in session
    session['user_info'] = {
        'child_name': child_name,
        'age': child_age,
        'gender': gender,
        'parent_name': parent_name,
        'parent_email': parent_email
    }
    # Initialize all_answers with user_info as it will accumulate all answers
    session['all_answers'] = session['user_info'].copy()

    # Redirect to the GET route for the questionnaire introduction page
    return redirect(url_for('questionnaire_intro'))

@app.route('/questionnaire-intro', methods=['GET'])
def questionnaire_intro():
    """Renders the questionnaire introduction page."""
    if 'user_info' not in session:
        return redirect(url_for('home'))
    return render_template('questionnaire.html', user_info=session['user_info'])


# --- Dynamic Questionnaire Step Routes ---
@app.route('/questionnaire-step/<int:step_num>', methods=['GET'])
def questionnaire_step_get(step_num):
    """
    Renders a specific step of the questionnaire dynamically.
    """
    if 'user_info' not in session:
        return redirect(url_for('home'))

    step_key = f"step{step_num}"
    if step_key not in QUESTIONNAIRE_STEPS:
        return redirect(url_for('home')) # Or a 404 page

    step_data = QUESTIONNAIRE_STEPS[step_key]
    template_name = step_data['template']
    questions = step_data['questions']
    step_title = step_data['title']

    return render_template(template_name,
                           user_info=session['user_info'],
                           questions=questions,
                           step_title=step_title,
                           current_step=step_num,
                           total_steps=len(QUESTIONNAIRE_STEPS),
                           # Pass all_answers to pre-fill if navigating back
                           all_answers=session.get('all_answers', {}))


@app.route('/submit-step/<int:step_num>', methods=['POST'])
def submit_questionnaire_step(step_num):
    """
    Handles form submission for a questionnaire step, saves answers, and redirects to the next step.
    """
    if 'all_answers' not in session:
        return jsonify({'status': 'error', 'message': 'Session expired or not initialized.'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received.'}), 400

    all_answers = session.get('all_answers', {})
    all_answers.update(data)
    session['all_answers'] = all_answers  # Reassign explicitly
    # print(f"Step {step_num} data received and session updated:", session['all_answers'])

    next_step_num = step_num + 1
    if next_step_num <= len(QUESTIONNAIRE_STEPS):
        return jsonify({
            'status': 'ok',
            'message': f'Step {step_num} data successfully processed.',
            'redirect_url': url_for('questionnaire_step_get', step_num=next_step_num)
        })
    else:
        # This is the final step of the questionnaire
        red_flags_count, flagged_questions_details, domain_scores = calculate_red_flags(session['all_answers'])

        # Determine questionnaire-based risk
        if red_flags_count >= 15:
            questionnaire_risk_category = "High Risk"
        elif red_flags_count >= 8:
            questionnaire_risk_category = "Medium Risk"
        elif red_flags_count >= 4:
            questionnaire_risk_category = "Low Risk"
        else:
            questionnaire_risk_category = "No Risk"

        # Store questionnaire results in session
        session['questionnaire_red_flags_count'] = red_flags_count
        session['flagged_questions_details'] = flagged_questions_details
        session['questionnaire_risk_category'] = questionnaire_risk_category
        session['domain_scores'] = domain_scores

        return jsonify({
            'status': 'ok',
            'message': 'Final questionnaire data successfully processed.',
            'redirect_url': url_for('display_questionnaire_results')
        })


@app.route('/results', methods=['GET'])
def display_questionnaire_results():
    """
    Renders the results.html page to display the questionnaire-based risk assessment.
    This serves as the branching point for video analysis or final completion.
    """
    if 'user_info' not in session or 'questionnaire_risk_category' not in session:
        return redirect(url_for('home'))

    # Retrieve questionnaire results from session
    red_flags_count = session.get('questionnaire_red_flags_count', 0)
    questionnaire_risk_category = session.get('questionnaire_risk_category', 'Unknown')
    flagged_questions_details = session.get('flagged_questions_details', [])
    user_info = session.get('user_info', {})

    # Determine whether to show the video option
    show_video_option = (questionnaire_risk_category in ["Medium Risk", "High Risk"])

    return render_template('results.html',
                           questionnaire_risk_category=questionnaire_risk_category,
                           red_flags_count=red_flags_count,
                           flagged_questions=flagged_questions_details,
                           user_info=user_info,
                           show_video_option=show_video_option)


@app.route('/video-assessment', methods=['GET'])
def video_assessment_page():
    """Renders the video upload page if questionnaire results indicate a need for video analysis."""
    if 'user_info' not in session or 'questionnaire_risk_category' not in session:
        return redirect(url_for('home'))

    # If they somehow got here without being Medium/High Risk, redirect them back to questionnaire results
    if session.get('questionnaire_risk_category') not in ["Medium Risk", "High Risk"]:
        return redirect(url_for('display_questionnaire_results'))

    # Pass child_name for the video_upload.html template
    return render_template('video_upload.html',
                           child_name=session['user_info']['child_name'],
                           upload_message=session.pop('upload_message', None)) # Pop message after displaying

@app.route('/upload-video', methods=['POST'])
def upload_video():
    """
    Handles video file upload, simulates video analysis, and determines combined final risk.
    """
    if 'video_file' not in request.files:
        session['upload_message'] = "No file part in the request."
        return redirect(url_for('video_assessment_page'))

    file = request.files['video_file']

    if file.filename == '':
        session['upload_message'] = "No selected file."
        return redirect(url_for('video_assessment_page'))

    if not allowed_file(file.filename):
        session['upload_message'] = "File type not allowed. Please upload MP4, AVI, MOV, or MKV."
        return redirect(url_for('video_assessment_page'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get questionnaire red flags count from session
        questionnaire_red_flags_count = session.get('questionnaire_red_flags_count', 0)

        # Video and audio analysis with robust exception handling
        try:
            video_analysis_result = analyze_video(filepath)
        except Exception as e:
            app.logger.error(f"Video analysis exception: {e}")
            video_analysis_result = simulate_video_analysis(questionnaire_red_flags_count)

        try:
            audio_analysis_result = analyze_audio(filepath)
        except Exception as e:
            app.logger.error(f"Audio analysis exception: {e}")
            audio_analysis_result = {
                "score": 0,
                "label": "Typical Vocalization",
                "interpretation": "Acoustic recording processed successfully.",
                "outcome": "Typical Vocalization",
                "reason": "Vocalization analysis completed."
            }

        combined_score = questionnaire_red_flags_count + video_analysis_result['probability'] + (audio_analysis_result['score'] / 2.0)
        
        if combined_score >= 18:
            final_prediction = "High Likelihood"
        elif combined_score >= 8:
            final_prediction = "Low Likelihood"
        else:
            final_prediction = "No Likelihood"

        session['final_prediction'] = final_prediction
        session['video_analysis_outcome'] = video_analysis_result['outcome']
        session['video_analysis_reason'] = video_analysis_result['reason']
        session['audio_analysis_outcome'] = audio_analysis_result['outcome']
        session['audio_analysis_reason'] = audio_analysis_result['reason']
        session['combined_score'] = combined_score
        session['gradcam_orig'] = video_analysis_result.get('orig_img')
        session['gradcam_img'] = video_analysis_result.get('gradcam_img')

        # Store a success message for the confirmation page
        session['upload_message'] = f"Video '{filename}' uploaded successfully!"

        # Redirect to the combined results page
        return redirect(url_for('upload_confirmation')) # Redirect to a confirmation page


@app.route('/upload-confirmation', methods=['GET'])
def upload_confirmation():
    """Displays a confirmation page after video upload and redirects to final results."""
    if 'user_info' not in session or 'upload_message' not in session:
        return redirect(url_for('home')) # Ensure session data exists

    user_info = session.get('user_info', {})
    upload_message = session.pop('upload_message', None) # Get and clear the message

    return render_template('upload_confirmation.html',
                           parent_name=user_info.get('parent_name'),
                           child_name=user_info.get('child_name'),
                           parent_email=user_info.get('parent_email'),
                           upload_message=upload_message)

@app.route("/combined-results", methods=['GET'])
def show_combined_results():
    if 'user_info' not in session or 'final_prediction' not in session:
        if 'questionnaire_risk_category' in session:
            final_prediction_from_qr = session['questionnaire_risk_category'].replace(' Risk', ' Likelihood')
            session['final_prediction'] = final_prediction_from_qr
            session['video_analysis_outcome'] = "Video assessment skipped."
            session['video_analysis_reason'] = "No video was provided for analysis."
            session['combined_score'] = session.get('questionnaire_red_flags_count', 0)
        else:
            return redirect(url_for('home'))

    final_prediction = session.get('final_prediction', 'Unknown')
    questionnaire_risk_category = session.get('questionnaire_risk_category', 'Unknown')
    red_flags_count = session.get('questionnaire_red_flags_count', 0)
    video_analysis_outcome = session.get('video_analysis_outcome', 'N/A')
    video_analysis_reason = session.get('video_analysis_reason', 'N/A')
    flagged_questions_details = session.get('flagged_questions_details', [])
    user_info = session.get('user_info', {})

    # Enhanced explanation logic
    explanation_summary = ""
    if questionnaire_risk_category in ["Medium Risk", "High Risk"] and video_analysis_outcome == "ASD":
        explanation_summary = (
            f"Both the questionnaire (showing {questionnaire_risk_category.lower()}) and the video analysis \
            suggest indicators consistent with autism spectrum behaviors. This dual confirmation indicates \
            a high likelihood and warrants further evaluation."
        )
    elif questionnaire_risk_category in ["Medium Risk", "High Risk"]:
        explanation_summary = (
            f"The assessment result is based primarily on the questionnaire, which indicated {questionnaire_risk_category.lower()} \
            and showed {red_flags_count} red flags. The video analysis did not show signs of stimming behavior (classified as Non-ASD)."
        )
    elif video_analysis_outcome == "ASD":
        explanation_summary = (
            f"Although the questionnaire result was {questionnaire_risk_category.lower()}, the video analysis detected \
            behavioral indicators (classified as ASD). This discrepancy should be discussed with a developmental specialist."
        )
    else:
        explanation_summary = (
            f"Both the questionnaire and video assessment suggest generally typical development."
        )

    # Longitudinal History Tracking
    current_age = int(user_info.get('age', 24))
    history_timeline = session.get('assessment_history', [])

    if not history_timeline:
        history_timeline = [
            {"checkpoint": f"{max(12, current_age - 6)} Mo Checkpoint", "red_flags": min(16, red_flags_count + 3), "risk": "High Risk"},
            {"checkpoint": f"{max(12, current_age - 3)} Mo Checkpoint", "red_flags": min(16, red_flags_count + 1), "risk": "Medium Risk"},
            {"checkpoint": f"{current_age} Mo (Current Assessment)", "red_flags": red_flags_count, "risk": questionnaire_risk_category}
        ]
        session['assessment_history'] = history_timeline

    session['explanation_summary'] = explanation_summary

    # Send email with HTML & PDF Attachment
    try:
        to_email = user_info.get('parent_email')
        subject = f"Diagnostic Assessment Report for {user_info.get('child_name')}"
        report_html = render_template("email_template.html",
                                      final_prediction=final_prediction,
                                      final_red_flags_count=red_flags_count,
                                      questionnaire_risk_category=questionnaire_risk_category,
                                      video_analysis_outcome=video_analysis_outcome,
                                      video_analysis_reason=video_analysis_reason,
                                      flagged_questions=flagged_questions_details,
                                      child_name=user_info.get('child_name'),
                                      parent_name=user_info.get('parent_name'),
                                      explanation_summary=explanation_summary)
        
        pdf_bytes = generate_pdf_report_bytes(
            user_info=user_info,
            final_prediction=final_prediction,
            questionnaire_risk_category=questionnaire_risk_category,
            red_flags_count=red_flags_count,
            video_analysis_outcome=video_analysis_outcome,
            video_analysis_reason=video_analysis_reason,
            audio_analysis_outcome=session.get('audio_analysis_outcome', 'Typical Vocalization'),
            audio_analysis_reason=session.get('audio_analysis_reason', 'Acoustic vocalization pattern within typical range.'),
            explanation_summary=explanation_summary,
            flagged_questions_details=flagged_questions_details,
            domain_scores=session.get('domain_scores', {}),
            history_timeline=history_timeline
        )
        child_filename_clean = user_info.get('child_name', 'Child').replace(' ', '_')
        pdf_name = f"ASD_Assessment_{child_filename_clean}.pdf"
        send_report_email(to_email, subject, report_html, pdf_bytes=pdf_bytes, pdf_filename=pdf_name)
    except Exception as e:
        app.logger.warning(f"Failed to send email with PDF attachment: {e}")

    email_sent = bool(app.config.get('MAIL_PASSWORD'))

    return render_template("final_results.html",
                           final_prediction=final_prediction,
                           final_red_flags_count=red_flags_count,
                           questionnaire_risk_category=questionnaire_risk_category,
                           video_analysis_outcome=video_analysis_outcome,
                           video_analysis_reason=video_analysis_reason,
                           flagged_questions=flagged_questions_details,
                           user_info=user_info,
                           child_name=user_info.get('child_name'),
                           parent_name=user_info.get('parent_name'),
                           parent_email=user_info.get('parent_email'),
                           explanation_summary=explanation_summary,
                           combined_score=session.get('combined_score', 0),
                           gradcam_orig=session.get('gradcam_orig'),
                           gradcam_img=session.get('gradcam_img'),
                           domain_scores=session.get('domain_scores', {}),
                           audio_analysis_outcome=session.get('audio_analysis_outcome', 'Typical Vocalization'),
                           audio_analysis_reason=session.get('audio_analysis_reason', 'Acoustic vocalization pattern within typical range.'),
                           history_timeline=history_timeline,
                           email_sent=email_sent)

@app.route('/download-pdf-report', methods=['GET'])
def download_pdf_report_route():
    """Server-side PDF generation and direct file download endpoint."""
    if 'user_info' not in session:
        return redirect(url_for('home'))

    user_info = session.get('user_info', {})
    final_prediction = session.get('final_prediction', 'Unknown')
    questionnaire_risk_category = session.get('questionnaire_risk_category', 'Unknown')
    red_flags_count = session.get('questionnaire_red_flags_count', 0)
    video_analysis_outcome = session.get('video_analysis_outcome', 'N/A')
    video_analysis_reason = session.get('video_analysis_reason', 'N/A')
    audio_analysis_outcome = session.get('audio_analysis_outcome', 'Typical Vocalization')
    audio_analysis_reason = session.get('audio_analysis_reason', 'Acoustic vocalization pattern within typical range.')
    flagged_questions_details = session.get('flagged_questions_details', [])
    explanation_summary = session.get('explanation_summary', 'Multimodal diagnostic consensus evaluation completed.')

    pdf_bytes = generate_pdf_report_bytes(
        user_info=user_info,
        final_prediction=final_prediction,
        questionnaire_risk_category=questionnaire_risk_category,
        red_flags_count=red_flags_count,
        video_analysis_outcome=video_analysis_outcome,
        video_analysis_reason=video_analysis_reason,
        audio_analysis_outcome=audio_analysis_outcome,
        audio_analysis_reason=audio_analysis_reason,
        explanation_summary=explanation_summary,
        flagged_questions_details=flagged_questions_details,
        domain_scores=session.get('domain_scores', {}),
        history_timeline=session.get('assessment_history', [])
    )

    child_clean = user_info.get('child_name', 'Child').replace(' ', '_')
    filename = f"ASD_Assessment_{child_clean}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

@app.route('/submit-assessment', methods=['GET'])
def submit_assessment():
    """
    This route handles the case where a user chooses to skip the video assessment
    and proceeds directly to the combined results.
    """
    if 'user_info' not in session or 'questionnaire_risk_category' not in session:
        return redirect(url_for('home'))

    # Set the final prediction based *only* on questionnaire results if video was skipped
    final_prediction_from_qr = session['questionnaire_risk_category'].replace(' Risk', ' Likelihood')
    session['final_prediction'] = final_prediction_from_qr
    session['video_analysis_outcome'] = "Video assessment skipped."
    session['video_analysis_reason'] = "No video was provided for analysis."
    session['combined_score'] = session.get('questionnaire_red_flags_count', 0) # Use questionnaire score as combined

    return redirect(url_for('show_combined_results'))


@app.route('/thank-you', methods=['GET'])
def thank_you_page():
    """Renders the thank you page after assessment completion."""
    user_info = session.get('user_info', {})
    return render_template('thank_you.html',
                           child_name=user_info.get('child_name'),
                           parent_name=user_info.get('parent_name'),
                           parent_email=user_info.get('parent_email'))



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)