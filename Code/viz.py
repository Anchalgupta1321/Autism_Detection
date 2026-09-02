import os
import cv2
import matplotlib.pyplot as plt

# --- Setup ---
data_dir = "UST_Data"
categories = ["ASD_Behaviour", "non_ASD_Behaviour"]
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

video_counts = []
total_durations = []
average_durations = []
total_sizes = []

def get_video_duration_and_size(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    return duration, size_mb

# --- Loop over categories ---
for category in categories:
    folder = os.path.join(data_dir, category)
    videos = [f for f in os.listdir(folder) if f.lower().endswith(video_extensions)]
    video_counts.append(len(videos))

    durations = []
    sizes = []
    for video_file in videos:
        filepath = os.path.join(folder, video_file)
        dur, size = get_video_duration_and_size(filepath)
        durations.append(dur)
        sizes.append(size)
    
    total_durations.append(sum(durations))
    average_durations.append(sum(durations) / len(durations) if durations else 0)
    total_sizes.append(sum(sizes))

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Pie Chart - Video count ratio
axs[0, 0].pie(video_counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
axs[0, 0].set_title("Video Count Distribution")

# 2. Bar Chart - Average durations
axs[0, 1].bar(categories, average_durations, color=['#ff9999','#66b3ff'])
axs[0, 1].set_title("Average Video Duration (seconds)")
axs[0, 1].set_ylabel("Seconds")

# 3. Bar Chart - Total durations
axs[1, 0].bar(categories, total_durations, color=['#ff9999','#66b3ff'])
axs[1, 0].set_title("Total Video Duration (seconds)")
axs[1, 0].set_ylabel("Seconds")

# 4. Bar Chart - Total sizes
axs[1, 1].bar(categories, total_sizes, color=['#ff9999','#66b3ff'])
axs[1, 1].set_title("Total Size of Videos (MB)")
axs[1, 1].set_ylabel("MB")

plt.tight_layout()
plt.show()
