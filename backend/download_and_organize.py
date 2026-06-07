import os
import shutil
import pandas as pd
import kagglehub

# 1. Force kagglehub to download the dataset cleanly
print("Starting dataset download via Kaggle API... Please wait.")
try:
    downloaded_path = kagglehub.dataset_download("rollas/artemis-dataset-including-10k-images")
    print(f"Dataset files successfully downloaded to: {downloaded_path}")
except Exception as e:
    print(f"Download Error: {str(e)}")
    print("If this fails, please download the ZIP manually from Kaggle.")
    exit()

# 2. Setup your target clean project dataset directory 
TARGET_DIR = "./My_Art_Data"

EMOTION_MAP = {
    "contentment": "Calmness",
    "awe": "Joy",
    "amusement": "Joy",
    "sadness": "Sadness",
    "fear": "Fear",
    "anger": "Anger"
}

# 3. Locate the CSV spreadsheet file dynamically
csv_path = None
images_dir = None

for root, dirs, files in os.walk(downloaded_path):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
    if "images" in dirs:
        images_dir = os.path.join(root, "images")

if not csv_path or not images_dir:
    print("Error: Could not locate the CSV data sheet or the images folder inside the downloaded package.")
    exit()

print(f"Found Metadata Sheet at: {csv_path}")
print(f"Found Raw Images Folder at: {images_dir}")

# 4. Read the data sheet and sort images into your 5 emotions folders
df = pd.read_csv(csv_path)
print("Sorting artwork images into 5 final emotion training splits...")

success_count = 0

for index, row in df.iterrows():
    raw_emotion = str(row['emotion']).strip().lower()
    filename = str(row['filename']).strip()
    
    if raw_emotion in EMOTION_MAP:
        target_class = EMOTION_MAP[raw_emotion]
        split_folder = "train" if (index % 5 != 0) else "validation"
        src_img_path = os.path.join(images_dir, filename)
        
        if os.path.exists(src_img_path):
            dest_dir = os.path.join(TARGET_DIR, split_folder, target_class)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_img_path, os.path.join(dest_dir, filename))
            success_count += 1

print(f"Finished processing. Successfully sorted {success_count} images into your project target folder.")
print(f"Data directories are completely built inside: {os.path.abspath(TARGET_DIR)}")