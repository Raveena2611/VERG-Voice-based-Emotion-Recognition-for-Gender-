import os
import glob

EMO_MAP = {
    "01":"neutral", "02":"calm", "03":"happy", "04":"sad",
    "05":"angry", "06":"fearful", "07":"disgust", "08":"surprised"
}

def parse_combined_label(fname):
    base = os.path.basename(fname)
    parts = base.split('-')
    if len(parts) >= 7:
        emotion_code = parts[2]
        actor_id = int(parts[6].split('.')[0])
        emotion = EMO_MAP.get(emotion_code)
        gender = 'male' if actor_id % 2 != 0 else 'female'
        if emotion and gender:
            return f"{emotion}_{gender}"
    return None

def load_files(dataset_path):
    files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    return [f for f in files if parse_combined_label(f) is not None]