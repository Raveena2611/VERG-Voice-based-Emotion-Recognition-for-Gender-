import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_loader import load_files, parse_combined_label
from src.feature_extraction import load_audio_3sec, features_for_rf, features_for_lstm
from src.augmentation import AUG_FUNCS
from src.train_rf import train_rf
from src.train_lstm import train_lstm
from src.stacking import train_stacking, stacking_predict

if __name__ == "__main__":
    print("Training Started")

    DATASET_PATH = r"F:\MY study material\VERG\RAVDESS\audio_speech_actors_01-24"

    files = load_files(DATASET_PATH)

    X_rf, X_seq, y = [], [], []

    for fp in files:
        label = parse_combined_label(fp)

        y_audio, sr = load_audio_3sec(fp)

        # original
        X_rf.append(features_for_rf(y_audio, sr))
        X_seq.append(features_for_lstm(y_audio, sr))
        y.append(label)

        # augmentation
        for _ in range(3):
            aug = random.choice(AUG_FUNCS)
            y_aug = aug(y_audio.copy()) if aug.__name__ != "aug_pitch" else aug(y_audio.copy(), sr)

            X_rf.append(features_for_rf(y_aug, sr))
            X_seq.append(features_for_lstm(y_aug, sr))
            y.append(label)

    X_rf = np.array(X_rf)
    X_seq = np.array(X_seq)
    y = np.array(y)

    print("Dataset ready:", X_rf.shape)

    # encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # split
    X_rf_tr, X_rf_te, X_seq_tr, X_seq_te, y_tr, y_te = train_test_split(
        X_rf, X_seq, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # scaling
    scaler = StandardScaler()
    X_rf_tr = scaler.fit_transform(X_rf_tr)
    X_rf_te = scaler.transform(X_rf_te)

    # RF
    print("Training RF...")
    rf = train_rf(X_rf_tr, y_tr)

    # LSTM
    print("Training LSTM...")
    lstm = train_lstm(X_seq_tr, y_tr, X_seq_te, y_te, len(le.classes_))

    # Stacking
    print("Training Stacking...")
    meta = train_stacking(rf, lstm, X_rf_te, X_seq_te, y_te)

    preds = stacking_predict(rf, lstm, meta, X_rf_te, X_seq_te)

    from sklearn.metrics import accuracy_score
    print("Final Accuracy:", accuracy_score(y_te, preds))

    from src.utils import create_output_dir, plot_confusion_matrix, print_metrics

create_output_dir()

# RF Predictions
rf_pred = rf.predict(X_rf_te)
plot_confusion_matrix(y_te, rf_pred, le.classes_, "RF Confusion Matrix", "rf_cm.png")
print_metrics(y_te, rf_pred, le.classes_, "Random Forest")

# LSTM Predictions
lstm_pred = np.argmax(lstm.predict(X_seq_te), axis=1)
plot_confusion_matrix(y_te, lstm_pred, le.classes_, "LSTM Confusion Matrix", "lstm_cm.png")
print_metrics(y_te, lstm_pred, le.classes_, "LSTM")

# Stacking Predictions
stack_pred = stacking_predict(rf, lstm, meta, X_rf_te, X_seq_te)
plot_confusion_matrix(y_te, stack_pred, le.classes_, "Stacking Confusion Matrix", "stack_cm.png")
print_metrics(y_te, stack_pred, le.classes_, "Stacking")

from src.utils import overall_metrics, plot_model_comparison

rf_acc, rf_sens, rf_spec = overall_metrics(y_te, rf_pred)
lstm_acc, lstm_sens, lstm_spec = overall_metrics(y_te, lstm_pred)
stack_acc, stack_sens, stack_spec = overall_metrics(y_te, stack_pred)

plot_model_comparison(
    ['RF', 'LSTM', 'Stacking'],
    [rf_acc, lstm_acc, stack_acc],
    [rf_sens, lstm_sens, stack_sens],
    [rf_spec, lstm_spec, stack_spec]
)
