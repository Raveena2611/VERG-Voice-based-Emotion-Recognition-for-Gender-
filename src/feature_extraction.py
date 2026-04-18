import numpy as np
import librosa
import random

SR = 22050
DURATION = 3.0
OFFSET = 0.5

# ---------------- SPEC AUGMENT ----------------
def spec_augment_mfcc(mfcc, max_mask_pct=0.10, n_freq_masks=2, n_time_masks=2, p=0.5):
    if random.random() > p:
        return mfcc

    mfcc = mfcc.copy()
    n_mfcc, n_frames = mfcc.shape

    for _ in range(random.randint(1, n_freq_masks)):
        f = random.randint(1, max(1, int(n_mfcc * max_mask_pct)))
        f0 = random.randint(0, max(0, n_mfcc - f))
        mfcc[f0:f0+f, :] = 0.0

    for _ in range(random.randint(1, n_time_masks)):
        t = random.randint(1, max(1, int(n_frames * max_mask_pct)))
        t0 = random.randint(0, max(0, n_frames - t))
        mfcc[:, t0:t0+t] = 0.0

    return mfcc

# ---------------- LOAD AUDIO ----------------
def load_audio_3sec(path):
    y, sr = librosa.load(path, sr=SR, duration=DURATION, offset=OFFSET)
    y, _ = librosa.effects.trim(y, top_db=30)

    target_len = int(DURATION * SR)
    if len(y) > target_len:
        y = y[:target_len]
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    return y, sr

# ---------------- RF FEATURES ----------------
def features_for_rf(y, sr):
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    y_harm = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr), axis=1)

    zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])
    rms = np.array([np.mean(librosa.feature.rms(y=y))])
    sc = np.array([np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))])
    rolloff = np.array([np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))])
    bw = np.array([np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))])

    return np.concatenate([
        mfcc_mean, mfcc_delta, mfcc_delta2,
        chroma, mel, contrast, tonnetz,
        zcr, rms, sc, rolloff, bw
    ]).astype(np.float32)

# ---------------- LSTM FEATURES ----------------
def features_for_lstm(y, sr, n_mfcc=40, max_len=130):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = spec_augment_mfcc(mfcc)

    mfcc = mfcc.T
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    if len(mfcc) > max_len:
        mfcc = mfcc[:max_len]
    else:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)))

    return mfcc.astype(np.float32)