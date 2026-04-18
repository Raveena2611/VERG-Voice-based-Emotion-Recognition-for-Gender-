import numpy as np
import random
import librosa

def aug_pitch(y, sr):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-2, -1, 1, 2]))

def aug_stretch(y):
    rate = random.choice([0.85, 0.9, 1.1, 1.15])
    try:
        y2 = librosa.effects.time_stretch(y, rate)
        if len(y2) > len(y): y2 = y2[:len(y)]
        else: y2 = np.pad(y2, (0, len(y)-len(y2)))
        return y2
    except:
        return y

def aug_noise(y):
    return y + np.random.randn(len(y)) * 0.004

def aug_shift(y):
    return np.roll(y, int(random.uniform(-0.15, 0.15) * len(y)))

def aug_volume(y):
    return y * random.uniform(0.8, 1.25)

AUG_FUNCS = [aug_pitch, aug_stretch, aug_noise, aug_shift, aug_volume]