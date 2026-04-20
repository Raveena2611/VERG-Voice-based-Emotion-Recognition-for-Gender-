Project Overview

This project detects human emotion and gender from speech audio using Machine Learning and Deep Learning techniques.

It combines multiple models to improve performance:

Random Forest (ML Model)
LSTM (Deep Learning Model)
Stacking Ensemble Model

The system predicts combined labels such as:

happy_male
sad_female
angry_male
neutral_female

# Technologies Used
Python
NumPy
Librosa (Audio Processing)
Scikit-learn
TensorFlow / Keras
Matplotlib & Seaborn

# Dataset
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Contains labeled emotional speech audio from multiple actors

Download: http s://zenodo.org/record/1188976

# Workflow
Load audio files
Apply data augmentation
Extract features (MFCC, etc.)
Train models:
Random Forest
LSTM
Combine predictions using stacking
Evaluate results

# How to Run
1. Clone Repository
git clone https://github.com/Raveena2611/VERG-Voice-based-Emotion-Recognition-for-Gender.git
cd VERG-Voice-based-Emotion-Recognition-for-Gender
2. Create Environment (Recommended)
conda create -n verg_env python=3.10
conda activate verg_env
3. Install Dependencies
pip install -r requirements.txt
4. Run Project
python main.py

## Model Details

# Random Forest
Uses extracted features like MFCC
Fast training and good baseline performance
# LSTM
Learns sequential patterns from audio
Better for capturing temporal dependencies
# Stacking Ensemble
Combines predictions from RF + LSTM
Improves overall accuracy

# Output
The project generates:

Accuracy score
Classification report
Confusion matrix
Model comparison graph
