# Speech Emotion & Gender Detection System

## Project Overview

This project is designed to detect **human emotion and gender from speech audio** using Machine Learning and Deep Learning techniques.

It uses:

* Random Forest (ML Model)
* LSTM (Deep Learning Model)
* Stacking Ensemble Model

The system predicts combined labels such as:

* `happy_male`
* `sad_female`
* `angry_male`
* `neutral_female`

## Project Structure

Speech-Emotion-Detection/
│
├── src/
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── feature_extraction.py
│   ├── train_rf.py
│   ├── train_lstm.py
│   └── stacking.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Librosa (Audio Processing)
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib & Seaborn

---

## 📊 Dataset

* Dataset used: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
* Contains emotional speech samples from multiple actors

⚠️ Note:
Dataset is not included due to large size.
Download it separately and place inside your project folder.

---

## 🚀 How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/speech-emotion-project.git
cd speech-emotion-project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Project

```bash
python main.py
```

---

## 🧠 Model Details

### 🔹 Random Forest

* Uses statistical audio features (MFCC)
* Fast and effective baseline model

### 🔹 LSTM Model

* Sequence-based learning
* Captures temporal patterns in speech

### 🔹 Stacking Model

* Combines RF + LSTM predictions
* Improves overall performance

---

## 📈 Output

* Emotion + Gender Prediction
* Model Training Output
* Accuracy Results

---

## ⚠️ Important Notes

* Make sure dataset path is correct in `main.py`
* Use Python 3.8+ for compatibility
* GPU recommended for faster LSTM training

---

## 👩‍💻 Author

**Raveena Rajak**

---

## 📌 Future Improvements

* Real-time voice prediction
* Web deployment (Flask / FastAPI)
* Improve accuracy with advanced models

---
