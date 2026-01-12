# 🛡️ DeepGuard – Multimodal Deepfake Detection System

DeepGuard is an AI-powered multimodal deepfake detection system that analyzes both images and audio to identify manipulated digital media. It uses deep learning models to detect visual artifacts in images and synthetic patterns in audio, helping users verify media authenticity through a simple web interface.

---

## 🚀 Features

- Image deepfake detection using CNN  
- Audio deepfake detection using Mel-spectrogram based 1D-CNN  
- Upload and analyze media files in real time  
- Displays Real / Fake result with confidence score  
- Clean and responsive web interface  
- Supports JPG, PNG, WAV, MP3, and M4A formats  

---

## 🛠️ Tech Stack

**Frontend**
- HTML
- CSS
- JavaScript

**Backend**
- Python
- Flask

**AI / Machine Learning**
- TensorFlow
- Keras
- Librosa
- OpenCV
- Pillow

---

## 📂 Project Structure

DeepGuard
├── data
│   ├── images
│   └── audio
├── models
│   ├── image_model.h5
│   └── audio_model.h5
├── static
├── templates
├── app.py
├── train_image_cnn.py
├── train_audio_cnn.py
├── prep_audio_to_mels.py
└── README.md

---

## ⚙️ How to Run the Project

### 1. Clone the repository
git clone https://github.com/Khanishk18/DeepGuard.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the models
python train_image_cnn.py
python prep_audio_to_mels.py
python train_audio_cnn.py

### 4. Run the web application
python app.py

Open in browser:
http://127.0.0.1:5000

---

## 👨‍💻 Author

**Khanishk Narra**  
B.Tech CSE Student  

---

## ⭐ If you like this project
Give it a ⭐ on GitHub 😊
