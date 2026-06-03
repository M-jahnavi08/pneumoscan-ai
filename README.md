# 🫁 PneumoScan AI — Chest X-Ray Pneumonia Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

An end-to-end deep learning system that detects **pneumonia** from chest X-rays
using EfficientNetB3 with **Grad-CAM explainability**.

## 🌐 Live Demo
👉 [Try PneumoScan AI Live]((https://pneumoscan-ai-ljkf8rt3a7vsfpkmd8pxqi.streamlit.app/))

## 🎯 Results
| Metric | Score |
|--------|-------|
| Test Accuracy | ~94% |
| Architecture | EfficientNetB3 |
| Dataset Size | 5,863 images |
| Explainability | Grad-CAM |

## 💡 Features
- ✅ Binary classification: NORMAL vs PNEUMONIA
- 🔥 Grad-CAM heatmaps showing WHERE the AI looks
- ⚖️ Class imbalance handling with weighted loss
- 🚀 Deployed interactive web app
- 🎨 Professional dark medical UI

## 🛠️ Tech Stack
PyTorch · EfficientNetB3 · Grad-CAM · Streamlit · OpenCV · Scikit-learn

## 📁 Project Structure
chest-xray-ai/
├── app.py           # Streamlit web application
├── train.py         # Model training script
├── gradcam_viz.py   # Grad-CAM explainability
├── explore.py       # Data exploration
└── requirements.txt

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

## ⚠️ Disclaimer
For educational purposes only. Not a certified medical device.
