# Autism Spectrum Disorder Detection – Flutter Mobile App

This repository contains a **Flutter mobile application** for **Autism Spectrum Disorder (ASD) screening** using **two AI models**:

1. **Facial Expression Model (CNN – ResNet50)**
2. **Eye-Tracking Model (LBP + GLCM + ANN)**

The app provides a **live scan interface** that captures an image from the device camera, sends it to the backend, and displays the prediction and confidence score.

---

## 1. Features

- 📱 **Flutter Mobile App**
  - Live camera scan
  - Image preview before sending
  - Result screen with class label & confidence
  - Simple and clean UI

- 🧠 **Two AI Models**
  - **Facial Expression Model**
    - ResNet50 transfer learning
    - Trained on facial images (ASD vs Non-ASD)
  - **Eye-Tracking Model**
    - LBP + GLCM feature extraction
    - ANN (MLPClassifier) for classification
    - Uses generated scan-path images

- ☁️ **Backend API**
  - Receives uploaded image from the app (POST)
  - Preprocesses the image
  - Runs inference on selected model
  - Returns JSON response:
    ```json
    {
      "model": "facial" | "eye_tracking",
      "prediction": "ASD" | "Non-ASD",
      "probabilities": {
        "ASD": 0.87,
        "Non-ASD": 0.13
      }
    }
    ```

---

## 2. Project Architecture

```text
Flutter Mobile App
  ├── Live Scan (Camera)
  ├── Image Selection (Gallery)
  ├── Choose Model: [Facial | Eye-Tracking]
  └── Sends image → Backend API (HTTP POST)

Backend (Python / Flask or Firebase Function)
  ├── Endpoint: /predict/facial
  │     ├── Load model_5.keras (ResNet50 fine-tuned)
  │     ├── Resize (224x224), normalize
  │     └── Return ASD / Non-ASD probabilities
  └── Endpoint: /predict/eye-tracking
        ├── Load eye_tracking_model.pkl + scaler.pkl
        ├── Grayscale → Hist. Equalization → Gaussian Blur
        ├── LBP histogram + GLCM properties
        ├── Concatenate → 260-dim vector → StandardScaler
        └── Return ASD / Non-ASD probabilities
