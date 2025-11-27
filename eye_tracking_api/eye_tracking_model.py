# ASD Eye Tracking Diagnosis Based on Research Article
# Model based on: Hybrid of LBP + GLCM + ANN / FFNN as per Electronics 2022, 11(4), 530

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from tqdm import tqdm
from collections import Counter
import joblib

# === CONFIG ===
IMAGE_DIR = r'C:\Users\Yaven_Singh\Desktop\My Eye Tracking Model\My Eye Tracking Model\Eye Tracking Dataset\Images'
IMAGE_SIZE = 224

# === HELPER FUNCTIONS ===
def extract_lbp_features(image_gray):
    lbp = local_binary_pattern(image_gray, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_glcm_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def extract_features(image_path, use_preprocess=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = preprocess_image(image) if use_preprocess else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_feat = extract_lbp_features(gray)
    glcm_feat = extract_glcm_features(gray)
    return np.hstack((lbp_feat, glcm_feat))

# === LOAD DATA ===
data = []
labels = []
skipped = 0

print("\n🔍 Loading dataset...")
for root, _, files in os.walk(IMAGE_DIR):
    for fname in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, fname)
            label = 0 if 'autistic' in root.lower() and 'non_autistic' not in root.lower() else 1
            try:
                features = extract_features(full_path)
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"[!] Error processing {fname}: {e}")
                skipped += 1

data = np.array(data)
labels = np.array(labels)
print(f"\n[✔] Loaded {len(data)} valid images. Skipped: {skipped}")
print("Label distribution:", Counter(labels))

if len(data) == 0:
    print("❌ No images loaded! Please check your dataset structure and paths.")
    exit(1)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === CLASSIFIER ===
model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === SAVE TRAINED MODEL AND SCALER ===
model_save_path = r'C:\Users\Yaven_Singh\Desktop\My Eye Tracking Model\My Eye Tracking Model\eye_tracking_model.pkl'
scaler_save_path = r'C:\Users\Yaven_Singh\Desktop\My Eye Tracking Model\My Eye Tracking Model\eye_tracking_scaler.pkl'
joblib.dump(model, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"\n✅ Model saved to: {model_save_path}")
print(f"✅ Scaler saved to: {scaler_save_path}")

# === STRATIFIED K-FOLD CV ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

print("\n🚀 Starting Stratified 5-Fold CV...")
for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels), 1):
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    scaler_cv = StandardScaler()
    X_train = scaler_cv.fit_transform(X_train)
    X_test = scaler_cv.transform(X_test)
    model_cv = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model_cv.fit(X_train, y_train)
    y_pred_cv = model_cv.predict(X_test)
    acc = accuracy_score(y_test, y_pred_cv)
    accuracies.append(acc)
    print(f"\n--- Fold {fold} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cv))
    print("Classification Report:\n", classification_report(y_test, y_pred_cv))
    print(f"Accuracy: {acc:.4f}")

print(f"\n✅ Average Accuracy over 5 folds: {np.mean(accuracies) * 100:.2f}%")
