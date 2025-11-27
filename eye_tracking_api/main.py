from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from typing import List
from collections import Counter
import uvicorn


app = FastAPI(
    title="Eye Tracking Diagnosis API",
    description="MLP-based API for eye tracking ASD diagnosis",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMAGE_SIZE = 224
model = joblib.load("eye_tracking_model.pkl")
scaler = joblib.load("eye_tracking_scaler.pkl")
class_names = ["Autistic", "Non-Autistic"]


def extract_lbp_features(image_gray):
    lbp = local_binary_pattern(image_gray, P=24, R=3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_glcm_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])


def extract_features_from_bytes(image_bytes):
    """Extract features from image bytes with enhanced error handling"""
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Cannot decode image - file may be corrupted or not a valid image")
            
        print(f"✅ Image decoded successfully: {image.shape}")
        
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp_feat = extract_lbp_features(gray)
        glcm_feat = extract_glcm_features(gray)
        
        features = np.hstack((lbp_feat, glcm_feat))
        print(f"✅ Features extracted: {features.shape}")
        
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        raise ValueError(f"Feature extraction failed: {str(e)}")


def validate_image_file(file: UploadFile, image_data: bytes) -> bool:
    """Validate if the uploaded file is a valid image"""
    
    # Check file size
    if len(image_data) == 0:
        print(f"❌ Empty file: {file.filename}")
        return False
        
    if len(image_data) < 100:  # Too small to be a real image
        print(f"❌ File too small ({len(image_data)} bytes): {file.filename}")
        return False
    
    # Check file signature (magic bytes)
    if len(image_data) >= 10:
        first_bytes = image_data[:10]
        print(f"🔍 First 10 bytes: {list(first_bytes)}")
        
        # JPEG signature: FF D8 FF
        if image_data[:3] == b'\xff\xd8\xff':
            print("✅ JPEG signature detected")
            return True
            
        # PNG signature: 89 50 4E 47
        if image_data[:4] == b'\x89PNG':
            print("✅ PNG signature detected")
            return True
            
        # Check if it looks like HTML (common error response)
        try:
            text_start = image_data[:50].decode('utf-8', errors='ignore').lower()
            if any(html_tag in text_start for html_tag in ['<html', '<!doctype', '<body', '404', 'error']):
                print(f"❌ File appears to be HTML/text, not image: {text_start[:30]}")
                return False
        except:
            pass
    
    # Try to decode with OpenCV as final validation
    try:
        file_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            print(f"✅ OpenCV validation passed: {image.shape}")
            return True
        else:
            print("❌ OpenCV could not decode image")
            return False
    except Exception as e:
        print(f"❌ OpenCV validation failed: {e}")
        return False


@app.get("/")
async def root():
    return {
        "message": "Eye Tracking Diagnosis API", 
        "status": "running",
        "version": "1.0.0",
        "model": "MLP with LBP + GLCM features",
        "classes": class_names
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "classes": class_names
    }


@app.post("/eye_tracking")
async def predict_eyes(files: List[UploadFile] = File(...)):
    """Enhanced eye tracking prediction with detailed debugging"""
    
    if not files or len(files) == 0:
        print("❌ No files uploaded")
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    print(f"👁️ Processing {len(files)} files for eye tracking analysis")
    
    predictions = []
    prediction_classes = []
    processed_files = 0
    
    for i, file in enumerate(files):
        print(f"\n🔍 Processing file {i+1}/{len(files)}: {file.filename}")
        print(f"📊 Content-Type: {file.content_type}")
        print(f"📊 File size: {getattr(file, 'size', 'unknown')}")
        
        try:
            # Read file data
            image_data = await file.read()
            actual_size = len(image_data)
            print(f"📊 Actual bytes read: {actual_size}")
            
            # Validate image file
            if not validate_image_file(file, image_data):
                error_msg = f"File {file.filename} is not a valid image"
                print(f"❌ {error_msg}")
                predictions.append({
                    "filename": file.filename,
                    "error": error_msg,
                    "file_size": actual_size,
                    "content_type": file.content_type
                })
                continue
            
            # Extract features and make prediction
            print(f"🔄 Extracting features from {file.filename}")
            features = extract_features_from_bytes(image_data).reshape(1, -1)
            
            print(f"🔄 Scaling features")
            features_scaled = scaler.transform(features)
            
            print(f"🔄 Making prediction")
            pred = model.predict(features_scaled)[0]
            class_name = class_names[pred]
            confidence = float(model.predict_proba(features_scaled)[0].max() * 100)
            
            result = {
                "filename": file.filename,
                "prediction": int(pred),
                "class": class_name,
                "confidence": round(confidence, 2),
                "file_size": actual_size,
                "processed_successfully": True
            }
            
            predictions.append(result)
            prediction_classes.append(class_name)
            processed_files += 1
            
            print(f"✅ Prediction for {file.filename}: {class_name} ({confidence:.1f}%)")
            
        except HTTPException as he:
            # Re-raise HTTP exceptions
            raise he
        except Exception as e:
            error_msg = f"Error processing {file.filename}: {str(e)}"
            print(f"❌ {error_msg}")
            predictions.append({
                "filename": file.filename,
                "error": error_msg,
                "file_size": len(image_data) if 'image_data' in locals() else 0,
                "content_type": file.content_type
            })
    
    print(f"\n📊 Processing complete: {processed_files}/{len(files)} files processed successfully")
    
    # Check if we have any valid predictions
    if not prediction_classes:
        error_details = {
            "message": "No valid images could be processed",
            "total_files": len(files),
            "processed_files": processed_files,
            "individual_results": predictions
        }
        print(f"❌ No valid predictions: {error_details}")
        raise HTTPException(status_code=400, detail=error_details)
    
    # Aggregate final prediction (majority vote)
    final_counts = Counter(prediction_classes)
    final_prediction, count = final_counts.most_common(1)[0]
    total_files = len(files)
    final_confidence = round((count / total_files) * 100, 2)
    
    result = {
        "individual_predictions": predictions,
        "final_prediction": {
            "class": final_prediction,
            "confidence": final_confidence,
            "total_files": total_files,
            "successful_files": processed_files
        },
        "processing_summary": {
            "total_uploaded": total_files,
            "successfully_processed": processed_files,
            "failed": total_files - processed_files,
            "success_rate": round((processed_files / total_files) * 100, 2)
        }
    }
    
    print(f"✅ Final result: {final_prediction} ({final_confidence}%) from {processed_files}/{total_files} files")
    return result


if __name__ == "__main__":
    print("🚀 Starting Eye Tracking Diagnosis API...")
    print("📊 Model loaded successfully")
    print("📊 Scaler loaded successfully") 
    print(f"📊 Classes: {class_names}")
    print("🌐 Server will be accessible at http://0.0.0.0:9000")
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
