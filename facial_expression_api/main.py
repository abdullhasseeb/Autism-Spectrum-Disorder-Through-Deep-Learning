from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import List, Dict
from collections import Counter


# Initialize FastAPI app
app = FastAPI(
    title="Facial Expression Classification API",
    description="ResNet50-based API for classifying facial expressions",
    version="1.0.0"
)


# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables
model = None
class_names = ['autistic', 'non_autistic']
IMAGE_SIZE = 224


@app.on_event("startup")
async def load_model():
    """Load the trained model when server starts"""
    global model
    try:
        model = tf.keras.models.load_model('./model_5.keras')
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image exactly like training pipeline"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise e


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
    
    # Try to decode with PIL as final validation
    try:
        image = Image.open(io.BytesIO(image_data))
        print(f"✅ PIL validation passed: {image.size}, mode: {image.mode}")
        return True
    except Exception as e:
        print(f"❌ PIL validation failed: {e}")
        return False


@app.get("/")
async def root():
    return {
        "message": "Facial Expression Classification API",
        "status": "running",
        "version": "1.0.0",
        "model": "ResNet50",
        "classes": class_names
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": class_names
    }


@app.post("/predict")
async def predict_expressions(files: List[UploadFile] = File(...)):
    """Enhanced facial expression prediction with detailed debugging"""
    
    if model is None:
        print("❌ Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not files or len(files) == 0:
        print("❌ No files uploaded")
        raise HTTPException(status_code=400, detail="No files uploaded")

    print(f"😊 Processing {len(files)} files for facial expression analysis")

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

            # Process image with PIL and TensorFlow
            print(f"🔄 Opening image with PIL: {file.filename}")
            image = Image.open(io.BytesIO(image_data))
            print(f"✅ PIL image opened: size={image.size}, mode={image.mode}")
            
            print(f"🔄 Preprocessing image")
            processed_image = preprocess_image(image)
            print(f"✅ Image preprocessed: shape={processed_image.shape}")
            
            print(f"🔄 Making prediction with ResNet50")
            prediction = model.predict(processed_image, verbose=0)
            print(f"✅ Prediction completed: {prediction[0]}")

            predicted_class_idx = int(np.argmax(prediction[0]))
            predicted_class = class_names[predicted_class_idx]
            confidence = float(round(float(np.max(prediction[0])) * 100, 2))

            probabilities = {c: float(round(float(p)*100, 2)) for c, p in zip(class_names, prediction[0])}

            result = {
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities,
                "file_size": actual_size,
                "processed_successfully": True
            }

            predictions.append(result)
            prediction_classes.append(predicted_class)
            processed_files += 1

            print(f"✅ Prediction for {file.filename}: {predicted_class} ({confidence:.1f}%)")

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
    print("🚀 Starting Facial Expression Classification API...")
    print("📊 TensorFlow version:", tf.__version__)
    print(f"📊 Classes: {class_names}")
    print("🌐 Server will be accessible at http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
