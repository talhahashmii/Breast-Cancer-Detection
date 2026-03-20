from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
from pathlib import Path
import os
from model_manager import ModelManager
from report_generator import generate_pdf_report

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Detection API",
    description="AI-powered mammography analysis system",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = "../ML-Pipeline/Model/finetuned_swin/best_model"

try:
    model_manager = ModelManager(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL_LOADED = False


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str


class DualViewPredictionResponse(BaseModel):
    cc_view: dict
    mlo_view: dict
    final_prediction: str
    final_confidence: float
    risk_level: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Breast Cancer Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": model_manager.device if MODEL_LOADED else "N/A"
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """
    Predict on a single mammogram image
    
    Args:
        file: Image file (PNG, JPG, etc.)
    
    Returns:
        Prediction result with confidence
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        prediction, confidence = model_manager.predict(image)
        risk_level = "HIGH" if prediction == "malignant" else "LOW"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict-dual-view")
async def predict_dual_view(cc_view: UploadFile = File(...), mlo_view: UploadFile = File(...)):
    """
    Predict on both CC and MLO views of mammogram
    
    Args:
        cc_view: Craniocaudal view image
        mlo_view: Mediolateral oblique view image
    
    Returns:
        Ensemble prediction from both views
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        cc_contents = await cc_view.read()
        mlo_contents = await mlo_view.read()
        
        cc_image = Image.open(io.BytesIO(cc_contents)).convert("RGB")
        mlo_image = Image.open(io.BytesIO(mlo_contents)).convert("RGB")
        
        result = model_manager.predict_dual_view(cc_image, mlo_image)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")


@app.post("/generate-report")
async def generate_report(cc_view: UploadFile = File(...), mlo_view: UploadFile = File(...)):
    """
    Generate PDF report from dual-view prediction
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        cc_contents = await cc_view.read()
        mlo_contents = await mlo_view.read()
        
        cc_image = Image.open(io.BytesIO(cc_contents)).convert("RGB")
        mlo_image = Image.open(io.BytesIO(mlo_contents)).convert("RGB")
        
        prediction_data = model_manager.predict_dual_view(cc_image, mlo_image)
        pdf_bytes = generate_pdf_report(prediction_data)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=breast_cancer_detection_report.pdf"}
        )
    except Exception as e:
        print(f"PDF generation error: {e}")
        raise HTTPException(status_code=400, detail=f"Error generating report: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch prediction on multiple images
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            prediction, confidence = model_manager.predict(image)
            risk_level = "HIGH" if prediction == "malignant" else "LOW"
            
            results.append({
                "filename": file.filename,
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "risk_level": risk_level,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }


@app.get("/info")
async def get_info():
    """Get model and system information"""
    return {
        "model_name": "Swin Transformer",
        "model_type": "Binary Classification (Benign vs Malignant)",
        "input_size": "224x224 pixels",
        "classes": ["benign", "malignant"],
        "dataset": "CBIS-DDSM Mammography Dataset",
        "device": model_manager.device if MODEL_LOADED else "N/A",
        "disclaimer": "This tool is for research purposes only. Not FDA approved. Consult a radiologist for diagnosis."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
