import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, SwinForImageClassification
from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["benign", "malignant"]


class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = DEVICE
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned model and processor"""
        print(f"Loading model from {self.model_path}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = SwinForImageClassification.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded on {self.device}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_data) -> Image.Image:
        """Convert various image formats to PIL Image"""
        if isinstance(image_data, bytes):
            return Image.open(Path(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, np.ndarray):
            if image_data.dtype == np.uint8:
                img_array = image_data
            else:
                img_array = (image_data * 255).astype(np.uint8)
            
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            return Image.fromarray(img_array).convert("RGB")
        else:
            raise ValueError("Unsupported image format")
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Run prediction on image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()
                confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
            
            return LABELS[predicted_class], float(confidence)
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def predict_dual_view(self, cc_image: Image.Image, mlo_image: Image.Image) -> dict:
        """Predict on both CC and MLO views"""
        try:
            cc_pred, cc_conf = self.predict(cc_image)
            mlo_pred, mlo_conf = self.predict(mlo_image)
            
            # Ensemble: average confidence if predictions match
            if cc_pred == mlo_pred:
                final_pred = cc_pred
                final_conf = (cc_conf + mlo_conf) / 2
            else:
                # If predictions differ, take the one with higher confidence
                if cc_conf > mlo_conf:
                    final_pred = cc_pred
                    final_conf = cc_conf
                else:
                    final_pred = mlo_pred
                    final_conf = mlo_conf
            
            return {
                "cc_view": {
                    "prediction": cc_pred,
                    "confidence": round(cc_conf * 100, 2)
                },
                "mlo_view": {
                    "prediction": mlo_pred,
                    "confidence": round(mlo_conf * 100, 2)
                },
                "final_prediction": final_pred,
                "final_confidence": round(final_conf * 100, 2),
                "risk_level": "HIGH" if final_pred == "malignant" else "LOW"
            }
        except Exception as e:
            print(f"Error in dual view prediction: {e}")
            raise
