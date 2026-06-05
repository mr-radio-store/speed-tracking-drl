import cv2
import numpy as np
import torch  # <-- Add this import
from ultralytics import YOLO

class VisionEngine:
    """Manages the neural network architecture, frame inference execution, 
    and fallback environment classification.
    """
    def __init__(self, model_variant: str = "yolov8n.pt"):
        print("🔧 Initializing Vision Pipeline Engine...")
        
        # Determine the fastest available processing backend
        if torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 NVIDIA GPU Detected! Utilizing CUDA hardware acceleration.")
        else:
            self.device = "cpu"
            print("⚠️ GPU Unreachable. Running on CPU (Expect performance drops).")
            
        self.model = YOLO(model_variant)
        
        # Explicitly push the model matrices onto the chosen device hardware
        self.model.to(self.device)
        print(f"✅ Neural Weights Matrix Loaded: {model_variant}")

    def process_frame(self, frame: np.ndarray) -> list:
        """Routes raw multidimensional pixel arrays into the inference stream."""
        # FIXED: Pass the target device into the model call
        results = self.model(frame, verbose=False, device=self.device)
        return results

    def check_environmental_occlusion(self, frame: np.ndarray) -> str:
        """Evaluates image luminosity maps to classify target-less environments."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luminosity = np.mean(gray)
        
        if mean_luminosity < 15.0:
            return "BLOCKED_DARK"
        elif mean_luminosity > 240.0:
            return "BLOCKED_BRIGHT"
        
        return "CLEAR_ENVIRONMENT"