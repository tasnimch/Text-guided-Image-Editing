import os

class Config:
    UNET_PATH = os.getenv("unet_path", "/home/tasnim/FastAPI_app/checkpoints")
    MODEL_PATH = "timbrooks/instruct-pix2pix"
    OUTPUT_PATH = "temp/output.png"
    FRONT_PATH = "http://localhost:4200"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    
class ModelArgs:
    num_inference_steps = 20
    image_guidance_scale = 1.5
    guidance_scale = 7