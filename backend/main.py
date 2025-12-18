from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import uvicorn
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import our custom modules
from models.simple_gan import SimpleGAN

app = FastAPI(title="SynGenAI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
MODELS_DIR = BASE_DIR / "data" / "models"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Store training status in memory (for simplicity)
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
    "epoch": 0,
    "total_epochs": 0
}

@app.get("/")
async def root():
    return {"message": "SynGenAI API is running!", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "upload_dir": str(UPLOAD_DIR),
        "models_available": ["GAN", "VAE", "Diffusion"]
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        allowed = [".csv", ".zip", ".png", ".jpg", ".jpeg"]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed:
            raise HTTPException(400, f"File type {ext} not supported")
        
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

# --- NEW: Training Logic ---

def train_background_task(filename: str, epochs: int):
    global training_status
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["message"] = "Preprocessing data..."
    
    try:
        # 1. Load Data
        file_path = UPLOAD_DIR / filename
        df = pd.read_csv(file_path)
        
        # 2. Preprocessing (Simple version for Tabular)
        # Convert categorical to numeric
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
            
        # Normalize numeric data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.values)
        
        # Convert to Tensor
        # Reshape for GAN: (N, C, H, W) -> For tabular, we treat as 1x1 image with C=features
        features = data_scaled.shape[1]
        tensor_x = torch.Tensor(data_scaled).view(-1, features, 1, 1)
        
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 3. Initialize GAN
        # Note: We reuse the image GAN architecture but adapt channels to feature count
        gan = SimpleGAN(latent_dim=10, img_size=1, channels=features) 
        
        # 4. Training Loop
        training_status["total_epochs"] = epochs
        
        for epoch in range(epochs):
            # Run one epoch of training
            # (We need to modify SimpleGAN slightly to accept callback or we just run it here)
            # For this demo, we'll implement a custom loop here or call gan.train
            
            # Simple manual loop to update progress
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for i, (real_imgs,) in enumerate(dataloader):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(gan.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(gan.device)
                fake_labels = torch.zeros(batch_size, 1).to(gan.device)
                
                # Train Discriminator
                gan.d_optimizer.zero_grad()
                real_output = gan.discriminator(real_imgs)
                d_loss_real = gan.criterion(real_output, real_labels)
                
                z = torch.randn(batch_size, gan.latent_dim, 1, 1).to(gan.device)
                fake_imgs = gan.generator(z)
                fake_output = gan.discriminator(fake_imgs.detach())
                d_loss_fake = gan.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                gan.d_optimizer.step()
                
                # Train Generator
                gan.g_optimizer.zero_grad()
                fake_output = gan.discriminator(fake_imgs)
                g_loss = gan.criterion(fake_output, real_labels)
                g_loss.backward()
                gan.g_optimizer.step()
            
            # Update Status
            progress = int(((epoch + 1) / epochs) * 100)
            training_status["progress"] = progress
            training_status["epoch"] = epoch + 1
            training_status["message"] = f"Training... Epoch {epoch+1}/{epochs}"
            print(f"Epoch {epoch+1}: Progress {progress}%")

        # 5. Save Model
        model_path = MODELS_DIR / f"gan_{filename}.pth"
        gan.save_models(str(model_path))
        
        training_status["is_training"] = False
        training_status["message"] = "Training Complete! Model Saved."
        training_status["progress"] = 100
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Error: {str(e)}"
        print(f"Training failed: {e}")

@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, filename: str, epochs: int = 50):
    """Start training in background"""
    if training_status["is_training"]:
        raise HTTPException(400, "Training already in progress")
    
    background_tasks.add_task(train_background_task, filename, epochs)
    return {"status": "started", "message": "Training started in background"}

@app.get("/api/status")
async def get_status():
    """Get current training status"""
    return training_status

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
