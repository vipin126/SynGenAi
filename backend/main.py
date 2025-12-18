from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import uvicorn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Import our model
from models.simple_gan import SimpleGAN

app = FastAPI(title="SynGenAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
MODELS_DIR = BASE_DIR / "data" / "models"

for d in [UPLOAD_DIR, SYNTHETIC_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "filename": file.filename, "path": str(file_path), "size": os.path.getsize(file_path)}

def train_background_task(filename: str, epochs: int):
    global training_status
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["message"] = "Preprocessing..."
    
    try:
        # Load
        file_path = UPLOAD_DIR / filename
        df = pd.read_csv(file_path)
        
        # Preprocess & Save Scalers
        le_dict = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
            
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.values)
        
        # Save Scalers for later generation
        joblib.dump({'scaler': scaler, 'le_dict': le_dict, 'columns': df.columns}, MODELS_DIR / f"preprocessor_{filename}.pkl")

        # Prepare Tensor
        features = data_scaled.shape[1]
        tensor_x = torch.FloatTensor(data_scaled)
        dataloader = DataLoader(TensorDataset(tensor_x), batch_size=32, shuffle=True)
        
        # Initialize & Train
        gan = SimpleGAN(latent_dim=10, channels=features)
        
        for epoch in range(epochs):
            for i, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.to(gan.device)
                
                # Train Disc
                gan.d_optimizer.zero_grad()
                d_loss = gan.criterion(gan.discriminator(real_data), torch.ones(batch_size, 1).to(gan.device)) + \
                         gan.criterion(gan.discriminator(gan.generator(torch.randn(batch_size, 10).to(gan.device)).detach()), torch.zeros(batch_size, 1).to(gan.device))
                d_loss.backward()
                gan.d_optimizer.step()
                
                # Train Gen
                gan.g_optimizer.zero_grad()
                g_loss = gan.criterion(gan.discriminator(gan.generator(torch.randn(batch_size, 10).to(gan.device))), torch.ones(batch_size, 1).to(gan.device))
                g_loss.backward()
                gan.g_optimizer.step()
            
            # Progress
            progress = int(((epoch + 1) / epochs) * 100)
            training_status["progress"] = progress
            training_status["message"] = f"Epoch {epoch+1}/{epochs}"

        # Save Model
        gan.save_models(str(MODELS_DIR / f"gan_{filename}.pth"))
        
        training_status["is_training"] = False
        training_status["message"] = "Training Complete"
        training_status["progress"] = 100
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Error: {str(e)}"
        print(f"Error: {e}")

@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, filename: str, epochs: int = 200):
    if training_status["is_training"]: raise HTTPException(400, "Busy")
    background_tasks.add_task(train_background_task, filename, epochs)
    return {"status": "started"}

@app.get("/api/status")
async def get_status():
    return training_status

@app.post("/api/generate")
async def generate_data(filename: str, num_samples: int = 50):
    """Generate synthetic data using trained model"""
    try:
        model_path = MODELS_DIR / f"gan_{filename}.pth"
        proc_path = MODELS_DIR / f"preprocessor_{filename}.pkl"
        
        if not model_path.exists():
            raise HTTPException(404, "Model not found. Train first!")
            
        # Load Preprocessors
        proc_data = joblib.load(proc_path)
        scaler = proc_data['scaler']
        le_dict = proc_data['le_dict']
        columns = proc_data['columns']
        
        # Load Model
        features = scaler.n_features_in_
        gan = SimpleGAN(latent_dim=10, channels=features)
        gan.load_models(str(model_path))
        
        # Generate
        noise = torch.randn(num_samples, 10).to(gan.device)
        fake_data_scaled = gan.generator(noise).detach().cpu().numpy()
        
        # Inverse Transform (Math -> Real Numbers)
        fake_data = scaler.inverse_transform(fake_data_scaled)
        df_fake = pd.DataFrame(fake_data, columns=columns)
        
        # Restore Categories (Numbers -> Text)
        for col, le in le_dict.items():
            # Round to nearest integer for categorical indices
            df_fake[col] = df_fake[col].round().astype(int).clip(0, len(le.classes_)-1)
            df_fake[col] = le.inverse_transform(df_fake[col])
            
        # Save to CSV
        output_path = SYNTHETIC_DIR / f"synthetic_{filename}"
        df_fake.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "count": num_samples,
            "data": df_fake.head(5).to_dict(orient="records"), # Send first 5 rows to frontend
            "download_url": str(output_path)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
