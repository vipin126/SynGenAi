import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import json

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data_type = self._detect_data_type()
        self.stats = {}
    
    def _detect_data_type(self) -> str:
        """Detect if data is CSV, images, or time-series"""
        if self.data_path.suffix == '.csv':
            return 'tabular'
        elif self.data_path.suffix in ['.png', '.jpg', '.jpeg']:
            return 'image'
        elif self.data_path.is_dir():
            # Check if directory contains images
            image_files = list(self.data_path.glob('*.png')) + list(self.data_path.glob('*.jpg'))
            if image_files:
                return 'image'
        return 'unknown'
    
    def load_data(self):
        """Load data based on type"""
        if self.data_type == 'tabular':
            return self._load_tabular()
        elif self.data_type == 'image':
            return self._load_images()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _load_tabular(self) -> pd.DataFrame:
        """Load and validate CSV data"""
        df = pd.read_csv(self.data_path)
        
        # Calculate statistics
        self.stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        return df
    
    def _load_images(self) -> List[np.ndarray]:
        """Load images from directory or single file"""
        images = []
        
        if self.data_path.is_dir():
            image_paths = list(self.data_path.glob('*.png')) + list(self.data_path.glob('*.jpg'))
        else:
            image_paths = [self.data_path]
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        # Calculate statistics
        if images:
            shapes = [img.shape for img in images]
            self.stats = {
                'count': len(images),
                'shape': shapes[0] if len(set(shapes)) == 1 else 'varying',
                'dtype': str(images[0].dtype),
                'min_value': int(np.min(images[0])),
                'max_value': int(np.max(images[0]))
            }
        
        return images
    
    def preprocess_tabular(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize tabular data"""
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        return df
    
    def preprocess_images(self, images: List[np.ndarray], target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Resize and normalize images"""
        processed = []
        
        for img in images:
            # Resize
            img_resized = cv2.resize(img, target_size)
            # Normalize to [-1, 1]
            img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
            processed.append(img_normalized)
        
        return np.array(processed)
    
    def get_stats(self) -> Dict:
        """Return dataset statistics"""
        return self.stats
    
    def save_stats(self, output_path: str):
        """Save statistics to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
