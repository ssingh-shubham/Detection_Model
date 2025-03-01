# takerapi.py
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Suppress macOS warning
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings (from original code)
PATHS = {
    'image_dir': 'criminal_images',
    'names_file': 'criminal_names.json',
    'criminal_records': 'criminal_records.json',
    'cascade_file': 'haarcascade_frontalface_default.xml'
}

CAMERA = {
    'index': 0,
    'width': 640,
    'height': 480
}

FACE_DETECTION = {
    'scale_factor': 1.3,
    'min_neighbors': 5,
    'min_size': (30, 30)
}

TRAINING = {
    'samples_needed': 20
}

# Pydantic models for data validation
class Crime(BaseModel):
    description: str

class CriminalBase(BaseModel):
    name: str = Field(..., description="Full name of the criminal")
    sex: str = Field(..., description="Sex (M/F/Other)")
    age: str = Field(..., description="Age of the criminal")
    address: str = Field(..., description="Address of the criminal")
    crimes: List[str] = Field([], description="List of crimes committed")
    notes: Optional[str] = Field(None, description="Additional notes")

class CriminalCreate(CriminalBase):
    pass

class CriminalResponse(CriminalBase):
    id: int
    registration_date: str

class CriminalUpdate(BaseModel):
    name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[str] = None
    address: Optional[str] = None
    crimes: Optional[List[str]] = None
    notes: Optional[str] = None

# Helper functions from original code
def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def get_face_id(directory: str) -> int:
    """
    Get the first available face ID by checking existing files.
    
    Parameters:
        directory (str): The path of the directory of images.
    Returns:
        int: The next available face ID
    """
    try:
        if not os.path.exists(directory):
            return 1
            
        user_ids = []
        for filename in os.listdir(directory):
            if filename.startswith('Criminal-'):
                try:
                    number = int(filename.split('-')[1])
                    user_ids.append(number)
                except (IndexError, ValueError):
                    continue
                    
        return max(user_ids + [0]) + 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        raise

def save_criminal_info(face_id: int, criminal_data: Dict[str, Any], filename: str) -> None:
    """
    Save criminal information to JSON file
    
    Parameters:
        face_id (int): The identifier of criminal
        criminal_data (Dict): The criminal information
        filename (str): Path to the JSON file
    """
    try:
        criminal_records: Dict[str, Dict[str, Any]] = {}
        
        # Load existing records if file exists and is not empty
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as fs:
                    content = fs.read().strip()
                    if content:  # Only try to load if file is not empty
                        criminal_records = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {filename}, starting fresh")
                criminal_records = {}
        
        # Add new record with all details
        criminal_records[str(face_id)] = criminal_data
        
        # Save updated records
        with open(filename, 'w') as fs:
            json.dump(criminal_records, fs, indent=4, ensure_ascii=False)
        logger.info(f"Saved criminal information for ID {face_id}")
    except Exception as e:
        logger.error(f"Error saving criminal information: {e}")
        raise

def get_current_date() -> str:
    """Get current date in ISO format"""
    return datetime.now().strftime("%Y-%m-%d")

def load_criminal_records() -> Dict[str, Any]:
    """Load all criminal records from file"""
    filename = PATHS['criminal_records']
    if not os.path.exists(filename):
        return {}
        
    try:
        with open(filename, 'r') as fs:
            content = fs.read().strip()
            if content:
                return json.loads(content)
            return {}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in {filename}")
        return {}

def save_face_image(face_id: int, image_data: bytes, count: int) -> str:
    """Save a face image to the file system"""
    create_directory(PATHS['image_dir'])
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face detection cascade
    face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
    if face_cascade.empty():
        raise ValueError("Error loading cascade classifier")
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION['scale_factor'],
        minNeighbors=FACE_DETECTION['min_neighbors'],
        minSize=FACE_DETECTION['min_size']
    )
    
    if not len(faces):
        raise ValueError("No face detected in the image")
    
    # Use the first detected face
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]
    
    # Save the face image
    img_path = f'./{PATHS["image_dir"]}/Criminal-{face_id}-{count}.jpg'
    cv2.imwrite(img_path, face_img)
    
    return img_path

def update_names_file(face_id: int, name: str) -> None:
    """Update the names file with new criminal name"""
    names_file = PATHS['names_file']
    save_name_data = {str(face_id): name}
    
    # Update existing names file
    if os.path.exists(names_file):
        try:
            with open(names_file, 'r') as fs:
                content = fs.read().strip()
                if content:
                    existing_names = json.loads(content)
                    existing_names.update(save_name_data)
                    save_name_data = existing_names
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {names_file}, starting fresh")
    
    # Save names file
    with open(names_file, 'w') as fs:
        json.dump(save_name_data, fs, indent=4, ensure_ascii=False)

# Modern FastAPI Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (previously in on_event("startup"))
    create_directory(PATHS['image_dir'])
    # Make sure criminal records and names files exist
    if not os.path.exists(PATHS['criminal_records']):
        with open(PATHS['criminal_records'], 'w') as f:
            f.write("{}")
    
    if not os.path.exists(PATHS['names_file']):
        with open(PATHS['names_file'], 'w') as f:
            f.write("{}")
    yield
    # Shutdown code would go here (if needed)

# Initialize FastAPI
app = FastAPI(
    title="Criminal Registration API",
    description="API for registering and managing criminal records with facial recognition",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Criminal Registration API", "version": "1.0.0"}

@app.post("/criminals/", response_model=CriminalResponse)
def create_criminal(criminal: CriminalCreate):
    """Register a new criminal without facial data"""
    try:
        # Get next available ID
        face_id = get_face_id(PATHS['image_dir'])
        
        # Prepare criminal data
        criminal_data = criminal.dict()
        criminal_data["registration_date"] = get_current_date()
        
        # Save criminal info
        save_criminal_info(face_id, criminal_data, PATHS['criminal_records'])
        
        # Update names file
        update_names_file(face_id, criminal_data["name"])
        
        # Return response with ID
        return {**criminal_data, "id": face_id}
    except Exception as e:
        logger.error(f"Error creating criminal record: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/criminals/", response_model=List[CriminalResponse])
def list_criminals():
    """List all criminals in the database"""
    try:
        criminal_records = load_criminal_records()
        result = []
        
        for id_str, record in criminal_records.items():
            result.append({
                **record,
                "id": int(id_str)
            })
            
        return result
    except Exception as e:
        logger.error(f"Error listing criminals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/criminals/{criminal_id}", response_model=CriminalResponse)
def get_criminal(criminal_id: int):
    """Get a specific criminal by ID"""
    try:
        criminal_records = load_criminal_records()
        id_str = str(criminal_id)
        
        if id_str not in criminal_records:
            raise HTTPException(status_code=404, detail="Criminal not found")
            
        return {**criminal_records[id_str], "id": criminal_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting criminal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/criminals/{criminal_id}", response_model=CriminalResponse)
def update_criminal(criminal_id: int, criminal_update: CriminalUpdate):
    """Update criminal information"""
    try:
        criminal_records = load_criminal_records()
        id_str = str(criminal_id)
        
        if id_str not in criminal_records:
            raise HTTPException(status_code=404, detail="Criminal not found")
            
        # Update only provided fields
        update_data = {k: v for k, v in criminal_update.dict().items() if v is not None}
        criminal_records[id_str].update(update_data)
        
        # Save updates
        save_criminal_info(criminal_id, criminal_records[id_str], PATHS['criminal_records'])
        
        # Update name in names file if it was changed
        if "name" in update_data:
            update_names_file(criminal_id, update_data["name"])
            
        return {**criminal_records[id_str], "id": criminal_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating criminal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/criminals/{criminal_id}")
def delete_criminal(criminal_id: int):
    """Delete a criminal record"""
    try:
        criminal_records = load_criminal_records()
        id_str = str(criminal_id)
        
        if id_str not in criminal_records:
            raise HTTPException(status_code=404, detail="Criminal not found")
            
        # Remove from criminal records
        del criminal_records[id_str]
        with open(PATHS['criminal_records'], 'w') as fs:
            json.dump(criminal_records, fs, indent=4, ensure_ascii=False)
            
        # Remove from names file
        names_file = PATHS['names_file']
        if os.path.exists(names_file):
            try:
                with open(names_file, 'r') as fs:
                    names_data = json.loads(fs.read().strip())
                    if id_str in names_data:
                        del names_data[id_str]
                with open(names_file, 'w') as fs:
                    json.dump(names_data, fs, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Error updating names file: {e}")
                
        # Delete face images
        img_dir = PATHS['image_dir']
        for filename in os.listdir(img_dir):
            if filename.startswith(f'Criminal-{criminal_id}-'):
                try:
                    os.remove(os.path.join(img_dir, filename))
                except Exception as e:
                    logger.warning(f"Error deleting image {filename}: {e}")
                    
        return {"message": f"Criminal with ID {criminal_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting criminal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/criminals/{criminal_id}/face")
async def upload_face_image(
    criminal_id: int, 
    image_number: int = Form(...),
    image: UploadFile = File(...)
):
    """Upload a face image for a criminal"""
    try:
        criminal_records = load_criminal_records()
        id_str = str(criminal_id)
        
        if id_str not in criminal_records:
            raise HTTPException(status_code=404, detail="Criminal not found")
            
        # Check if the image number is valid
        if image_number < 1 or image_number > TRAINING['samples_needed']:
            raise HTTPException(
                status_code=400, 
                detail=f"Image number must be between 1 and {TRAINING['samples_needed']}"
            )
            
        # Read image content
        image_content = await image.read()
        
        # Save face image
        try:
            img_path = save_face_image(criminal_id, image_content, image_number)
            return {"message": f"Face image saved successfully", "path": img_path}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading face image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/criminals/{criminal_id}/face-count")
def get_face_count(criminal_id: int):
    """Get the number of face images for a criminal"""
    try:
        # Check if criminal exists
        criminal_records = load_criminal_records()
        id_str = str(criminal_id)
        
        if id_str not in criminal_records:
            raise HTTPException(status_code=404, detail="Criminal not found")
        
        # Count images
        count = 0
        img_dir = PATHS['image_dir']
        if os.path.exists(img_dir):
            for filename in os.listdir(img_dir):
                if filename.startswith(f'Criminal-{criminal_id}-'):
                    count += 1
                    
        return {"criminal_id": criminal_id, "face_count": count, "required": TRAINING['samples_needed']}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/criminals/batch-upload")
async def batch_upload_faces(background_tasks: BackgroundTasks, criminal: CriminalCreate, images: List[UploadFile] = File(...)):
    """Register a new criminal with multiple face images in one request"""
    try:
        # Get next available ID
        face_id = get_face_id(PATHS['image_dir'])
        
        # Prepare criminal data
        criminal_data = criminal.dict()
        criminal_data["registration_date"] = get_current_date()
        
        # Save criminal info
        save_criminal_info(face_id, criminal_data, PATHS['criminal_records'])
        
        # Update names file
        update_names_file(face_id, criminal_data["name"])
        
        # Process face images in the background
        async def process_images():
            for i, image in enumerate(images):
                if i >= TRAINING['samples_needed']:
                    break
                image_content = await image.read()
                try:
                    save_face_image(face_id, image_content, i + 1)
                except Exception as e:
                    logger.error(f"Error saving image {i+1} for criminal {face_id}: {e}")
                    
        background_tasks.add_task(process_images)
        
        return {
            **criminal_data, 
            "id": face_id, 
            "message": f"Criminal registered successfully. Processing {min(len(images), TRAINING['samples_needed'])} images in the background."
        }
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("takerapi:app", host="0.0.0.0", port=8000, reload=True)