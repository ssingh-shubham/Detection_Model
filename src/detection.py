import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import (Body, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define settings
CAMERA = {
    'width': 640,
    'height': 480
}

FACE_DETECTION = {
    'scale_factor': 1.2,
    'min_neighbors': 5,
    'min_size': (30, 30)
}

PATHS = {
    'cascade_file': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    'trainer_file': 'trainer.yml',
    'names_file': 'criminal_names.json',
    'criminal_records': 'criminal_records.json'
}

# Pydantic models for request and response
class FrameData(BaseModel):
    image: str  # Base64 encoded image

class DetectionResult(BaseModel):
    processed_image: str  # Base64 encoded processed image
    detections: List[Dict[str, Any]]  # Detection information

class CriminalRecord(BaseModel):
    id: str
    name: str
    sex: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    crimes: List[str] = []

# Initialize FastAPI app
app = FastAPI(title="Criminal Face Recognition API",
             description="API for detecting and identifying faces in images with criminal records overlay")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for face recognition
face_cascade = None
recognizer = None
names = {}
criminal_records = {}
confidence_threshold = 40

@app.on_event("startup")
async def startup_event():
    """Initialize components when the application starts"""
    success = initialize_face_recognition()
    if not success:
        logger.error("Failed to initialize face recognition components")

def initialize_face_recognition():
    """Initialize face recognition components"""
    global face_cascade, recognizer, names, criminal_records
    
    try:
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError(f"Error loading cascade classifier from {PATHS['cascade_file']}")
        logger.info(f"Loaded face cascade from {PATHS['cascade_file']}")
        
        # Initialize face recognizer
        if os.path.exists(PATHS['trainer_file']):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(PATHS['trainer_file'])
            logger.info(f"Loaded face recognizer from {PATHS['trainer_file']}")
        else:
            logger.warning(f"Trainer file not found: {PATHS['trainer_file']}")
            logger.info("Using dummy recognition for testing...")
            
            # Create a dummy recognizer for testing
            class DummyRecognizer:
                def predict(self, face_img):
                    return 1, 10  # ID 1, low distance (high confidence)
                    
            recognizer = DummyRecognizer()
        
        # Load names and criminal records
        names = load_names(PATHS['names_file'])
        criminal_records = load_criminal_records(PATHS['criminal_records'])
        
        # For testing, if no criminal records, create a sample one
        if not criminal_records:
            criminal_records = {
                '1': {
                    'name': 'Test Person',
                    'sex': 'Male',
                    'age': 30,
                    'address': '123 Test Street',
                    'crimes': ['For Testing Only', 'Not a real record']
                }
            }
            logger.info("Created sample criminal record for testing")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing face recognition: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_names(filename):
    """Load name mappings from JSON file"""
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
            logger.info(f"Loaded {len(names_json)} names from {filename}")
        else:
            logger.warning(f"Names file not found: {filename}")
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

def load_criminal_records(filename):
    """Load criminal records from JSON file"""
    try:
        records = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    records = json.loads(content)
            logger.info(f"Loaded {len(records)} criminal records from {filename}")
        else:
            logger.warning(f"Criminal records file not found: {filename}")
        return records
    except Exception as e:
        logger.error(f"Error loading criminal records: {e}")
        return {}

def process_image(img):
    """Process an image for face detection and recognition"""
    # Store detection results
    detections = []
    
    # Create a copy for display
    display_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION['scale_factor'],
        minNeighbors=FACE_DETECTION['min_neighbors'],
        minSize=FACE_DETECTION['min_size']
    )
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare detection record
        detection = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "id": None,
            "name": "Unknown",
            "confidence": 0,
            "criminal_record": None
        }
        
        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Convert confidence (lower is better in OpenCV face recognition)
        confidence_score = 100 - confidence if confidence < 100 else 0
        detection["confidence"] = round(float(confidence_score), 1)
        
        # Check confidence and display result
        if confidence_score >= confidence_threshold:
            criminal_id = str(id)
            detection["id"] = criminal_id
            
            # Get criminal record if available
            if criminal_id in criminal_records:
                record = criminal_records[criminal_id]
                detection["name"] = record.get('name', 'Unknown')
                detection["criminal_record"] = record
                
                # Show detailed criminal information
                display_criminal_info(display_img, record, (x, y), w)
                
                # Alert text for high confidence matches
                if confidence_score > 85:
                    cv2.putText(display_img, "CRIMINAL IDENTIFIED", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Fall back to name-only display if detailed record not found
                detection["name"] = names.get(criminal_id, "Unknown")
                cv2.putText(display_img, detection["name"], (x+5, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display confidence percentage
            confidence_text = f"Match: {confidence_score:.1f}%"
            cv2.putText(display_img, confidence_text, (x+5, y+h+20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Unknown person
            cv2.putText(display_img, "Unknown", (x+5, y-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        detections.append(detection)
    
    return display_img, detections

def display_criminal_info(img, criminal_record, position, width):
    """Display criminal information on the image"""
    x, y = position
    y_offset = y + 30
    
    # Set up background rectangle for text
    cv2.rectangle(img, (x-10, y-30), (x+width+200, y+200), (0, 0, 0), -1)
    cv2.rectangle(img, (x-10, y-30), (x+width+200, y+200), (0, 255, 0), 2)
    
    # Display name with larger font
    cv2.putText(img, f"Name: {criminal_record.get('name', 'Unknown')}", 
                (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    # Display basic information
    info_items = [
        f"Sex: {criminal_record.get('sex', 'N/A')}",
        f"Age: {criminal_record.get('age', 'N/A')}",
        f"Address: {criminal_record.get('address', 'N/A')}"
    ]
    
    for item in info_items:
        cv2.putText(img, item, (x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Display crimes (first 2 only to avoid cluttering the display)
    crimes = criminal_record.get('crimes', [])
    cv2.putText(img, "Crimes:", (x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += 20
    
    for i, crime in enumerate(crimes[:2]):  # Limit to first 2 crimes
        cv2.putText(img, f"- {crime}", (x+10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    if len(crimes) > 2:
        cv2.putText(img, f"...and {len(crimes)-2} more", (x+10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Criminal Face Recognition API is running"}

@app.post("/detect", response_model=DetectionResult)
async def detect_faces(frame_data: FrameData):
    """Detect faces in an uploaded image"""
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image
        processed_img, detections = process_image(img)
        
        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return DetectionResult(
            processed_image=processed_b64,
            detections=detections
        )
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DetectionResult)
async def upload_image(file: UploadFile = File(...)):
    """Process an uploaded image file"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        processed_img, detections = process_image(img)
        
        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return DetectionResult(
            processed_image=processed_b64,
            detections=detections
        )
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/criminals", response_model=Dict[str, CriminalRecord])
async def get_criminal_records():
    """Get all criminal records"""
    return criminal_records

@app.get("/criminal/{criminal_id}", response_model=CriminalRecord)
async def get_criminal_record(criminal_id: str):
    """Get a specific criminal record by ID"""
    if criminal_id not in criminal_records:
        raise HTTPException(status_code=404, detail="Criminal record not found")
    return criminal_records[criminal_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)