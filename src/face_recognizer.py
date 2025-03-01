import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import json
import logging
import os

import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define settings directly in the test file
CAMERA = {
    'index': 0,
    'width': 640,
    'height': 480
}

FACE_DETECTION = {
    'scale_factor': 1.2,  # Try a smaller scale factor for better detection
    'min_neighbors': 5,
    'min_size': (30, 30)
}

PATHS = {
    'cascade_file': cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    'trainer_file': 'trainer.yml',  # Update this path to your actual trainer file
    'names_file': 'names.json',
    'criminal_records': 'criminal_records.json'
}

def initialize_camera(camera_index=0):
    """Initialize the camera with error handling"""
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

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

def display_criminal_info(img, criminal_record, position, width):
    """Display criminal information on the image"""
    x, y = position
    y_offset = y + 30
    
    # Add debug log to verify the function is being called
    logger.info(f"Displaying criminal info for: {criminal_record.get('name', 'Unknown')}")
    
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

def test_recognition_with_overlay():
    """Test face recognition with overlay information"""
    try:
        logger.info("Starting face recognition with overlay test...")
        
        # Check if trainer file exists
        if not os.path.exists(PATHS['trainer_file']):
            # For testing without a trained model, we'll create a dummy recognizer
            logger.warning(f"Trainer file not found: {PATHS['trainer_file']}")
            logger.info("Using dummy recognition for testing overlay...")
            
            # Create a test function that simulates recognition
            def dummy_predict(face_img):
                # Always return ID 1 with high confidence for testing
                return 1, 10  # ID 1, low distance (high confidence)
                
            # Create a dummy recognizer object with a predict method
            class DummyRecognizer:
                def predict(self, face_img):
                    return dummy_predict(face_img)
                    
            recognizer = DummyRecognizer()
            logger.info("Using dummy recognizer for testing")
        else:
            # Initialize face recognizer with the real model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(PATHS['trainer_file'])
            logger.info(f"Loaded face recognizer from {PATHS['trainer_file']}")
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError(f"Error loading cascade classifier from {PATHS['cascade_file']}")
        logger.info(f"Loaded face cascade from {PATHS['cascade_file']}")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        logger.info("Camera initialized successfully")
        
        # Load names and criminal records
        names = load_names(PATHS['names_file'])
        criminal_records = load_criminal_records(PATHS['criminal_records'])
        
        # For testing, if no criminal records, create a sample one
        if not criminal_records and '1' not in criminal_records:
            criminal_records = {
                '1': {
                    'name': 'Test Person',
                    'sex': 'Not Specified',
                    'age': 30,
                    'address': '123 Test Street',
                    'crimes': ['For Testing Only', 'Not a real record']
                }
            }
            logger.info("Created sample criminal record for testing")
        
        logger.info("Face recognition with overlay test active. Press 'ESC' to exit.")
        
        # Set confidence threshold lower for testing
        confidence_threshold = 50  # Lower threshold to make matching easier
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
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
            
            logger.info(f"Detected {len(faces)} faces")
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Recognize the face
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Convert confidence (lower is better in OpenCV face recognition)
                # For LBPH: 0 is perfect match, higher values are worse
                confidence_score = 100 - confidence if confidence < 100 else 0
                
                logger.info(f"Face recognition result: ID={id}, Confidence={confidence_score:.1f}%")
                
                # Check confidence and display result
                if confidence_score >= confidence_threshold:
                    criminal_id = str(id)
                    
                    # Get criminal record if available
                    if criminal_id in criminal_records:
                        # Show detailed criminal information
                        display_criminal_info(display_img, criminal_records[criminal_id], (x, y), w)
                        
                        # Alert text for high confidence matches
                        if confidence_score > 70:
                            cv2.putText(display_img, "CRIMINAL IDENTIFIED", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Fall back to name-only display if detailed record not found
                        name = names.get(criminal_id, "Unknown")
                        cv2.putText(display_img, name, (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display confidence percentage
                    confidence_text = f"Match: {confidence_score:.1f}%"
                    cv2.putText(display_img, confidence_text, (x+5, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    # Unknown person
                    cv2.putText(display_img, "Unknown", (x+5, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Even for unknown, show the confidence score
                    confidence_text = f"Match: {confidence_score:.1f}%"
                    cv2.putText(display_img, confidence_text, (x+5, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Always display instructions
            cv2.putText(display_img, "Press ESC to exit", (10, display_img.shape[0]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition With Overlay Test', display_img)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        logger.info("Face recognition with overlay test completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        if 'cam' in locals() and cam is not None:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_recognition_with_overlay()