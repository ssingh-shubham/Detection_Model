# Suppress macOS warning
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import json
import logging
import os
from typing import Any, Dict, Optional

import cv2
from settings.settings import CAMERA, FACE_DETECTION, PATHS, TRAINING

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize the camera with error handling
    
    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture or None: Initialized camera object
    """
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

def collect_criminal_info() -> Dict[str, Any]:
    """
    Collect criminal information from user input
    
    Returns:
        Dict: Dictionary containing criminal information
    """
    print("\n=== CRIMINAL REGISTRATION SYSTEM ===")
    
    # Collect basic information
    full_name = input("Enter full name: ").strip()
    if not full_name:
        raise ValueError("Name cannot be empty")
    
    # Collect additional information
    sex = input("Enter sex (M/F/Other): ").strip()
    age = input("Enter age: ").strip()
    address = input("Enter address: ").strip()
    
    # Collect crimes committed (multiple entries)
    crimes = []
    print("\nEnter crimes committed (leave blank and press Enter when done):")
    crime_num = 1
    while True:
        crime = input(f"  Crime #{crime_num}: ").strip()
        if not crime:
            break
        crimes.append(crime)
        crime_num += 1
    
    # Additional notes
    notes = input("\nAdditional notes (if any): ").strip()
    
    # Create criminal record
    return {
        "name": full_name,
        "sex": sex,
        "age": age,
        "address": address,
        "crimes": crimes,
        "notes": notes,
        "registration_date": get_current_date()
    }

def get_current_date() -> str:
    """Get current date in ISO format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

if __name__ == '__main__':
    try:
        # Initialize directories and files
        create_directory(PATHS['image_dir'])
        
        # Make sure criminal records file exists
        criminal_records_file = PATHS.get('criminal_records', 'criminal_records.json')
        names_file = PATHS['names_file']
        
        # Load face detection
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
            
        # Set up camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
            
        # Collect criminal information
        criminal_data = collect_criminal_info()
        face_id = get_face_id(PATHS['image_dir'])
        
        # Store full criminal record
        save_criminal_info(face_id, criminal_data, criminal_records_file)
        
        # Store just name for recognition (backward compatibility)
        save_name_data = {str(face_id): criminal_data["name"]}
        
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
            
        logger.info(f"Initializing face capture for criminal ID: {face_id}")
        logger.info("Look at the camera and wait...")
        
        count = 0
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                if count < TRAINING['samples_needed']:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    face_img = gray[y:y+h, x:x+w]
                    img_path = f'./{PATHS["image_dir"]}/Criminal-{face_id}-{count+1}.jpg'
                    cv2.imwrite(img_path, face_img)
                    
                    count += 1
                    
                    progress = f"Capturing: {count}/{TRAINING['samples_needed']}"
                    cv2.putText(img, progress, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    break
            
            cv2.imshow('Criminal Face Capture', img)
            
            if cv2.waitKey(100) & 0xff == 27:  # ESC key
                break
            if count >= TRAINING['samples_needed']:
                break
                
        logger.info(f"Successfully captured {count} images for criminal ID: {face_id}")
        logger.info(f"Criminal record saved: {criminal_data['name']} (ID: {face_id})")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()