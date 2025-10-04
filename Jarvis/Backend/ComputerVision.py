import cv2
import os
import pickle
import face_recognition
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import logging
import time
import re
import random
import json
from groq import Groq
from dotenv import dotenv_values

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# COCO class names for object detection
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class VisionSystem:
    def __init__(self, env_vars):
        self.env_vars = env_vars
        self.known_faces_dir = "Data/KnownFaces"
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Initialize PyTorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Load pre-trained Faster R-CNN model
        try:
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            logging.info("Vision model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load vision model: {str(e)}")
            self.model = None
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Camera instance
        self.camera = None
        self.last_capture_time = 0
        self.camera_initialized = False
        
        # Groq client for natural language generation
        env_vars = dotenv_values(".env")
        api_key = env_vars.get("GroqAPIKey")
        if not api_key:
            logging.error("GroqAPIKey not found in .env file")
            raise ValueError("GroqAPIKey not found in .env file")
        self.groq_client = Groq(api_key=api_key)
        self.system_prompt = (
            "You are an AI assistant with visual capabilities. Describe what you see in a natural, "
            "conversational way as if you're observing the scene. Be concise (1-2 sentences), and "
            "incorporate any relevant context from previous interactions. Respond directly without "
            "markdown formatting or prefix."
        )
        
    def load_known_faces(self):
        """Load known faces from the database"""
        try:
            if not os.path.exists(self.known_faces_dir):
                os.makedirs(self.known_faces_dir)
                logging.info(f"Created known faces directory: {self.known_faces_dir}")
                
            for file in os.listdir(self.known_faces_dir):
                if file.endswith(".pkl"):
                    with open(os.path.join(self.known_faces_dir, file), 'rb') as f:
                        data = pickle.load(f)
                        self.known_face_encodings.append(data['encoding'])
                        self.known_face_names.append(data['name'])
            logging.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {str(e)}")
    
    def save_face(self, name, encoding):
        """Save a new face to the database"""
        try:
            if not name or name.lower() in ["unknown", "this"]:
                return {"status": "error", "message": "I need a valid name to remember this person."}
                
            data = {'name': name, 'encoding': encoding, 'date': datetime.now().isoformat()}
            filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
            filepath = os.path.join(self.known_faces_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Update in-memory database
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            logging.info(f"Saved new face: {name}")
            
            return {
                "status": "success",
                "name": name,
                "raw_data": {"faces": [name]},
                "description": f"I'll remember you as {name}"
            }
        except Exception as e:
            logging.error(f"Error saving face: {str(e)}")
            return {"status": "error", "message": "I couldn't save that face. Please try again."}
    
    def initialize_camera(self):
        """Initialize or reinitialize the camera"""
        if self.camera and self.camera.isOpened():
            return True
            
        # Try different camera indices
        for camera_index in range(0, 3):
            try:
                logging.info(f"Trying to initialize camera index {camera_index}")
                camera = cv2.VideoCapture(camera_index)
                if camera.isOpened():
                    # Set lower resolution for compatibility
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Warm up the camera with minimal frames
                    for _ in range(2):
                        camera.read()
                    
                    self.camera = camera
                    self.camera_initialized = True
                    logging.info(f"Successfully initialized camera index {camera_index}")
                    return True
            except Exception as e:
                logging.error(f"Camera initialization error (index {camera_index}): {str(e)}")
        
        logging.error("Failed to initialize any camera")
        self.camera_initialized = False
        return False
    
    def capture_image(self):
        """Capture an image from the webcam"""
        try:
            # Initialize camera if needed
            if not self.camera_initialized:
                if not self.initialize_camera():
                    return None, "I couldn't access the camera."
            
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                # Attempt to reinitialize camera on failure
                if not self.initialize_camera():
                    return None, "I couldn't capture an image."
                ret, frame = self.camera.read()
                if not ret:
                    return None, "I couldn't capture an image."
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_capture_time = time.time()
            return rgb_frame, None
        except Exception as e:
            logging.error(f"Capture error: {str(e)}")
            return None, "I encountered a problem with the camera."
    
    def extract_name(self, query):
        """Extract name from a 'remember me' query"""
        # Look for patterns like "remember me as John"
        match = re.search(r'remember me (?:as )?([a-zA-Z]+)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for patterns like "my name is John"
        match = re.search(r'my name is ([a-zA-Z]+)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None
    
    def recognize_face(self, name=None):
        """Recognize faces in the captured image"""
        try:
            if name and isinstance(name, str):
                name = self.extract_name(name)
            
            frame, error = self.capture_image()
            if error:
                return {"status": "error", "message": error}
            
            # Find faces in the image
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                return {
                    "status": "success",
                    "raw_data": {"faces": []},
                    "description": "I don't see anyone in the image."
                }
            
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            # Recognize faces
            recognized_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    face_name = self.known_face_names[best_match_index]
                
                recognized_names.append(face_name)
            
            # If we were asked to remember someone
            if name:
                # Save the first face as the specified name
                return self.save_face(name, face_encodings[0])
            
            # Prepare response
            if not recognized_names or all(name == "Unknown" for name in recognized_names):
                return {
                    "status": "success",
                    "raw_data": {"faces": []},
                    "description": "I don't recognize anyone in the image."
                }
                
            # Create natural description
            if len(recognized_names) == 1:
                description = f"I see {recognized_names[0]} here with me."
            else:
                description = f"I see {', '.join(recognized_names[:-1])} and {recognized_names[-1]}."
                
            return {
                "status": "success",
                "raw_data": {"faces": recognized_names},
                "description": description
            }
        except Exception as e:
            logging.error(f"Face recognition error: {str(e)}")
            return {"status": "error", "message": "I encountered an error while recognizing faces."}
    
    def generate_natural_description(self, vision_data, query):
        """Generate a natural language description using the chatbot model"""
        if vision_data.get("status") != "success":
            return vision_data.get("message", "I'm having trouble seeing right now.")
            
        # Prepare the raw data for the chatbot
        raw_description = vision_data.get("description", "I see something interesting")
        raw_data = vision_data.get("raw_data", {})
        
        # Create the prompt
        prompt = (
            f"User asked: '{query}'\n\n"
            f"Raw vision data: {json.dumps(raw_data)}\n\n"
            f"Initial description: {raw_description}\n\n"
            "Please generate a natural, conversational response describing what you see "
            "as if you're observing the scene. Be concise (1-2 sentences)."
        )
        
        # Use Groq API to generate natural response
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.7,
                max_tokens=150,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            # Extract and clean the response
            response = chat_completion.choices[0].message.content
            response = response.replace("**", "").replace("*", "").strip()
            
            # Remove any assistant prefix if present
            if response.startswith("Assistant:"):
                response = response.replace("Assistant:", "").strip()
                
            return response
        except Exception as e:
            logging.error(f"Natural description generation failed: {str(e)}")
            return raw_description
    
    def analyze_scene(self):
        """Analyze the scene for objects using PyTorch model"""
        try:
            if not self.model:
                return {
                    "status": "error",
                    "message": "Vision system is not available."
                }
            
            frame, error = self.capture_image()
            if error:
                return {"status": "error", "message": error}
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Transform and add batch dimension
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run object detection
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Process predictions
            pred = predictions[0]
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter detections with confidence > 0.7
            high_conf_indices = scores > 0.7
            labels = labels[high_conf_indices]
            
            if len(labels) == 0:
                return {
                    "status": "success",
                    "raw_data": {"objects": {}},
                    "description": "I don't see any recognizable objects at the moment."
                }
            
            # Count objects
            object_counts = {}
            for label in labels:
                class_name = COCO_CLASSES[label]
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
            
            # Create initial description
            items = []
            for obj, count in object_counts.items():
                if count == 1:
                    items.append(f"{obj}")
                else:
                    items.append(f"{count} {obj}s")
            
            if not items:
                description = "I don't see any recognizable objects."
            elif len(items) == 1:
                description = f"I see {items[0]}."
            else:
                description = f"I see {', '.join(items[:-1])} and {items[-1]}."
            
            return {
                "status": "success",
                "raw_data": {"objects": object_counts},
                "description": description
            }
        except Exception as e:
            logging.error(f"Scene analysis error: {str(e)}")
            return {
                "status": "error",
                "message": "I had trouble analyzing the scene."
            }

    def __del__(self):
        """Clean up resources when object is destroyed"""
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None