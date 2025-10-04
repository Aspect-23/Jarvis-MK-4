# Main Application File
from Frontend.GUI import (
    GraphicalUserInterface,
    SetAssistantStatus,
    ShowTextToScreen,
    TempDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    QueryModifier,
    GetMicrophoneStatus,
    GetAssistantStatus
)
from Backend.Model import FirstLayerDMM
from Backend.RealtimeSearchEngine import RealtimeSearchEngine
from Backend.Automation import Automation
from Backend.SpeechToText import SpeechRecognition
from Backend.Chatbot import ChatBot
from Backend.TextToSpeech import TextToSpeech
from Backend.ComputerVision import VisionSystem
from Backend.FileManager import FileManager
from dotenv import dotenv_values
from asyncio import run
from time import sleep, time
import subprocess
import threading
import json
import os
import pyautogui
import keyboard
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_assistant.log"),
        logging.StreamHandler()
    ]
)

# Configuration Constants
env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
AssistantName = env_vars.get("AssistantName")
subprocess = []
Functions = ["open", "close", "play", "system", "content", "control", "vision"]

# Initialize FileManager
file_manager = FileManager()

# Vision triggers - phrases that should activate the vision system
VISION_TRIGGERS = [
    "what do you see", "what's in front of you", "describe what you see",
    "who do you see", "recognize this", "remember me", "what is this",
    "look around", "scan the room", "identify objects", "who am i",
    "do you recognize me", "what's in the room", "what's around you",
    "describe the scene", "what's in front of the camera", "what do you notice",
    "what's visible", "can you see anything", "describe your view"
]

# Vision memory file
VISION_MEMORY_FILE = "Data/VisionMemory.json"

# Proactive speaking settings
PROACTIVE_COOLDOWN = 300  # 5 minutes between proactive messages
last_proactive_time = 0
last_interaction_time = time()
important_events = []

# Chat log lock to prevent concurrent access
chat_log_lock = threading.Lock()

# Helper functions
def read_chat_log():
    """Read chat log with thread safety"""
    with chat_log_lock:
        if not os.path.exists(r'Data\ChatLog.json'):
            return []
        try:
            with open(r'Data\ChatLog.json', 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Error reading chat log: {str(e)}")
            return []

def write_chat_log(chat_log):
    """Write to chat log with thread safety"""
    with chat_log_lock:
        try:
            os.makedirs('Data', exist_ok=True)
            with open(r'Data\ChatLog.json', 'w', encoding='utf-8') as file:
                json.dump(chat_log, file, indent=2)
        except Exception as e:
            logging.error(f"Error writing chat log: {str(e)}")

def ShowDefaultChatIfNoChats():
    os.makedirs('Data', exist_ok=True)
    # Initialize vision memory
    if not os.path.exists(VISION_MEMORY_FILE):
        with open(VISION_MEMORY_FILE, 'w') as f:
            json.dump({"visual_memories": [], "known_faces": [], "last_vision": ""}, f)
    
    # Only create initial message if chat log doesn't exist
    if not os.path.exists(r'Data\ChatLog.json'):
        # Generate initial message
        initial_query = f"Hello {AssistantName}, how are you?"
        initial_answer = ChatBot(initial_query)
        
        # Create initial chat log
        initial_chat = [
            {"role": "user", "content": initial_query, "timestamp": time()},
            {"role": "assistant", "content": initial_answer, "timestamp": time()}
        ]
        write_chat_log(initial_chat)

def ChatLogIntegration():
    chat_log = read_chat_log()
    formatted_chatlog = ""
    for entry in chat_log:
        role_label = Username if entry["role"] == "user" else AssistantName
        formatted_chatlog += f"{role_label}: {entry['content']}\n"
    
    with open(TempDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted_chatlog))

def ShowChatsOnGUI():
    with open(TempDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        pass
    with open(TempDirectoryPath('Database.data'), 'r', encoding='utf-8') as file:
        Data = file.read()
    if Data:
        with open(TempDirectoryPath('Responses.data'), "w", encoding='utf-8') as file:
            file.write('\n'.join(Data.split('\n')))

def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatsOnGUI()

InitialExecution()

# Initialize Vision System with environment variables
vision_system = VisionSystem(env_vars)
camera_lock = threading.Lock()

def should_trigger_vision(query):
    """Check if the query should trigger the vision system"""
    query_lower = query.lower()
    for trigger in VISION_TRIGGERS:
        if trigger in query_lower:
            return True
    return False

def update_vision_memory(result, memory_type="objects"):
    """Update vision memory with new information"""
    try:
        # Load existing memory
        if os.path.exists(VISION_MEMORY_FILE):
            with open(VISION_MEMORY_FILE, 'r') as f:
                memory = json.load(f)
        else:
            memory = {"visual_memories": [], "known_faces": [], "last_vision": ""}
        
        # Update memory based on type
        if memory_type == "objects":
            # Only keep the last 3 object detections
            memory["visual_memories"].append({"time": time(), "result": result})
            if len(memory["visual_memories"]) > 3:
                memory["visual_memories"] = memory["visual_memories"][-3:]
        elif memory_type == "faces":
            memory["known_faces"].append({"time": time(), "result": result})
        
        # Always update last vision
        memory["last_vision"] = result
        
        # Save updated memory
        with open(VISION_MEMORY_FILE, 'w') as f:
            json.dump(memory, f)
            
        return True
    except Exception as e:
        logging.error(f"Failed to update vision memory: {str(e)}")
        return False

def get_vision_context():
    """Get vision context for chatbot"""
    try:
        if not os.path.exists(VISION_MEMORY_FILE):
            return ""
            
        with open(VISION_MEMORY_FILE, 'r') as f:
            memory = json.load(f)
            
        context = ""
        if memory["last_vision"]:
            context += f"Last thing I saw: {memory['last_vision']}\n"
            
        if memory["visual_memories"]:
            context += "Recently I've seen:\n"
            for obj in memory["visual_memories"][-2:]:
                context += f"- {obj['result']}\n"
                
        if memory["known_faces"]:
            context += "People I recognize:\n"
            for face in memory["known_faces"]:
                context += f"- {face['result']}\n"
                
        return context
    except Exception as e:
        logging.error(f"Failed to get vision context: {str(e)}")
        return ""

def get_self_awareness_context():
    """Get context about the assistant's self-awareness"""
    context = (
        "You are an AI assistant with visual capabilities. "
        "You can see through a camera and understand your surroundings. "
        "You remember what you've seen and who you've met. "
        "Respond naturally and conversationally, incorporating your visual understanding."
    )
    return context

def process_vision_command(query):
    """Process a vision-related command"""
    query_lower = query.lower()
    name = None
    
    # Extract name if provided
    if "my name is" in query_lower:
        name_start = query_lower.find("my name is") + len("my name is")
        name = query[name_start:].split()[0].strip()
    elif "remember me" in query_lower:
        name = "You"
    
    # Determine command type
    if "remember me" in query_lower or "who am i" in query_lower or name:
        # Face recognition mode
        result = vision_system.recognize_face(name or "You")
        update_vision_memory(result, "faces")
        return result
    
    elif "who do you see" in query_lower or "who is this" in query_lower:
        result = vision_system.recognize_face()
        update_vision_memory(result, "faces")
        return result
    
    else:
        # Object detection mode
        result = vision_system.analyze_scene()
        update_vision_memory(result, "objects")
        return result

def MainExecution():
    global last_interaction_time
    last_interaction_time = time()  # Update interaction time
    
    TaskExecution = ControlExecution = ImageExecution = VisionExecution = False
    ImageGenerationQuery = ""
    VisionQuery = ""
    assistant_response = None

    SetAssistantStatus("Listening...")
    Query = SpeechRecognition()
    ShowTextToScreen(f"{Username} : {Query}")
    
    # Add user message to chat log
    chat_log = read_chat_log()
    chat_log.append({
        "role": "user",
        "content": Query,
        "timestamp": time()
    })
    write_chat_log(chat_log)
    
    # Check for file-related commands
    if Query.lower().startswith("create folder"):
        folder_name = Query.split(" ", 2)[2]
        success = file_manager.create_folder(folder_name)
        if success:
            assistant_response = f"Folder '{folder_name}' created."
        else:
            assistant_response = f"Folder '{folder_name}' already exists."
    elif Query.lower().startswith("create file"):
        parts = Query.split(" ", 2)
        if len(parts) > 2:
            file_path = parts[2]
            success = file_manager.create_file(file_path)
            if success:
                assistant_response = f"File '{file_path}' created."
            else:
                assistant_response = f"Failed to create file '{file_path}'."
        else:
            assistant_response = "Please specify the file path."
    elif Query.lower().startswith("open in visual studio"):
        parts = Query.split(" ", 3)
        if len(parts) > 3:
            file_path = parts[3]
            file_manager.open_file(file_path)
            assistant_response = f"Opening '{file_path}' in Visual Studio."
        else:
            assistant_response = "Please specify the file path."
    elif Query.lower().startswith("add to main.py"):
        content = Query.split(" ", 2)[2]
        main_py_path = __file__  # Get the path of the current file
        file_manager.write_to_file(main_py_path, f"\n{content}", 'a')
        assistant_response = f"Added '{content}' to main.py."
    elif Query.lower().startswith("generate code for"):
        # Format: "generate code for <description> and add it to <file_path>"
        parts = Query.split(" and add it to ", 1)
        if len(parts) == 2:
            description = parts[0].split(" ", 2)[2]  # Remove "generate code for"
            file_path = parts[1]
            # Use ChatBot to generate code
            code_generation_prompt = f"Generate Python code for {description}"
            generated_code = ChatBot(code_generation_prompt)
            # Write to file
            success = file_manager.write_to_file(file_path, generated_code, 'a')
            if success:
                assistant_response = f"Generated code for {description} and added to {file_path}."
            else:
                assistant_response = f"Failed to add code to {file_path}."
        else:
            assistant_response = "Please specify the description and file path correctly."
    else:
        # Existing logic for other commands
        if should_trigger_vision(Query):
            VisionExecution = True
            VisionQuery = Query
        else:
            SetAssistantStatus("Thinking...")
            Decision = FirstLayerDMM(Query)
            logging.info(f"\nDecision: {Decision}\n")

            C = any(i.startswith("control") for i in Decision)

            for queries in Decision:
                if "generate " in queries:
                    ImageGenerationQuery = str(queries)
                    ImageExecution = True
                    
                if "vision" in queries:
                    VisionQuery = str(queries).replace("vision ", "")
                    VisionExecution = True

            for queries in Decision:
                if not TaskExecution and any(queries.startswith(func) for func in Functions):
                    run(Automation(list(Decision)))
                    TaskExecution = True

            if C:
                run(Automation([q for q in Decision if q.startswith("control")]))
                SetAssistantStatus("Controlling...")
                sleep(1)

            if ImageExecution:
                with open(r"Frontend\Files\ImageGeneration.data", "w") as file:
                    file.write(f"{ImageGenerationQuery},True")
                try:
                    p1 = subprocess.Popen(["python", r'Backend\ImageGeneration.py'],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.PIPE, shell=False)
                    subprocess.append(p1)
                except Exception as e:
                    logging.error(f"ImageGeneration error: {e}")

            for Queries in Decision:
                if "general" in Queries:
                    SetAssistantStatus("Thinking...")
                    # Add vision context to general queries
                    vision_context = get_vision_context()
                    if vision_context:
                        modified_query = f"{get_self_awareness_context()}\n\n{vision_context}\n\nUser: {QueryModifier(Queries.replace('general ',''))}"
                    else:
                        modified_query = QueryModifier(Queries.replace("general ",""))
                    
                    assistant_response = ChatBot(modified_query)
                    ShowTextToScreen(f"{AssistantName} : {assistant_response}")
                    SetAssistantStatus("Answering...")
                    TextToSpeech(assistant_response)

    if VisionExecution:
        SetAssistantStatus("Looking...")
        try:
            # Use camera lock to prevent concurrent access
            with camera_lock:
                result = process_vision_command(VisionQuery)
                assistant_response = result.get("description", "I couldn't generate a description.")
                ShowTextToScreen(f"{AssistantName}: {assistant_response}")
                TextToSpeech(assistant_response)
        except Exception as e:
            assistant_response = "I'm having trouble with the vision system."
            logging.error(f"Vision error: {str(e)}")
            ShowTextToScreen(f"{AssistantName}: {assistant_response}")
            TextToSpeech(assistant_response)
    
    # Add assistant response to chat log if exists
    if assistant_response:
        chat_log = read_chat_log()
        chat_log.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": time()
        })
        write_chat_log(chat_log)
    
    return True

# ================= ENHANCED PROACTIVE BEHAVIOR =================
def check_for_important_events():
    """Check for events that warrant proactive speaking"""
    global last_proactive_time, last_interaction_time, important_events
    
    current_time = time()
    # Only check every 30 seconds and if cooldown has passed
    if current_time - last_proactive_time < PROACTIVE_COOLDOWN:
        return None
    
    try:
        # 1. User presence detection - if user returns after long absence
        if current_time - last_interaction_time > 1800:  # 2. User presence detection - if user returns after long absence
            with camera_lock:
                vision_data = vision_system.recognize_face()
                if vision_data.get("status") == "success":
                    faces = vision_data.get("raw_data", {}).get("faces", [])
                    if any(face != "Unknown" for face in faces):
                        last_interaction_time = current_time
                        return f"You just recognized {', '.join([f for f in faces if f != 'Unknown'])}, who you haven't seen in a while."
        
        # 2. Check for new people in view
        with camera_lock:
            vision_data = vision_system.recognize_face()
            
            if vision_data.get("status") == "success":
                faces = vision_data.get("raw_data", {}).get("faces", [])
                known_faces = [f for f in faces if f != "Unknown"]
                
                # If we see a known face we haven't seen recently
                if known_faces:
                    # Check when we last saw each face
                    if os.path.exists(VISION_MEMORY_FILE):
                        with open(VISION_MEMORY_FILE, 'r') as f:
                            memory = json.load(f)
                            
                        last_seen = {face["name"]: face["last_seen"] for face in memory.get("known_faces", [])}
                        
                        for face in known_faces:
                            if face not in last_seen or current_time - last_seen[face] > 3600:  # 1 hour
                                return f"You just recognized {face}, who you haven't seen in a while."
        
        # 3. Check for important objects
        with camera_lock:
            vision_data = vision_system.analyze_scene()
            
            if vision_data.get("status") == "success":
                objects = vision_data.get("raw_data", {}).get("objects", {})
                logging.info(f"Detected objects: {objects}")  # Log detected objects for debugging
                
                # Define important objects
                important_objs = ["person", "fire", "smoke", "car", "dog", "cat", "package", "delivery"]
                
                for obj in important_objs:
                    if obj in objects:
                        if obj == "person" and objects[obj] > 1:
                            return f"You just detected {objects[obj]} people in the room."
                        elif obj == "fire" or obj == "smoke":
                            return f"You just detected {obj} in the room."
                        elif obj == "dog" or obj == "cat":
                            return f"You just detected a {obj} in the room."
                        elif obj == "package" or obj == "delivery":
                            return f"You just detected a {obj} in the room."
        
        # 4. Time-based reminders
        inactivity_minutes = (current_time - last_interaction_time) // 60
        if inactivity_minutes > 60:  # 1 hour
            return f"It has been {inactivity_minutes} minutes since the last interaction."
        
        # 5. Scheduled events (simulated)
        current_hour = time.localtime().tm_hour
        if current_hour == 9 and random.random() > 0.7:  # Morning
            return "It is currently morning."
        elif current_hour == 13 and random.random() > 0.7:  # Lunch
            return "It is currently lunch time."
        elif current_hour == 18 and random.random() > 0.7:  # Evening
            return "It is currently evening."
        
        # 6. Other proactive thoughts
        if random.random() < 0.05:  # 5% chance
            return "You feel like offering assistance or making a general comment."
            
        return None
    except Exception as e:
        logging.error(f"Proactive event check failed: {str(e)}")
        return None

def vision_monitor():
    """Continuously monitor the camera feed for important objects"""
    while True:
        try:
            # Only monitor when not in conversation
            if GetMicrophoneStatus() == "False" and GetAssistantStatus() == "Standby":
                with camera_lock:
                    vision_data = vision_system.analyze_scene()
                
                if vision_data.get("status") == "success":
                    objects = vision_data.get("raw_data", {}).get("objects", {})
                    logging.info(f"Vision monitor detected objects: {objects}")  # Log for debugging
                    
                    # Define important objects
                    important_objs = ["person", "fire", "smoke", "car", "dog", "cat", "package", "delivery"]
                    
                    for obj in important_objs:
                        if obj in objects:
                            global last_proactive_time
                            last_proactive_time = time()
                            SetAssistantStatus("Proactive")
                            
                            # Generate context for ChatBot
                            if obj == "person" and objects[obj] > 1:
                                context = f"You just detected {objects[obj]} people in the room."
                            else:
                                context = f"You just detected a {obj} in the room."
                            
                            # Generate proactive message using ChatBot
                            proactive_query = f"Generate a proactive message based on the following context: {context}"
                            vision_context = get_vision_context()
                            full_prompt = f"{get_self_awareness_context()}\n\n{vision_context}\n\nUser: {proactive_query}"
                            response = ChatBot(full_prompt)
                            
                            ShowTextToScreen(f"{AssistantName} (Proactive): {response}")
                            TextToSpeech(response)
                            
                            # Add proactive message to chat history
                            chat_log = read_chat_log()
                            chat_log.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": time(),
                                "proactive": True
                            })
                            write_chat_log(chat_log)
                            
                            sleep(3)  # Give time for the message to be spoken
                            SetAssistantStatus("Standby")
                            break  # Exit loop after handling one event to respect cooldown
                            
        except Exception as e:
            logging.error(f"Vision monitor error: {str(e)}")
        
        sleep(5)  # Check every 5 seconds

def proactive_monitor():
    """Monitor for important events and speak proactively"""
    while True:
        try:
            # Only speak when not in conversation
            if GetMicrophoneStatus() == "False" and GetAssistantStatus() == "Standby":
                context = check_for_important_events()
                if context:
                    global last_proactive_time
                    last_proactive_time = time()
                    SetAssistantStatus("Proactive")
                    
                    # Generate proactive message using ChatBot
                    proactive_query = f"Generate a proactive message based on the following context: {context}"
                    vision_context = get_vision_context()  # Get current vision context
                    full_prompt = f"{get_self_awareness_context()}\n\n{vision_context}\n\nUser: {proactive_query}"
                    response = ChatBot(full_prompt)
                    
                    ShowTextToScreen(f"{AssistantName} (Proactive): {response}")
                    TextToSpeech(response)
                    
                    # Add proactive message to chat history
                    chat_log = read_chat_log()
                    chat_log.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time(),
                        "proactive": True
                    })
                    write_chat_log(chat_log)
                    
                    sleep(3)  # Give time for the message to be spoken
                    SetAssistantStatus("Standby")
        except Exception as e:
            logging.error(f"Proactive monitor error: {str(e)}")
        
        sleep(30)  # Check every 30 seconds for non-vision events

# ================= END OF ENHANCED PROACTIVE BEHAVIOR =================

def FirstThread():
    while True:
        if GetMicrophoneStatus() == "True":
            MainExecution()
        else:
            if GetAssistantStatus() != "Standby":
                SetAssistantStatus("Standby")
            sleep(0.5)

def SecondThread():
    GraphicalUserInterface()

if __name__ == "__main__":
    logging.info("Starting AI Assistant...")
    
    # Pre-initialize camera in background
    def pre_init_camera():
        logging.info("Pre-initializing camera...")
        vision_system.initialize_camera()
    threading.Thread(target=pre_init_camera, daemon=True).start()
    
    # Start proactive monitoring
    threading.Thread(target=proactive_monitor, daemon=True).start()
    
    # Start vision monitoring
    threading.Thread(target=vision_monitor, daemon=True).start()
    
    threading.Thread(target=FirstThread, daemon=True).start()
    SecondThread()