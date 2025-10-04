import asyncio
from random import randint
from PIL import Image
import requests
from dotenv import get_key
import os
from time import sleep
from pathlib import Path

# Set up paths
DESKTOP = Path.home() / "Desktop"
JARVIS_ROOT = DESKTOP / "Jarvis"
DATA_DIR = JARVIS_ROOT / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Correct API endpoint for Stable Diffusion 2.1 (free tier compatible)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {get_key('.env', 'HUGGINGFACE_API_KEY')}"}

async def query(payload):
    try:
        response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

async def generate_images(prompt: str):
    tasks = []
    for _ in range(4):
        payload = {
            "inputs": f"{prompt}, high quality, detailed, sharp focus",
            "options": {
                "use_cache": True,
                "wait_for_model": True
            }
        }
        tasks.append(asyncio.create_task(query(payload)))
    
    image_bytes_list = await asyncio.gather(*tasks)

    # Check for successful generations
    success_count = 0
    for i, image_bytes in enumerate(image_bytes_list):
        if image_bytes and len(image_bytes) > 1024:  # Minimum size check
            file_path = DATA_DIR / f"{prompt.replace(' ', '_')}{i+1}.jpg"
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            success_count += 1
    
    return success_count

def open_images(prompt):
    prompt = prompt.replace(" ", "_")
    found_files = 0
    for i in range(1, 5):
        image_path = DATA_DIR / f"{prompt}{i}.jpg"
        try:
            if image_path.exists():
                img = Image.open(image_path)
                print(f"Opening image: {image_path}")
                img.show()
                sleep(1)
                found_files += 1
            else:
                print(f"Image not generated: {image_path}")
        except IOError:
            print(f"Invalid image file: {image_path}")
    
    return found_files

def GenerateImage(prompt: str):
    print(f"Using data directory: {DATA_DIR}")
    success_count = asyncio.run(generate_images(prompt))
    sleep(2)
    found_count = open_images(prompt)
    
    if success_count == 0:
        print("Failed to generate any images. Possible reasons:")
        print("- Invalid/expired API key")
        print("- Insufficient API credits")
        print("- Model loading timeout")
        print("Check your Hugging Face account status")

# Main loop
while True:
    try:
        data_file = JARVIS_ROOT / "Frontend" / "Files" / "ImageGeneration.data"
        with open(data_file, "r") as f:
            data = f.read()

        prompt, status = data.split(",")
        
        if status.strip() == "True":
            print("Generating Images...")
            GenerateImage(prompt=prompt.strip())
            
            with open(data_file, "w") as f:
                f.write("False,False")
            break
        else:
            sleep(1)
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        sleep(1)