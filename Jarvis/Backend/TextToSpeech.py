import pygame
import random
import asyncio
import os
import aiohttp
from dotenv import dotenv_values
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

env_vars = dotenv_values(".env")
ELEVEN_API_KEY = env_vars.get("ELEVEN_API_KEY")
AssistantVoice = env_vars.get("ELEVEN_VOICE_ID")  # ElevenLabs Voice ID

# Initialize mixer once at program start
pygame.mixer.init()

async def TextToAudioFile(text) -> None:
    file_path = r"Data\speech.mp3"

    if os.path.exists(file_path):
        os.remove(file_path)

    if not ELEVEN_API_KEY or not AssistantVoice:
        raise ValueError("Missing ElevenLabs API key or Voice ID in environment variables.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{AssistantVoice}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                error_message = await response.text()
                raise Exception(f"ElevenLabs API request failed: {error_message}")

def TTS(text, func=lambda r=None: True):
    while True:
        try:
            asyncio.run(TextToAudioFile(text))
            
            # Check if mixer needs reinitialization
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(r"Data\speech.mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if func() == False:
                    break
                pygame.time.Clock().tick(10)

            return True
        except Exception as e:
            print(f"Error in TTS: {e}")
        finally:
            try:
                func(False)
                # Only stop music if mixer is initialized
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
            except Exception as e:
                print(f"Error in TTS finally block: {e}")

def TextToSpeech(Text, func=lambda r=None: True):
    # Handle case where Text is a dictionary from vision system
    if isinstance(Text, dict):
        Text = Text.get("description", "I couldn't generate a description.")
    
    # Ensure Text is a string
    if not isinstance(Text, str):
        logging.error(f"Invalid input type for TextToSpeech: {type(Text)}")
        Text = "I'm sorry, I couldn't process that."

    Data = str(Text).split(".")

    responses = [
        "The rest of the result has been printed to the chat screen, kindly check it out sir.",
        "The rest of the text is now on the chat screen, sir, please check it.",
        "You can see the rest of the text on the chat screen, sir.",
    ]
    
    if len(Data) > 4 and len(Text) >= 250:
        TTS(" ".join(Text.split(".")[0:2]) + "." + random.choice(responses), func)
    else:
        TTS(Text, func)

if __name__ == "__main__":
    while True:
        TTS(input("Enter text:"))