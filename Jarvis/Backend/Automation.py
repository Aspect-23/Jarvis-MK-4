from AppOpener import close, open as appopen
from webbrowser import open as webopen
from pywhatkit import search, playonyt
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from rich import print
from groq import Groq
import win32gui
import win32process
import psutil
import pywinauto
import pythoncom
import wmi
import pyautogui
import webbrowser
import subprocess
import requests
import keyboard
import asyncio
import time
import os
import yaml
import re

# Initialize Windows COM
pythoncom.CoInitialize()

# Environment Setup
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")

# Constants and Configurations
classes = ["zCubwf", "hgKElc", "LTKOO SY7ric", "ZOLcW", "gsrt vk_bk FzvWSb YwPhnf", 
          "pclqee", "tw-Data-text tw-text-small tw-ta", "IZ6rdc", "05uR6d LTKOO", 
          "vlzY6d", "webanswers-webanswers_table_webanswers-table", "dDoNo ikb4Bb gsrt", 
          "sXLa0e", "LWkfKe", "VQF4g", "qv3Wpe", "kno-rdesc", "SPZz6b"]
useragent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'

# Groq Client Setup
client = Groq(api_key=GroqAPIKey)

professional_responses = [
    "Your satisfaction is my top priority, and I am always here for you sir.",
    "I am dedicated to providing you with the best service possible, sir.",
    "Your needs are my command, and I will always strive to meet them, sir.",
]

messages = []
SystemChatBot = [{"role": "system", "content": f"Hello, I am {os.environ['Username']}, You're a content writer. You have to write content like letter, blogs, email or reply to messages of {os.environ['Username']} friends on behalf of him if he is busy, etc. You have to write content in a professional way. You have to use the following professional responses when required: {professional_responses}. Always give what user is asking for and be on point and concise in your responses. Do not use any unnecessary words or sentences. Do not mention your training data or any other information about yourself."}]

# Plugin Setup
plugins_dir = os.path.join(os.path.dirname(__file__), 'plugins')

def setup_example_plugin():
    """Set up an example Chrome plugin if plugins directory is empty."""
    chrome_dir = os.path.join(plugins_dir, 'chrome')
    images_dir = os.path.join(chrome_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    chrome_yaml = {
        'app': 'Chrome',
        'actions': [
            {
                'name': 'open',
                'pattern': 'open',
                'steps': [
                    {'type': 'click', 'image': 'chrome_icon.png'},
                    {'type': 'wait', 'seconds': 2}
                ]
            },
            {
                'name': 'search',
                'pattern': 'search for (.*)',
                'parameters': ['query'],
                'steps': [
                    {'type': 'click', 'image': 'search_bar.png'},
                    {'type': 'type', 'text': '{query}'},
                    {'type': 'press', 'key': 'enter'}
                ]
            }
        ]
    }
    
    with open(os.path.join(plugins_dir, 'chrome.yaml'), 'w') as f:
        yaml.safe_dump(chrome_yaml, f)
    print("Created example Chrome plugin in plugins/chrome.yaml")
    print("Please capture 'chrome_icon.png' and 'search_bar.png' in plugins/chrome/images/")

if not os.path.exists(plugins_dir):
    os.makedirs(plugins_dir)
    setup_example_plugin()

def load_plugins():
    """Load all YAML plugin files from the plugins directory."""
    plugins = {}
    for filename in os.listdir(plugins_dir):
        if filename.endswith('.yaml'):
            with open(os.path.join(plugins_dir, filename), 'r') as f:
                plugin = yaml.safe_load(f)
                app_name = plugin['app'].lower()
                plugins[app_name] = plugin
    return plugins

plugins = load_plugins()

def execute_action(app, steps, params):
    """Execute the steps defined in a plugin action."""
    for step in steps:
        step_type = step['type']
        if step_type == 'click':
            image_path = os.path.join(plugins_dir, app, 'images', step['image'])
            location = None
            for _ in range(3):  # Retry 3 times
                location = pyautogui.locateOnScreen(image_path, confidence=0.8)
                if location:
                    break
                time.sleep(1)
            if location:
                pyautogui.click(location)
            else:
                raise RuntimeError(f"Image {step['image']} not found for app {app}")
        elif step_type == 'type':
            text = step['text'].format(**params)
            pyautogui.typewrite(text)
        elif step_type == 'press':
            key = step['key']
            pyautogui.press(key)
        elif step_type == 'wait':
            seconds = step['seconds']
            time.sleep(seconds)
        else:
            print(f"Unknown step type: {step_type}")

def capture_image():
    """Capture a screenshot of a selected screen region."""
    print("Move mouse to top-left corner of the region and press Enter")
    input()
    x1, y1 = pyautogui.position()
    print("Move mouse to bottom-right corner of the region and press Enter")
    input()
    x2, y2 = pyautogui.position()
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    app_name = input("Enter app name (e.g., 'chrome'): ").strip().lower()
    image_name = input("Enter image name (e.g., 'icon.png'): ").strip()
    image_dir = os.path.join(plugins_dir, app_name, 'images')
    os.makedirs(image_dir, exist_ok=True)
    filepath = os.path.join(image_dir, image_name)
    screenshot.save(filepath)
    print(f"Saved screenshot to '{filepath}'")

# Universal Control Functions
def get_installed_apps():
    """Retrieve installed applications from Windows registry"""
    try:
        installed_apps = []
        c = wmi.WMI()
        for product in c.Win32_Product():
            installed_apps.append(product.Name.lower())
        return installed_apps
    except Exception as e:
        print(f"Error retrieving installed apps: {e}")
        return []

def get_active_app():
    """Get currently focused application"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        return process.name().lower().replace('.exe', '')
    except Exception as e:
        print(f"Error getting active app: {e}")
        return "unknown"

def universal_control(action: str):
    """Universal application control system"""
    try:
        current_app = get_active_app()
        installed_apps = get_installed_apps()
        
        # Universal action mapping
        universal_actions = {
            'close': 'alt+f4',
            'minimize': 'alt+space+n',
            'maximize': 'alt+space+x',
            'play/pause': 'space',
            'next': 'ctrl+right',
            'previous': 'ctrl+left',
            'search': 'ctrl+f',
            'volume up': 'volumeup',
            'volume down': 'volumedown',
            'mute': 'volumemute'
        }

        # Try to find matching app from installed list
        target_app = next((app for app in installed_apps if app in action.lower()), None)
        
        # App-specific automation
        try:
            if target_app == 'spotify':
                app = pywinauto.Application(backend='uia').connect(path='Spotify.exe')
                if 'play' in action.lower():
                    app.window().set_focus().type_keys('^p')
                    return "Playing Spotify"
            
            elif target_app == 'chrome':
                if 'new tab' in action.lower():
                    pyautogui.hotkey('ctrl', 't')
                    return "New tab opened"
        
        except Exception as app_error:
            print(f"App-specific control failed: {app_error}")

        # Try universal action mapping
        action_key = next((key for key in universal_actions if key in action.lower()), None)
        if action_key:
            if '+' in universal_actions[action_key]:
                keys = universal_actions[action_key].split('+')
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(universal_actions[action_key])
            return f"Executed {action_key}"
        
        return "Action not recognized"
    
    except Exception as e:
        return f"Control error: {str(e)}"

# Core Functionality
def GoogleSearch(Topic):
    search(Topic)
    return True

def Content(Topic):
    def OpenNotepad(File):
        default_text_editor = "notepad.exe"
        subprocess.Popen([default_text_editor, File])

    def ContentWriterAI(prompt):
        messages.append({"role": "user", "content": f"{prompt}"})

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=SystemChatBot + messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=1.0,
            stream=True,
            stop=None
        )

        Answer = ""
        in_think_tag = False
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if "<think>" in content:
                    in_think_tag = True
                    content = content.split("<think>")[-1]
                if "</think>" in content:
                    in_think_tag = False
                    content = content.split("</think>")[-1]
                if not in_think_tag:
                    Answer += content
        
        Answer = Answer.replace("</s>", "")
        messages.append({"role": "assistant", "content": Answer})
        return Answer
    
    Topic = Topic.replace("Content ", "")
    ContentByAI = ContentWriterAI(Topic)

    with open(rf"Data\{Topic.lower().replace(' ', '')}.txt", "w", encoding="utf-8") as file:
        file.write(ContentByAI)
    
    OpenNotepad(rf"Data\{Topic.lower().replace(' ', '')}.txt")
    return True

def YouTubeSearch(Topic):
    Url4Search = f"https://www.youtube.com/results?search_query={Topic}"
    webbrowser.open(Url4Search)
    return True

def PlayYoutube(query):
    playonyt(query)
    return True

def OpenApp(app, sess=requests.Session()):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)
        return True
    except:
        def extract_links(html):
            soup = BeautifulSoup(html, 'html.parser')
            return [link.get('href') for link in soup.find_all('a', {'jsname': 'UWckNb'})]
        
        def search_google(query):
            response = sess.get(f"https://www.google.com/search?q={query}", 
                             headers={'User-Agent': useragent})
            return response.text if response.status_code == 200 else None
        
        if html := search_google(app):
            if links := extract_links(html):
                webopen(links[0])
        return True

def CloseApp(app):
    if "chrome" not in app:
        try:
            close(app, match_closest=True, output=True, throw_error=True)
            return True
        except:
            return False
    return False

def System(command):
    key_actions = {
        "mute": 'volume mute',
        "unmute": 'volume mute',
        "volume up": 'volume up',
        "volume down": 'volume down'
    }
    if command in key_actions:
        keyboard.press_and_release(key_actions[command])
    return True

# Command Processor
async def TranslateAndExecute(commands: list[str]):
    funcs = []
    for command in commands:
        if "in " in command.lower():
            try:
                parts = command.split(", ", 1)
                if len(parts) == 2 and parts[0].lower().startswith("in "):
                    app = parts[0][3:].strip().lower()
                    action_str = parts[1].strip()
                    if app in plugins:
                        for action in plugins[app]['actions']:
                            pattern = action['pattern']
                            match = re.match(pattern, action_str)
                            if match:
                                params = match.groups()
                                param_names = action.get('parameters', [])
                                if len(params) == len(param_names):
                                    param_dict = dict(zip(param_names, params))
                                    funcs.append(asyncio.to_thread(execute_action, app, action['steps'], param_dict))
                                    break
                        else:
                            print(f"No matching action for '{action_str}' in app '{app}'")
                    else:
                        print(f"App '{app}' not found in plugins")
            except Exception as e:
                print(f"Error processing plugin command '{command}': {e}")
        else:
            cmd_map = {
                "open ": (OpenApp, "open "),
                "close ": (CloseApp, "close "),
                "play ": (PlayYoutube, "play "),
                "content ": (Content, "content "),
                "google search ": (GoogleSearch, "google search "),
                "youtube search ": (YouTubeSearch, "youtube search "),
                "system ": (System, "system "),
                "control ": (universal_control, "control ")
            }
            
            for prefix, (func, remove_prefix) in cmd_map.items():
                if command.startswith(prefix):
                    funcs.append(asyncio.to_thread(func, command.removeprefix(remove_prefix)))
                    break
            else:
                print(f"No Function Found for {command}")

    results = await asyncio.gather(*funcs, return_exceptions=True)
    for result in results:
        yield str(result)

async def Automation(commands: list[str]):
    async for result in TranslateAndExecute(commands):
        print(f"Action Result: {result}")
    return True

# Main Execution
if __name__ == "__main__":
    while True:
        choice = input("Enter 'c' to capture an image, 'r' to run a command, or 'exit' to quit: ").lower()
        if choice == 'exit':
            break
        elif choice == 'c':
            capture_image()
        elif choice == 'r':
            command = input("Enter command: ")
            asyncio.run(Automation([command]))
        else:
            print("Invalid option. Try again.")