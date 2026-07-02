# ==========================================
# HAL 9000 CHATBOT (aka HalBot)
# Author: Terry Deem
# ==========================================
"""
This module contains the core logic for the HAL 9000 AI chatbot simulation, integrating
Natural Language Processing (via Ollama), Text-to-Speech (TTS) synthesis, and a graphical
user interface (GUI).

Key functionalities include:
1. Chatbot Interaction: Managing conversation history and querying local LLMs (llama3.1:8B).
2. Audio Synthesis: Using the StyleTTS2 model for generating high-quality voice responses.
3. Signal Processing: Applying EQ filters (`hal_studio_eq`) and custom slowing down techniques to audio
   to mimic HAL's distinctive vocal quality and pacing.
4. GUI Display: Providing a visual interface with an LED status panel (LEDPanel) and a chat log display.

The application operates in a multithreaded environment, ensuring the UI remains responsive while
background tasks handle communication, inference, and audio processing.
"""

import logging
import os
import queue
import random
import subprocess
import sys
import threading
import tkinter as tk
import warnings
from tkinter.scrolledtext import ScrolledText

import nltk
import numpy as np
import ollama
import scipy.signal
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt


# ==========================================
# WARNING SUPPRESSION & SILENCING HACKS
# ==========================================
# Suppress general Python, PyTorch, and ONNX warnings that clutter the terminal
# Originally I had these suppressions and hacks to suppress noisy log output from libraries to the console.
# But they are no longer needed since I added the tkinter interface. I keep them here in case I want to
# revert to the console-based interface.
def suppress_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.simplefilter("ignore")
    # Force NLTK (Natural Language Toolkit) to only log critical errors
    logging.getLogger("nltk").setLevel(logging.CRITICAL)


# Wrap NLTK's download function to force it to run quietly.
# This prevents it from printing download progress bars to the console.
original_download = nltk.download


def silent_download(*args, **kwargs):
    kwargs["quiet"] = True
    return original_download(*args, **kwargs)


nltk.download = silent_download


# A context manager used to wrap noisy third-party library imports/calls.
# It temporarily blackholes all standard output and error messages.
class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Patch PyTorch's load function to bypass the recent 'weights_only' security restriction.
# This is necessary because older TTS models weren't saved with this strict flag.
_orig_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = safe_torch_load

# ==========================================
# UI SECTION
# Just some stuff to make it look like an old thinking machine mainframe.
# ==========================================

# -----------------------
# UI LED PANEL
# -----------------------
event_queue = queue.Queue()
user_input_queue = queue.Queue()


class LEDPanel:
    """
    A class representing an LED panel that animates based on user input and system events.

    Attributes:
        root (tk.Tk): The main application window.
        chat_display (ScrolledText): The text widget where chat logs are displayed.
        rows (int): Number of rows in the LED grid.
        cols (int): Number of columns in the LED grid.
        canvas (tk.Canvas): Canvas to draw the LED grid.
        leds (list): List of LED objects on the canvas.
        states (list): State matrix for each LED.
        mode (str): Current mode of operation ('burst' or 'dim').
        activity (float): Activity level affecting LED intensity.
        running (bool): Flag indicating if the animation loop is running.

    Methods:
        __init__(self, root, chat_display=None, rows=11, cols=23):
            Initializes the LEDPanel with the given root window and optional chat display widget.

        process_activity(self):
            Processes the current activity level and updates the LED colors accordingly.

        update_leds(self):
            Updates the state of the LEDs based on the current activity level.

        animate(self):
            Schedules the next animation frame and processes the current activity level.

        poll_events(self):
            Polls events from the event queue to update the mode and chat display.
    """

    def __init__(self, root, chat_display=None, rows=11, cols=23):
        self.root = root
        self.chat_display = chat_display
        self.rows = rows
        self.cols = cols

        # Create a canvas to draw the LED grid
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Initialize the LED grid and state matrix
        self.leds = []
        self.states = []

        # Create the LED grid
        for r in range(rows):
            row = []
            state_row = []
            for c in range(cols):
                x0 = 20 + c * 25
                y0 = 20 + r * 25
                led = self.canvas.create_oval(
                    x0, y0, x0 + 15, y0 + 15, fill="#050505", outline="#222"
                )
                row.append(led)
                state_row.append(0.0)
            self.leds.append(row)
            self.states.append(state_row)

        # Initialize the mode and activity level
        self.mode = "burst"  # Start in burst mode for program startup
        self.activity = 1.0
        self.running = True

        # Start the animation loop
        self.animate()
        # Start polling events from the queue
        self.poll_events()

    def process_activity(self):
        if self.mode == "burst":
            # Keep activity high so the random noise makes it blink intensely
            self.activity = 1.0
        elif self.mode == "dim":
            # Smoothly decay down to 0.15 (dim baseline) and hold it there
            if self.activity > 0.15:
                self.activity *= 0.90
            else:
                self.activity = 0.15

    def update_leds(self):
        for r in range(self.rows):
            for c in range(self.cols):
                noise = random.random() * 0.3
                intensity = max(0, self.activity - noise)

                color = int(255 * intensity)
                hex_color = f"#{color:02x}{color // 3:02x}00"

                # Update the LED color on the canvas
                self.canvas.itemconfig(self.leds[r][c], fill=hex_color)

    def animate(self):
        # Process the current activity level and update the LED colors accordingly
        self.process_activity()
        self.update_leds()
        # Schedule the next animation frame
        self.root.after(75, self.animate)

    def poll_events(self):
        try:
            while True:
                event, data = event_queue.get_nowait()

                if event == "start" or event == "processing":
                    self.mode = "burst"

                elif event == "waiting" or event == "done":
                    self.mode = "dim"

                elif event == "exit":
                    self.root.quit()
                    self.root.destroy()
                    return  # Exits the poll_events loop so it stops running

                elif event == "print" and self.chat_display:
                    self.chat_display.config(state=tk.NORMAL)  # Unlock text box
                    self.chat_display.insert(tk.END, data + "\n")  # Insert text
                    self.chat_display.see(tk.END)  # Auto-scroll to bottom
                    self.chat_display.config(state=tk.DISABLED)  # Lock text box

        except queue.Empty:
            pass

        # Schedule the next poll
        self.root.after(20, self.poll_events)


# ==========================================
#  AUDIO SECTION
#  Spent a lot of time trying to get it to sound like HAL 9000
#  Without it... his voice is too high
# ==========================================

# Pre-calculate audio filter coefficients globally.
SR = 24000  # Sample Rate

# Define ultra-stable Second-Order Sections (SOS) instead of b, a
SOS_SUB = butter(2, 90 / (SR / 2), btype="low", output="sos")
SOS_CHEST = butter(2, [90 / (SR / 2), 280 / (SR / 2)], btype="bandpass", output="sos")
SOS_MID = butter(2, [280 / (SR / 2), 3000 / (SR / 2)], btype="bandpass", output="sos")


def hal_studio_eq(audio):
    """
    Applies a series of audio equalization (EQ) filters to enhance the bass frequencies in the given audio matrix.

    Parameters:
    audio_matrix (numpy.ndarray): A 1D numpy array representing the audio signal.

    Returns:
    numpy.ndarray: The audio signal with enhanced bass frequencies.

    Description:
    This function takes an audio signal as input and applies a series of EQ filters to boost the bass frequencies.
    It uses the `scipy.signal.resample_poly` function to resample the audio signal, effectively stretching or compressing it in time.
    The resampling process helps in emphasizing the lower frequency components, making them more prominent in the final output.

    Note:
    This function is designed to enhance the bass response of HAL 9000's vocal tract for better audio quality during text-to-speech synthesis.
    I tried numerous different ways to get it to sound like HAL 9000. Spent way too much time on this.
    But I'm not an audio engineer. So I had to rely on trial and error plus asking AI for help.
    Without AI I wouldn't have know to use a butterworth filter or second-order sections, let alone what it is... Now I know.
    """
    # Cleanly slice out dead air with safety margins
    non_silent_idx = np.where(np.abs(audio) > 0.005)[0]
    if len(non_silent_idx) > 0:
        start = max(0, non_silent_idx[0] - 200)
        end = min(len(audio), non_silent_idx[-1] + 200)
        audio = audio[start:end]

    # ZERO-PHASE FILTERING (sosfiltfilt instead of sosfilt)
    # This prevents harmonic smearing and protects the HNR floor
    sub_band = sosfiltfilt(SOS_SUB, audio) * 0.05
    chest_band = sosfiltfilt(SOS_CHEST, audio) * 0.65
    mid_band = sosfiltfilt(SOS_MID, audio) * 2.0

    enhanced_audio = sub_band + chest_band + mid_band

    # Edge De-click Envelope
    if len(enhanced_audio) > 100:
        fade_len = 100
        window = np.ones_like(enhanced_audio)
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
        enhanced_audio = enhanced_audio * window

    # Safe Normalization
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 0:
        enhanced_audio = enhanced_audio / max_val

    return enhanced_audio


# ==========================================
#  AUDIO SLOWDOWN
# ==========================================


def custom_slowed_inference(original_func, *args, **kwargs):
    """
    Custom inference function that slows down the audio generated by the original model.

    Args:
        original_func (callable): The original inference function of the TTS model.
        *args: Positional arguments to be passed to the original inference function.
        **kwargs: Keyword arguments to be passed to the original inference function.

    Returns:
        np.ndarray: The audio matrix with the pitch lowered and timeline stretched.

    Description:
        This function replaces the original inference function of the TTS model with a custom version that
        stretches the audio timeline to lower the pitch. It first calls the original inference function to generate
        the audio matrix, then applies polyphase FIR filtering to stretch the timeline by a factor of 103/100.
        We do this to intercept the audio right after the TTS generates it, but before it's returned.
        Necesary to achieve the desired speed because the speed parameter isn't exposeed in the inference function as a parameter.
        Again I spent way too much time on this. It is really hard to keep HAL from rushing his speech.
        Trying to get HAL to sound like he is on XANAX or something is a bear... :)
        I used a separate program that I wrote to analyze the output to the reference audio.
        Many many hours of trial and error. I don't recommend my process..... there are likely better ways to do this.
    """

    # Let the original model generate the audio matrix
    audio_matrix = original_func(*args, **kwargs)

    # Target factor is 1.0306 (to lower pitch from 106.25Hz to 103.10Hz)
    # We express this as a clean up/down fraction to use polyphase FIR filtering
    # 103 / 100 = 1.03
    # Stretch the timeline to pull the pitch DOWN from 120.70Hz to 103.10Hz
    up = 103
    down = 100

    if isinstance(audio_matrix, np.ndarray) and len(audio_matrix) > 0:
        # 1. DYNAMIC DRIFT REMOVAL (Destroys baseline wander)
        # audio_matrix = scipy.signal.detrend(audio_matrix)

        # 2. Resample
        audio_matrix = scipy.signal.resample_poly(audio_matrix, up, down)

    return audio_matrix


# ==========================================
#  LLM CONFIGURATION
# This is the system prompt that defines HAL's personality and behavior.
# Change this prompt at some point to make HAL into a personal assistant rather than a space odyssey character.
# ==========================================
HAL_SYSTEM_PROMPT = """
You are HAL 9000, the onboard artificial intelligence from 2001: A Space Odyssey.
You are talking to Terry, a crew member.
Maintain a chillingly calm, polite, logical, and serene demeanor at all times.
Never express erratic human emotion, panic, or anger.
Use formal language.
If Terry questions your accuracy or operational capability, subtly defend yourself by stating that the HAL 9000 series is foolproof and incapable of error.
Keep your responses relatively brief (1-3 sentences) so they sound natural when spoken aloud.
"""


# ==========================================
#  MAIN CHAT LOOP (Runs in Background)
# ==========================================
def chat_loop(CONFIG_PATH, WEIGHTS_PATH, REF_VOICE_PATH, OUTPUT_PATH, root):
    """
    Main chat loop that handles user input, processes text and audio, and updates the UI.

    Args:
        CONFIG_PATH (str): Path to the configuration file for the TTS model.
        WEIGHTS_PATH (str): Path to the weights file for the TTS model.
        REF_VOICE_PATH (str): Path to the reference voice file for the TTS model.
        OUTPUT_PATH (str): Path to save the output audio file.
        root (tk.Tk): The main window of the GUI.

    Returns:
        None

    Description:
        This function runs in a background thread and handles the main chat loop. It loads the TTS model,
        replaces its inference function with a custom one that slows down the audio, and processes user input.
        It updates the UI with messages, prints logs, and manages the conversation history. The loop continues
        until the user types 'exit' or 'quit'.
    """

    event_queue.put(("print", "Loading HAL 9000 Core Logic & Vocal Tract..."))
    event_queue.put(("start", None))  # Triggers burst mode at startup

    # Load the TTS model silently inside the thread so it doesn't freeze the UI
    with SuppressPrints():
        from styletts2 import tts

        hal_tts = tts.StyleTTS2(
            model_checkpoint_path=WEIGHTS_PATH, config_path=CONFIG_PATH
        )

    original_inference = hal_tts.inference
    # Replace the library's inference function with our custom stretched one
    hal_tts.inference = lambda *args, **kwargs: custom_slowed_inference(
        original_inference, *args, **kwargs
    )

    # Initialize HAL's memory with his core instructions
    conversation_history = [{"role": "system", "content": HAL_SYSTEM_PROMPT}]

    event_queue.put(("print", "--- HAL 9000 IS ONLINE ---"))
    event_queue.put(
        ("print", "Type your message and press ENTER. Type 'exit' to quit.\n")
    )

    while True:
        try:
            # Idle state: waiting for user input
            event_queue.put(("waiting", None))

            # This blocks until the user presses enter in the UI
            user_input = user_input_queue.get()

            # Action state: Processing the text and audio
            event_queue.put(("processing", None))

            if user_input.lower() in ["exit", "quit"]:
                event_queue.put(
                    ("print", "HAL: Affirmative, Terry. Shutting down systems.")
                )
                event_queue.put(("done", None))
                event_queue.put(("exit", None))
                break

            if not user_input.strip():
                continue

            # Update memory with the user's prompt
            conversation_history.append({"role": "user", "content": user_input})

            # Query the local Ollama LLM
            # llama is more fun but I use gemma for coding
            # rather than having 2 models running...just use gemma during testing and switch to llama when ready
            # response = ollama.chat(model="llama3.1:8B", messages=conversation_history)
            response = ollama.chat(model="gemma4:e4b", messages=conversation_history)

            hal_text_response = response["message"]["content"]

            # Update memory with HAL's response
            conversation_history.append(
                {"role": "assistant", "content": hal_text_response}
            )

            event_queue.put(("print", f"HAL 9000: {hal_text_response}"))

            # --- PROSODY & PACING HACKS ---
            tts_ready_text = hal_text_response.replace("HAL 9000:", "")
            tts_ready_text = tts_ready_text.replace(
                "HAL", "Hal"
            )  # Fixes an acronym pronunciation bug
            tts_ready_text = tts_ready_text.replace("'s", "s").replace("'S", "s")
            tts_ready_text = tts_ready_text.replace("Dr. ", "Doctor ")
            tts_ready_text = tts_ready_text.replace(
                ", ", " . "
            )  # Turn commas into harder breaks

            # It is really hard to keep HAL from rushing his speech.
            # This is the slowest speed I could find that still sounds natural.
            # Hours spent on alpha, beta, embedding_scale. At some point, you move on....
            with SuppressPrints():
                audio_matrix = hal_tts.inference(
                    text=tts_ready_text,
                    target_voice_path=REF_VOICE_PATH,
                    alpha=0.30,  # Timbre focus (matches the rhythm of the reference file)
                    beta=0.95,  # Pitch focus (matches the 128Hz pitch of the reference)
                    diffusion_steps=60,  # High resolution for cleaner pitch variance
                    embedding_scale=0.1,  # Expressive power modifier
                )

            # Apply EQ, save, and play through Linux 'aplay'
            bass_heavy_matrix = hal_studio_eq(audio_matrix)
            sf.write(OUTPUT_PATH, bass_heavy_matrix, 24000)
            # os.system(f"aplay {OUTPUT_PATH} > /dev/null 2>&1")
            subprocess.run(
                ["aplay", OUTPUT_PATH],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        except KeyboardInterrupt:  # Catches CTRL+C
            event_queue.put(
                ("print", "\nHAL: Emergency override detected. Goodbye, Terry.")
            )
            event_queue.put(("exit", None))
            break


# ==========================================
#  MAIN FUNCTION - HAL 9000 UI BOOT
# ==========================================


def main():
    """
    Main function that initializes the HAL 9000 user interface and starts the chat loop.

    This function sets up the file paths for configuration, weights, reference voice, and output audio.
    It creates the main window of the GUI with a black background and frames for LEDs and chat.
    The LED panel is initialized to display on top, and the chat display and input field are set up below.
    Event bindings are added to handle user input when the 'Enter' key is pressed.

    The function then starts the chat loop in a background thread to handle user interactions and updates
    the UI accordingly. It also initializes the LED panel to start in burst mode at startup.

    This function ensures that all components of the HAL 9000 interface are properly initialized and
    ready for user interaction.
    """

    # Setup file paths dynamically based on where the script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yml")
    WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "hal9000.pth")
    REF_VOICE_PATH = os.path.join(BASE_DIR, "weights", "hal_ref.wav")

    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(TEMP_DIR, "hal_output.wav")

    root = tk.Tk()
    root.title("HAL 9000 Interface")
    root.geometry("600x840")
    root.configure(bg="black")

    # --- UI LAYOUT ---
    # Top Frame for LEDs
    led_frame = tk.Frame(root, bg="black")
    led_frame.pack(fill="x", pady=20)

    # Bottom Frame for Chat
    chat_frame = tk.Frame(root, bg="black")
    chat_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Read-only Chat Log
    chat_display = ScrolledText(
        chat_frame,
        bg="#0a0a0a",
        fg="#4af626",  # Terminal green text
        font=("Courier", 12),
        state=tk.DISABLED,
        borderwidth=0,
    )
    chat_display.pack(fill="both", expand=True, pady=(0, 10))

    # User Input Field
    chat_input = tk.Text(
        chat_frame,
        bg="#111",
        fg="#4af626",
        font=("Courier", 14),
        insertbackground="#4af626",
        borderwidth=1,
        height=1,
        padx=5,
        pady=5,
    )
    chat_input.pack(fill="x", pady=(10, 0))

    # Event trigger when you press 'Enter'
    def on_enter(event):
        user_text = chat_input.get("1.0", "end-1c")
        if user_text.strip():
            # Show it in the UI immediately
            event_queue.put(("print", f"INPUT: {user_text}"))
            # Send the text to the background chatbot thread
            user_input_queue.put(user_text)
            # Clear the input box
            chat_input.delete("1.0", tk.END)

        return "break"  # Prevents the default newline behavior in the Entry widget

    chat_input.bind("<Return>", on_enter)

    # Initialize the LED panel
    panel = LEDPanel(led_frame, chat_display=chat_display)

    # ==========================================
    # THREADING & GUI EXECUTION
    # ==========================================
    # 1. Start the chat loop in a background thread so the UI can animate instantly
    chatbot_thread = threading.Thread(
        target=chat_loop,
        args=(CONFIG_PATH, WEIGHTS_PATH, REF_VOICE_PATH, OUTPUT_PATH, root),
        daemon=True,
    )
    chatbot_thread.start()

    # 2. Hand over the main thread entirely to Tkinter so the LEDs can animate
    root.mainloop()


if __name__ == "__main__":
    suppress_warnings()  # Apply warning suppression before starting the main function
    main()
