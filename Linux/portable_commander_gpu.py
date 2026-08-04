import os
import subprocess
import tempfile
import uuid
import threading
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import numpy as np
import pyperclip
from pynput import keyboard
import sys
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

WHISPER_CPP_DIR = os.path.join(BASE_DIR, "whisper.cpp")
WHISPER_EXECUTABLE = os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-cli")
WHISPER_MODEL_PATH = os.path.join(WHISPER_CPP_DIR, "models", "ggml-medium.en.bin")

# Add whisper library paths to LD_LIBRARY_PATH
WHISPER_LIB_DIR = os.path.join(WHISPER_CPP_DIR, "build", "src")
GGML_LIB_DIR = os.path.join(WHISPER_CPP_DIR, "build", "ggml", "src")
GGML_CUDA_DIR = os.path.join(WHISPER_CPP_DIR, "build", "ggml", "src", "ggml-cuda")
lib_paths = [WHISPER_LIB_DIR, GGML_LIB_DIR, GGML_CUDA_DIR]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_paths = [p for p in lib_paths if os.path.exists(p) and p not in current_ld_path]
if new_paths:
    os.environ['LD_LIBRARY_PATH'] = ":".join(new_paths + [current_ld_path])
HOTKEYS = [keyboard.Key.f8, keyboard.Key.f9, keyboard.Key.ctrl_r]
SAMPLERATE = 16000

# GPU Configuration
GPU_LAYERS = 33  # Medium model has 33 layers
USE_GPU = True
ENFORCE_GPU_ONLY = True  # Fail if GPU not available, never fall back to CPU

# LLM Post-Processing Configuration
ENABLE_LLM_PROCESSING = os.environ.get("VC_ENABLE_LLM", "true").lower() == "true"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_OUTPUT_FORMAT = os.environ.get("VC_LLM_FORMAT", "xml")  # plain, xml, json


def verify_gpu_available():
    """Verify CUDA GPU is available before starting"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        return True
    except FileNotFoundError:
        return False


def _get_active_window_classes_x11():
    try:
        active_line = subprocess.check_output(["xprop", "-root", "_NET_ACTIVE_WINDOW"], text=True)
        win_id = active_line.strip().split()[-1]
        if win_id == "0x0":
            return None
        class_line = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"], text=True)
        if "=" not in class_line:
            return None
        classes_str = class_line.split("=", 1)[1].strip()
        return [c.strip().strip('"').lower() for c in classes_str.split(",")]
    except Exception:
        return None

def _is_terminal_focused_linux():
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11":
        return False
    classes = _get_active_window_classes_x11()
    if not classes:
        return False
    terminal_hints = {
        "gnome-terminal","org.gnome.terminal","gnome-terminal-server","alacritty","kitty",
        "org.wezfurlong.kitty","konsole","yakuake","terminator","xfce4-terminal","mate-terminal",
        "tilix","guake","wezterm","foot","xterm","urxvt","st"
    }
    return any(cls in terminal_hints for cls in classes)

def send_paste(controller: keyboard.Controller):
    mode = os.environ.get("VC_PASTE_MODE", "auto").strip().lower()
    if mode not in {"auto","ctrl_v","ctrl_shift_v"}:
        mode = "auto"
    if mode == "auto":
        chosen = "ctrl_shift_v" if sys.platform.startswith("linux") and _is_terminal_focused_linux() else "ctrl_v"
    else:
        chosen = mode
    if chosen == "ctrl_shift_v":
        with controller.pressed(keyboard.Key.ctrl, keyboard.Key.shift):
            controller.press('v')
            controller.release('v')
    else:
        with controller.pressed(keyboard.Key.ctrl):
            controller.press('v')
            controller.release('v')

def refine_with_llm(text):
    """Post-process transcribed text with Gemini API"""
    if not ENABLE_LLM_PROCESSING or not GEMINI_API_KEY:
        return text
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Build format-specific prompt
        if LLM_OUTPUT_FORMAT == "xml":
            format_instruction = """
Format the output as XML:
<prompt>
  <task>main task description</task>
  <context>any relevant context</context>
  <requirements>specific requirements if any</requirements>
</prompt>"""
        elif LLM_OUTPUT_FORMAT == "json":
            format_instruction = """
Format the output as JSON:
{
  "task": "main task description",
  "context": "any relevant context",
  "requirements": ["requirement1", "requirement2"]
}"""
        else:
            format_instruction = ""
        
        prompt = f"""You are an expert prompt engineer. Clean and structure this transcribed speech into a professional LLM prompt.

Rules:
1. Remove filler words: um, uh, like, you know, so, well, actually
2. Delete [bracketed artifacts] completely: [MUSIC PLAYING], [NOISE], etc.
3. Fix grammar, spelling, and sentence structure
4. Preserve ALL technical terms, code snippets, numbers, and domain-specific language EXACTLY
5. Maintain the user's original intent and meaning

Examples:
Input: "um so like I want to [NOISE] create a function that uh calculates fibonacci"
Output: <prompt><task>Create a function that calculates the Fibonacci sequence</task></prompt>

Input: "can you help me debug this numpy array issue where the shapes don't match"
Output: <prompt><task>Debug a NumPy array issue where shapes don't match</task></prompt>

{format_instruction}

Input: {text}

Output:"""
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        )
        
        print("Refining with Gemini Flash Lite...")
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=contents,
            config=config,
        )
        
        refined = response.text.strip()
        
        if refined:
            print(f"Refined: {refined}")
            return refined
        else:
            print("LLM returned empty response, using original")
            return text
            
    except ImportError:
        print("google-genai library not found. Install with: pip install google-genai")
        print("Using original transcription without LLM refinement")
        return text
    except Exception as e:
        print(f"LLM processing failed: {e}")
        print("Using original transcription")
        return text

class Recorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.lock = threading.Lock()
        self.raw_mode = False

    def start(self, raw_mode=False):
        with self.lock:
            if self.recording:
                return
            self.raw_mode = raw_mode
            mode_label = "RAW" if raw_mode else "REFINED"
            print(f">>> Recording started ({mode_label} mode). Press Right Ctrl/F8/F9 to stop.")
            self.recording = True
            self.audio_data = []
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()

    def stop_and_process(self):
        with self.lock:
            if not self.recording:
                return
            print(">>> Recording stopped. Processing...")
            self.recording = False
        
        def run_processing():
            if hasattr(self, 'record_thread'):
                self.record_thread.join(timeout=1.0)
            self._process_audio_data()
            
        threading.Thread(target=run_processing, daemon=True).start()

    def _record_loop(self):
        chunk_size = 1600  # 100ms chunks to reduce stop latency
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='int16') as stream:
            while self.recording:
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("Warning: Audio buffer overflowed")
                with self.lock:
                    if self.recording:
                        self.audio_data.append(audio_chunk)

    def _process_audio_data(self):
        with self.lock:
            if not self.audio_data:
                print("No audio recorded.")
                return
            recording = np.concatenate(self.audio_data, axis=0)

        print("Transcribing with GPU...")
        tmp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        write_wav(tmp_audio_path, SAMPLERATE, recording)

        # Build command with GPU enforcement
        command = [WHISPER_EXECUTABLE, "-m", WHISPER_MODEL_PATH, "-f", tmp_audio_path, "-nt"]
        
        # Force GPU usage by setting CUDA environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        env['GGML_CUDA_NO_PINNED'] = '0'   # Enable pinned memory for GPU
        
        result = subprocess.run(command, capture_output=True, text=True, env=env)

        # Check if GPU was actually used - look for actual error indicators, not just word "cpu"
        if ENFORCE_GPU_ONLY and result.stderr:
            stderr_lower = result.stderr.lower()
            # Only fail on actual GPU errors, not system info mentioning CPU features
            gpu_error_indicators = [
                'use gpu    = 0' in stderr_lower,  # GPU explicitly disabled
                'cuda' in stderr_lower and 'error' in stderr_lower,  # CUDA errors
                'cuda' in stderr_lower and 'failed' in stderr_lower,  # CUDA failures
                'no cuda' in stderr_lower,  # No CUDA available
                'gpu not available' in stderr_lower,  # Explicit GPU unavailable message
            ]
            if any(gpu_error_indicators):
                os.remove(tmp_audio_path)
                print("ERROR: GPU not available or Whisper fell back to CPU!")
                print("Voice Commander requires GPU. Please check CUDA installation.")
                return

        os.remove(tmp_audio_path)

        original_transcribed_text = result.stdout.strip()
        
        if not original_transcribed_text and result.stderr:
            print(f"Error: {result.stderr}")
            return

        if original_transcribed_text:
            transcribed_text = original_transcribed_text.lower()
            command_text = transcribed_text.strip().rstrip("!.,;:")
            print(f"Transcribed: {original_transcribed_text}")

            controller = keyboard.Controller()
            if command_text == "copy":
                print("Executing: Copy")
                with controller.pressed(keyboard.Key.ctrl):
                    controller.press('c')
                    controller.release('c')
                with controller.pressed(keyboard.Key.ctrl, keyboard.Key.shift):
                    controller.press('c')
                    controller.release('c')
            elif command_text == "paste":
                print("Executing: Paste")
                send_paste(controller)
            elif command_text == "tab" or command_text == "tap":
                print("Executing: Alt+Tab")
                with controller.pressed(keyboard.Key.alt):
                    controller.press(keyboard.Key.tab)
                    controller.release(keyboard.Key.tab)
            elif command_text == "dash":
                print("Executing: Alt+-")
                with controller.pressed(keyboard.Key.alt):
                    controller.press('-')
                    controller.release('-')
            elif command_text == "switch":
                print("Executing: Ctrl+PageDown (Next Terminal Tab)")
                with controller.pressed(keyboard.Key.ctrl):
                    controller.press(keyboard.Key.page_down)
                    controller.release(keyboard.Key.page_down)
            elif command_text == "desktop":
                print("Executing: Super+D (Show Desktop)")
                with controller.pressed(keyboard.Key.cmd):
                    controller.press('d')
                    controller.release('d')
            elif command_text == "exit":
                print("Executing: Ctrl+D")
                with controller.pressed(keyboard.Key.ctrl):
                    controller.press('d')
                    controller.release('d')
            elif command_text == "enter":
                print("Executing: Enter")
                controller.press(keyboard.Key.enter)
                controller.release(keyboard.Key.enter)
            elif command_text == "delete":
                print("Executing: Delete")
                controller.press(keyboard.Key.delete)
                controller.release(keyboard.Key.delete)
            elif command_text == "escape":
                print("Executing: Escape")
                controller.press(keyboard.Key.esc)
                controller.release(keyboard.Key.esc)
            else:
                if self.raw_mode:
                    final_text = original_transcribed_text
                    print("Raw mode: Skipping LLM refinement")
                else:
                    final_text = refine_with_llm(original_transcribed_text)

                pyperclip.copy(final_text)
                print("Text copied to clipboard. Pasting...")
                send_paste(controller)
                print("Paste command sent.")
        elif result.returncode != 0:
            print("Whisper.cpp failed:")
            print(result.stderr)

recorder = Recorder()

def on_press(key):
    if key in HOTKEYS:
        if recorder.recording:
            recorder.stop_and_process()
        else:
            recorder.start(raw_mode=False)

def main():
    # Verify GPU availability at startup
    if ENFORCE_GPU_ONLY and not verify_gpu_available():
        print("=" * 60)
        print("ERROR: GPU NOT AVAILABLE")
        print("=" * 60)
        print("Voice Commander requires CUDA-capable GPU.")
        print("Please ensure:")
        print("  1. NVIDIA GPU is installed")
        print("  2. CUDA drivers are installed (nvidia-smi works)")
        print("  3. Whisper.cpp was compiled with CUDA support")
        print("\nSet ENFORCE_GPU_ONLY=False in code to allow CPU fallback.")
        print("=" * 60)
        sys.exit(1)
    
    print(f"VoiceCommander (GPU) is active.")
    print("  Right Ctrl / F8 / F9: Start/Stop recording → LLM refinement → paste")
    print("GPU acceleration enabled with CUDA.")
    print("⚠️  GPU-ONLY MODE: Will fail if GPU unavailable (no CPU fallback)")
    if ENABLE_LLM_PROCESSING and GEMINI_API_KEY:
        print("LLM post-processing enabled with Gemini Flash Lite")
    else:
        print("LLM post-processing disabled")
    print("The transcribed text will be pasted at your cursor's location.")
    print("Close this window or press Ctrl+C to exit.")
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()
