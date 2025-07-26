# Voice Commander 🎙🫡
Local voice transcription for coding using whisper.cpp. Works offline, privacy-focused.

## Example Usage

Uploading test_lq.mp4…

## Features
- F8/F9 hotkeys for recording
- Auto-paste transcribed text
- GPU acceleration support
- VS Code extension included

## Setup
1. **Install whisper.cpp:**
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   ```

2. **Download model:**
   ```bash
   bash ./models/download-ggml-model.sh medium.en
   ```

3. **Install Python dependencies:**
   ```bash
   pip install sounddevice scipy numpy pyperclip pynput
   ```

4. **Run Voice Commander:**
   ```bash
   python portable_commander.py
   ```

## VS Code Extension
See `VScode_extension/` folder for VS Code integration.

## Usage
- Press F8 to start recording
- Press F9 to stop and paste text
- Works in any application

## Requirements
- whisper.cpp compiled in parent directory
- Python 3.7+
- Microphone access 
