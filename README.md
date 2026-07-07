---
name: Voice Commander ~ (vc)
type: DEX (software that improves developer experience)
---

# Voice Commander 🎙️

**Local voice transcription with AI-powered refinement for developers**

Transform your speech into clean, structured prompts using Whisper.cpp (local, GPU-accelerated) + Gemini API (cloud refinement).

## ✨ Features

- 🎤 **Hotkey Recording**: F8/ F9/ R-Ctrl to start/stop
- 🚀 **GPU Acceleration**: CUDA-powered Whisper transcription
- 🤖 **AI Refinement(Optional)**: Gemini-flash cleans up filler words, fixes grammar, structures output
- 📝 **Structured Output**: XML/JSON/plain text formats
- 🔒 **Privacy-First**: Transcription runs locally, only refined text hits API
- ⚡ **Auto-Paste**: Seamlessly inserts text at cursor

## 🎬 Demo

https://github.com/user-attachments/assets/1eeacd19-4602-4b91-96ab-f201c0fc4dd9

## 🎯 Use Cases

- Dictate code comments with "um" and "uh" 
- Convert rambling thoughts into structured prompts
- Hands-free coding when keyboard is unavailable or if you just hate typing!

## Setup

### Linux (GPU-accelerated)
1. **Build whisper.cpp with CUDA:**
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   mkdir build && cd build
   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release -j$(nproc)
   cd ../..
   ```

2. **Download model:**
   ```bash
   cd whisper.cpp/models
   bash download-ggml-model.sh medium.en
   cd ../..
   ```

3. **Install Python dependencies:**
   ```bash
   pip install sounddevice scipy numpy pyperclip pynput python-dotenv google-genai
   ```

4. **Configure AI refinement (optional but recommended):**
   
   Copy the example config:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```bash
   GEMINI_API_KEY=your-api-key-here
   VC_ENABLE_LLM=true
   VC_LLM_FORMAT=xml  # Options: plain, xml, json
   ```
   
   Get a free API key: https://aistudio.google.com/apikey

5. **Run Voice Commander:**
   ```bash
   python Linux/portable_commander_gpu.py
   ```

### macOS/Windows
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

## Usage
- Press F8/ R-ctrl to start recording
- Press F9/ R-ctrl to stop and paste text
- Works everywhere; namely inside your terminal.

## ⚙️ Configuration

Edit `.env` file:

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `VC_ENABLE_LLM` | `true`/`false` | `true` | Enable AI refinement |
| `VC_LLM_FORMAT` | `plain`/`xml`/`json` | `xml` | Output structure |
| `GEMINI_API_KEY` | Your API key | - | Required for refinement |
| `VC_PASTE_MODE` | `auto`/`ctrl_v`/`ctrl_shift_v` | `auto` | Paste behavior |

## 📋 Requirements

- Python 3.7+
- CUDA-capable GPU (for acceleration)
- whisper.cpp compiled in parent directory
- Microphone access
- Gemini API key (free tier available)

**Example:**
```
Input:  "um so like I want to [NOISE] create a function that uh calculates fibonacci"
Output: <prompt><task>Create a function that calculates the Fibonacci sequence</task></prompt>
```

## 🤝 Contributing

PRs welcome! Areas for improvement:
- Additional LLM providers (OpenAI, Anthropic)
- Multi-language support

## 📄 License

MIT License - see [LICENSE](LICENSE) file
