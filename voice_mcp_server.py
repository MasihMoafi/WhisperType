#!/usr/bin/env python3
"""MCP Server for WhisperType: OpenRouter or local Whisper.cpp transcription, plus dictation control."""

import os
import sys
import json
import base64
import shutil
import tempfile
import subprocess
from typing import Optional, Dict, Any, List

# Preload system libstdc++ if needed (prevents conda/system libjack conflict with sounddevice)
for p in ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', '/lib/x86_64-linux-gnu/libstdc++.so.6']:
    if os.path.exists(p):
        try:
            import ctypes
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            break
        except Exception:
            pass

# Load environment variables (.env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("WhisperType")

# Paths and configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_CPP_DIR = os.path.join(BASE_DIR, "whisper.cpp")
WHISPER_MODELS_DIR = os.path.join(WHISPER_CPP_DIR, "models")
WHISPERTYPE_CLI = os.path.join(BASE_DIR, "whispertype")
BACKEND = os.environ.get("WHISPERTYPE_BACKEND", "openrouter").strip().lower()
ENGINES = ("cloud", "local", "parakeet")

sys.path.insert(0, os.path.join(BASE_DIR, "Linux"))
import cloud_transcribe as cloud  # noqa: E402  (shared with the dictation engine)


def whispertype(*args: str, env: Optional[Dict[str, str]] = None) -> str:
    """Run the WhisperType CLI, the single owner of dictation state."""
    result = subprocess.run([WHISPERTYPE_CLI, *args], capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, timeout=30)
    return result.stdout.strip() or json.dumps({"error": result.stderr.strip()})


def get_whisper_executable() -> Optional[str]:
    """Find the whisper-cli executable in standard locations."""
    candidates = [
        os.path.join(WHISPER_CPP_DIR, "build", "bin", "whisper-cli"),
        os.path.join(WHISPER_CPP_DIR, "build", "bin", "Release", "whisper-cli"),
        os.path.join(WHISPER_CPP_DIR, "main"),
        shutil.which("whisper-cli"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
    return None


def get_available_models() -> List[Dict[str, Any]]:
    """List available whisper GGML models."""
    models = []
    if os.path.isdir(WHISPER_MODELS_DIR):
        for fname in sorted(os.listdir(WHISPER_MODELS_DIR)):
            if fname.endswith(".bin") and not fname.startswith("for-tests"):
                fpath = os.path.join(WHISPER_MODELS_DIR, fname)
                try:
                    size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                    models.append({
                        "name": fname,
                        "path": fpath,
                        "size_mb": size_mb
                    })
                except OSError:
                    pass
    return models


def get_default_model_path() -> Optional[str]:
    """Find default model path."""
    preferred = [
        os.path.join(WHISPER_MODELS_DIR, "ggml-medium.en.bin"),
        os.path.join(WHISPER_MODELS_DIR, "ggml-base.en.bin"),
        os.path.join(WHISPER_MODELS_DIR, "ggml-small.en.bin"),
        os.path.join(WHISPER_MODELS_DIR, "ggml-tiny.en.bin"),
    ]
    for model in preferred:
        if os.path.isfile(model) and os.path.getsize(model) > 1000000:
            return model

    models = get_available_models()
    if models:
        return models[0]["path"]
    return None


def get_whisper_env() -> Dict[str, str]:
    """Prepare environment variables with whisper shared library paths."""
    env = os.environ.copy()
    lib_dirs = [
        os.path.join(WHISPER_CPP_DIR, "build", "src"),
        os.path.join(WHISPER_CPP_DIR, "build", "ggml", "src"),
        os.path.join(WHISPER_CPP_DIR, "build", "ggml", "src", "ggml-cuda"),
        os.path.join(WHISPER_CPP_DIR, "build", "bin"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    new_paths = [p for p in lib_dirs if os.path.isdir(p) and p not in existing]
    if new_paths:
        env["LD_LIBRARY_PATH"] = ":".join(new_paths + ([existing] if existing else []))
    return env


def check_gpu_status() -> Dict[str, Any]:
    """Check NVIDIA GPU / CUDA status."""
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            gpu_info = res.stdout.strip().split("\n")[0]
            return {"available": True, "details": gpu_info}
    except Exception:
        pass
    return {"available": False, "details": "NVIDIA GPU or nvidia-smi not detected"}


def check_microphone_status() -> Dict[str, Any]:
    """Check microphone and audio input devices."""
    try:
        import sounddevice as sd
        default_input = sd.default.device[0]
        devices = sd.query_devices()
        input_devices = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append({
                    "id": idx,
                    "name": dev.get("name"),
                    "channels": dev.get("max_input_channels"),
                    "samplerate": dev.get("default_samplerate"),
                    "is_default": (idx == default_input)
                })
        return {
            "status": "ready" if input_devices else "no_input_devices",
            "default_device_id": default_input,
            "input_devices_count": len(input_devices),
            "input_devices": input_devices[:5]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_devices_count": 0,
            "input_devices": []
        }


def refine_transcription_llm(text: str, output_format: str = "xml") -> str:
    """Optional post-processing using Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return text

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        format_instruction = ""
        if output_format == "xml":
            format_instruction = "Format the output as XML: <prompt><task>...</task><context>...</context></prompt>"
        elif output_format == "json":
            format_instruction = "Format the output as JSON: {\"task\": \"...\", \"context\": \"...\"}"

        prompt = f"""You are an expert prompt engineer. Clean and structure this transcribed speech into a professional LLM prompt.
Rules:
1. Remove filler words (um, uh, like, you know)
2. Remove bracketed noise artifacts ([NOISE], [MUSIC])
3. Fix grammar and spelling while preserving exact technical terms and code symbols.
{format_instruction}

Input: {text}
Output:"""

        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        refined = response.text.strip() if response and response.text else text
        return refined
    except Exception:
        return text


@mcp.tool()
def voice_status() -> str:
    """Checks microphone status, Whisper.cpp binary availability, models readiness, and GPU support.

    Returns:
        JSON string containing comprehensive status of the WhisperType system.
    """
    executable = get_whisper_executable()
    models = get_available_models()
    default_model = get_default_model_path()
    gpu = check_gpu_status()
    mic = check_microphone_status()
    gemini_key = bool(os.environ.get("GEMINI_API_KEY"))

    status = {
        "status": "ready" if (executable and default_model) else "degraded",
        "whisper_executable": {
            "found": executable is not None,
            "path": executable
        },
        "models": {
            "default_model": default_model,
            "available_count": len(models),
            "available_models": models
        },
        "audio_input": mic,
        "gpu": gpu,
        "backend": {
            "default": BACKEND,
            "model": cloud.MODEL,
            "company_key_configured": bool(os.environ.get("OPENROUTER_API_KEY_COMPANY")),
        },
        "dictation": json.loads(whispertype("status", "--json")),
        "llm_refinement": {
            "gemini_api_key_configured": gemini_key,
            "format": os.environ.get("VC_LLM_FORMAT", "xml")
        }
    }
    return json.dumps(status, indent=2)


@mcp.tool()
def transcribe_audio(
    audio_path: str = "",
    audio_base64: str = "",
    language: str = "en",
    refine_prompt: bool = False,
    model_path: str = "",
    threads: int = 4,
    backend: str = "",
    cloud_model: str = "",
) -> str:
    """Transcribes an audio file or base64-encoded audio with OpenRouter (default) or local Whisper.cpp.

    Args:
        audio_path: Path to local audio file (.wav, .mp3, etc.)
        audio_base64: Base64-encoded audio data (WAV/audio buffer)
        language: Language code for transcription (default: "en")
        refine_prompt: Whether to run Gemini LLM refinement on the transcribed text
        model_path: Optional path to specific ggml model file
        threads: Number of CPU threads to use (default: 4)
        backend: "openrouter" or "local"; empty uses WHISPERTYPE_BACKEND (default openrouter)
        cloud_model: OpenRouter model ID; empty uses WHISPERTYPE_CLOUD_MODEL

    Returns:
        JSON string containing transcription results, raw text, and metadata.
    """
    if not audio_path and not audio_base64:
        return json.dumps({
            "success": False,
            "error": "Either audio_path or audio_base64 must be provided."
        }, indent=2)

    chosen_backend = (backend or BACKEND).strip().lower()
    executable = get_whisper_executable()
    if chosen_backend == "local" and not executable:
        return json.dumps({
            "success": False,
            "error": f"Whisper executable not found in {WHISPER_CPP_DIR}/build/bin/whisper-cli"
        }, indent=2)

    chosen_model = model_path if model_path and os.path.isfile(model_path) else get_default_model_path()
    if chosen_backend == "local" and (not chosen_model or not os.path.isfile(chosen_model)):
        return json.dumps({
            "success": False,
            "error": f"Whisper model file not found in {WHISPER_MODELS_DIR}"
        }, indent=2)

    temp_file_to_clean = None
    target_audio = audio_path

    try:
        if audio_base64:
            try:
                decoded_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to decode audio_base64: {e}"
                }, indent=2)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(decoded_bytes)
            tmp.close()
            temp_file_to_clean = tmp.name
            target_audio = tmp.name

        if not target_audio or not os.path.isfile(target_audio):
            return json.dumps({
                "success": False,
                "error": f"Audio file not found: {target_audio}"
            }, indent=2)

        if chosen_backend != "local":
            model = cloud_model or cloud.MODEL
            raw_text = cloud.transcribe(target_audio, model=model)
            final_text = refine_transcription_llm(raw_text, os.environ.get("VC_LLM_FORMAT", "xml")) if refine_prompt and raw_text else raw_text
            return json.dumps({"success": True, "text": final_text, "raw_transcription": raw_text, "language": language,
                               "model": model, "backend": "openrouter", "refined": refine_prompt}, indent=2)

        cmd = [
            executable,
            "-m", chosen_model,
            "-f", target_audio,
            "-nt",
            "-t", str(threads),
            "-l", language,
        ]

        env = get_whisper_env()
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if res.returncode != 0:
            return json.dumps({
                "success": False,
                "error": f"Whisper transcription failed with exit code {res.returncode}",
                "stderr": res.stderr.strip()
            }, indent=2)

        raw_text = res.stdout.strip()
        final_text = raw_text

        if refine_prompt and raw_text:
            output_format = os.environ.get("VC_LLM_FORMAT", "xml")
            final_text = refine_transcription_llm(raw_text, output_format=output_format)

        return json.dumps({
            "success": True,
            "text": final_text,
            "raw_transcription": raw_text,
            "language": language,
            "model": os.path.basename(chosen_model),
            "backend": "local",
            "refined": refine_prompt
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Exception during transcription: {str(e)}"
        }, indent=2)

    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try:
                os.remove(temp_file_to_clean)
            except OSError:
                pass


@mcp.tool()
def dictation_status() -> str:
    """Reports whether hotkey dictation (Right Ctrl/F8/F9 → text at the cursor) is running, and its engine and model."""
    return whispertype("status", "--json")


@mcp.tool()
def start_dictation(engine: str = "cloud", model: str = "") -> str:
    """Starts hotkey dictation in the background. Only one engine runs at a time.

    Args:
        engine: "cloud" (OpenRouter), "local" (GPU Whisper) or "parakeet" (GPU Parakeet)
        model: OpenRouter model ID for the cloud engine, e.g. nvidia/parakeet-tdt-0.6b-v3
    """
    if engine not in ENGINES:
        return json.dumps({"success": False, "error": f"engine must be one of {', '.join(ENGINES)}"})
    return whispertype(engine, "--background", env={"WHISPERTYPE_CLOUD_MODEL": model} if model else None)


@mcp.tool()
def stop_dictation() -> str:
    """Stops hotkey dictation."""
    return whispertype("stop")


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
