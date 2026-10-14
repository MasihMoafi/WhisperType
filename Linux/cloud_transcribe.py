#!/usr/bin/env python3
"""OpenRouter speech-to-text, shared by the dictation engine and `whispertype transcribe FILE`."""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

OPENROUTER_API = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("WHISPERTYPE_CLOUD_MODEL", "openai/gpt-4o-mini-transcribe")
LANGUAGE = "en"
SESSION = requests.Session()


def headers():
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY_COMPANY']}"}


def warm():
    """Open the TLS connection while the user speaks, so stop-to-text skips the handshake."""
    try:
        SESSION.get(f"{OPENROUTER_API}/key", headers=headers(), timeout=10)
    except Exception:
        pass


def encode_audio(path):
    """FLAC is ~40% smaller than WAV, so it uploads faster; fall back to the original file without ffmpeg."""
    flac_path = path + ".flac"  # a seekable file gets a complete FLAC header; a pipe does not
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
             "-compression_level", "0", flac_path],
            capture_output=True, check=True,
        )
        with open(flac_path, "rb") as f:
            return f.read(), "flac"
    except (OSError, subprocess.CalledProcessError):
        with open(path, "rb") as f:
            return f.read(), os.path.splitext(path)[1].lstrip(".").lower() or "wav"
    finally:
        if os.path.exists(flac_path):
            os.remove(flac_path)


def transcribe(path, model=None):
    """Return the transcription text; raise on any request failure."""
    audio, audio_format = encode_audio(path)
    response = SESSION.post(
        f"{OPENROUTER_API}/audio/transcriptions",
        headers=headers(),
        json={"model": model or MODEL, "language": LANGUAGE,
              "input_audio": {"data": base64.b64encode(audio).decode(), "format": audio_format}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("text", "").strip()


def main(source):
    """Transcribe a file, or with "-" read audio from stdin after warming the connection."""
    if source == "-":
        warm()  # runs while the caller is still recording
        path = os.path.join(tempfile.mkdtemp(prefix="whispertype-"), "clip")
        with open(path, "wb") as f:
            f.write(sys.stdin.buffer.read())
    else:
        path = source
    started = time.time()  # speech end: the audio is complete
    try:
        text = transcribe(path)
    except Exception as error:
        print(json.dumps({"error": str(error), "model": MODEL}))
        return 1
    finally:
        if source == "-":
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    print(json.dumps({"text": text, "seconds": round(time.time() - started, 3), "model": MODEL}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: cloud_transcribe.py AUDIO_FILE|-")
    sys.exit(main(sys.argv[1]))
