# parakeet-commander

Replaces `whisper.cpp` with **parakeet.cpp** (`mudler/parakeet.cpp`) running
the `tdt_ctc-110m-f16.gguf` model — a 110 M FastConformer that is measurably
faster than `whisper large-v3` on English while beating it on WER.

## One-time setup

```bash
# from the Voice-Commander repo root:
bash parakeet-commander/setup.sh
```

What the script does:

| Step | Action |
|------|--------|
| 1 | `git clone --recursive` `mudler/parakeet.cpp` into `../parakeet.cpp/` |
| 2 | CMake build with `PARAKEET_GGML_CUDA=ON` (falls back to CPU if no GPU) |
| 3 | Downloads `tdt_ctc-110m-f16.gguf` (~268 MB) from `mudler/parakeet-cpp-gguf` on HF |

## Run

```bash
python parakeet-commander/portable_commander_parakeet.py
```

Same Python deps as the whisper variant:

```bash
pip install sounddevice scipy numpy pyperclip pynput python-dotenv google-genai
```

## Configuration

Same `.env` variables as the whisper variant:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VC_ENABLE_LLM` | `true` | Gemini post-processing |
| `VC_LLM_FORMAT` | `xml` | Output format: `plain` / `xml` / `json` |
| `GEMINI_API_KEY` | — | Required only when LLM is on |
| `VC_PASTE_MODE` | `auto` | `ctrl_v` / `ctrl_shift_v` / `auto` |
| `PARAKEET_DEVICE` | _(auto)_ | Override compute device: `cpu`, `CUDA0`, `Vulkan0`, … |

## Differences from the whisper variant

| | `portable_commander_gpu.py` | `portable_commander_parakeet.py` |
|---|---|---|
| Engine | `whisper-cli` | `parakeet-cli` |
| Model | `ggml-medium.en.bin` | `tdt_ctc-110m-f16.gguf` |
| GPU flag | `CUDA_VISIBLE_DEVICES=0` | `PARAKEET_DEVICE` env var (auto-detect) |
| GPU enforcement | Hard-fail if no GPU | Soft (parakeet runs fine on CPU) |
| Speed | baseline | ~10–50× faster |

All voice-command shortcuts (copy, paste, tab, enter, escape, …) and the
optional Gemini refinement step are preserved verbatim.
