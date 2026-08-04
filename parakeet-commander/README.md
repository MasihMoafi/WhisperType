# parakeet-commander

High-speed local English speech dictation using **parakeet.cpp** (`mudler/parakeet.cpp`) with the **NVIDIA Parakeet TDT 0.6B** (`tdt-0.6b-v2-f16.gguf`) model.

Built on non-autoregressive FastConformer architecture running via CUDA GPU acceleration.

---

## Benchmark Comparison (GPU)

Measured sequentially on real 30-second audio dictation on NVIDIA RTX 3070 Laptop GPU:

| Engine | Model | Parameter Count | Latency (30s audio) | Speedup | Accuracy (WER) |
|---|---|---|---|---|---|
| **Whisper.cpp** | `medium.en` | 769 Million | 4,821 ms | 1.0x (baseline) | High |
| **Parakeet.cpp** | `tdt-0.6b-v2-f16` | 600 Million | **712 ms** | **6.77x faster** | **Whisper Large v3 level** |

---

## Key Features

- **CUDA-Backed Acceleration**: Uses `PARAKEET_DEVICE=CUDA0` for fast GPU inference.
- **Zero-Latency Filler Word Removal**: Deterministic regex filter automatically strips spoken hesitations (`um`, `uh`, `ah`, `er`, `hm`, `hmm`, `mhm`) without cloud API delays.
- **Recurrent Mishearing Correction**: In-memory `WORD_REPLACEMENTS` map to fix domain-specific terms.
- **Direct Clipboard Insertion**: Copies transcribed text straight to clipboard and sends active window paste (`Ctrl+V` or `Ctrl+Shift+V`).

---

## One-Time Setup

```bash
# From the Voice-Commander repo root:
bash parakeet-commander/setup.sh
```

What `setup.sh` does:
1. Clones `mudler/parakeet.cpp` into `parakeet.cpp/` (if missing).
2. Builds `parakeet-cli` with `PARAKEET_GGML_CUDA=ON`.
3. Downloads `tdt-0.6b-v2-f16.gguf` (~1.4 GB) from Hugging Face.

---

## Running

Using shortcut alias (if added to `~/.bash_aliases`):
```bash
vc2
```

Or directly via Python:
```bash
python parakeet-commander/portable_commander_parakeet.py
```

### Hotkeys
- **`F8` / `F9` / `Right Ctrl`**: Start/stop recording.

---

## Latency Benchmark Tool

To measure exact processing latency on your GPU:
```bash
python parakeet-commander/benchmark.py
```

---

## Custom Word Replacements

To auto-correct recurrently misheard technical terms, edit `WORD_REPLACEMENTS` in `portable_commander_parakeet.py`:

```python
WORD_REPLACEMENTS = {
    "am I article": "a my article",
    "codex": "Codex",
}
```
