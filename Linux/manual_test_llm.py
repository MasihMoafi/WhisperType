#!/usr/bin/env python3
"""
Manual test for LLM refinement - tests the exact function used in Voice Commander
"""

import sys
import json

def refine_with_llm(text, model="qwen3:8b"):
    """Post-process transcribed text with Ollama LLM - EXACT copy from portable_commander_gpu.py"""
    try:
        import requests

        # Quick connection check first (1s timeout)
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
        except:
            print("⚠️  Ollama not running - using original text")
            return text

        prompt = f"""Fix any grammar, spelling, or dictation errors in the following text. Preserve technical terms, code, and commands exactly as they are. Return ONLY the corrected text without explanations.

Text: {text}

Corrected:"""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        }

        print(f"Refining with {model}...")
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            refined = result.get('response', '').strip()
            if refined:
                print(f"✓ Refined: {refined}")
                return refined
            else:
                print("LLM returned empty response, using original")
                return text
        else:
            print(f"LLM API error {response.status_code}, using original")
            return text
            
    except ImportError:
        print("❌ requests library not found. Install with: pip install requests")
        return text
    except Exception as e:
        print(f"❌ LLM processing failed: {e}")
        return text

if __name__ == "__main__":
    import sys
    sys.stdout.flush()  # Force immediate output

    print("=" * 70)
    print("Voice Commander LLM Refinement Test")
    print("=" * 70)
    sys.stdout.flush()

    # Test cases simulating whisper.cpp output
    test_cases = [
        "this is a test sentance with speling erors",
        "i want to create a funtion that handels user input",
        "the quick brown fox jumps over the lasy dog",
        "lets right some code for a python scrip",
        "import numpy as np and create an aray"
    ]
    
    print("\nTesting LLM refinement with sample transcriptions:\n")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Original: '{test_text}'")
        sys.stdout.flush()
        refined = refine_with_llm(test_text)
        print()
        sys.stdout.flush()
    
    print("=" * 70)
    print("Test complete! If you see refined text above, integration works.")
    print("=" * 70)
