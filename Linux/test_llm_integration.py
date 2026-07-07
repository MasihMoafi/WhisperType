#!/usr/bin/env python3
"""
Test script for LLM integration with Voice Commander
Tests Ollama connectivity and text refinement
"""

import sys
import json

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        import requests
    except ImportError:
        print("❌ FAIL: requests library not installed")
        print("   Install with: pip install requests")
        return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✓ Ollama is running")
            print(f"  Available models: {', '.join(model_names) if model_names else 'None'}")
            return True, model_names
        else:
            print(f"❌ FAIL: Ollama returned status {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ FAIL: Cannot connect to Ollama")
        print("   Start Ollama with: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False, []

def test_model_availability(model_name="qwen3:8b"):
    """Check if the specified model is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if model_name in model_names:
                print(f"✓ Model {model_name} is available")
                return True
            else:
                print(f"❌ FAIL: Model {model_name} not found")
                print(f"   Pull it with: ollama pull {model_name}")
                return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def test_text_refinement(model_name="qwen3:8b"):
    """Test actual text refinement with sample input"""
    try:
        import requests
        
        test_cases = [
            "this is a test sentance with speling erors",
            "i want to create a funtion that handels user input",
            "the quick brown fox jumps over the lasy dog"
        ]
        
        print(f"\n--- Testing text refinement with {model_name} ---")
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest {i}:")
            print(f"  Input:  '{test_text}'")
            
            prompt = f"""Fix any grammar, spelling, or dictation errors in the following text. Preserve technical terms, code, and commands exactly as they are. Return ONLY the corrected text without explanations.

Text: {test_text}

Corrected:"""
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                refined = result.get('response', '').strip()
                print(f"  Output: '{refined}'")
                print(f"  ✓ Success")
            else:
                print(f"  ❌ FAIL: API returned {response.status_code}")
                return False
        
        print("\n✓ All refinement tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def main():
    print("=" * 60)
    print("Voice Commander LLM Integration Test")
    print("=" * 60)
    
    print("\n1. Testing Ollama connection...")
    ollama_ok, models = test_ollama_connection()
    
    if not ollama_ok:
        print("\n❌ TEST FAILED: Ollama is not accessible")
        sys.exit(1)
    
    print("\n2. Checking model availability...")
    model_ok = test_model_availability("qwen3:8b")
    
    if not model_ok:
        print("\n❌ TEST FAILED: Required model not available")
        sys.exit(1)
    
    print("\n3. Testing text refinement...")
    refine_ok = test_text_refinement("qwen3:8b")
    
    if not refine_ok:
        print("\n❌ TEST FAILED: Text refinement failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nYour Voice Commander is ready to use with LLM post-processing!")
    print("Run with: python Linux/portable_commander_gpu.py")

if __name__ == "__main__":
    main()
