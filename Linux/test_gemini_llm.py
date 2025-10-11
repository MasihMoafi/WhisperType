#!/usr/bin/env python3
"""Test Gemini LLM integration for voice commander"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_refinement():
    """Test the Gemini API refinement function"""
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
        
        print("✓ API key loaded")
        
        # Test with sample transcription
        test_input = "[MUSIC PLAYING] um, so like, I want to create a function that uh calculates the fibonacci sequence"
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"""You are a prompt refinement assistant. Take the user's spoken input and:
1. Remove filler words (um, uh, like, you know, etc.)
2. Remove anything in [brackets] - these are transcription artifacts
3. Fix grammar and spelling errors
4. Restructure into a clear, well-formed prompt suitable for an LLM
5. Preserve technical terms, code, and commands exactly

Return ONLY the refined prompt. No explanations, no thinking process, no commentary.

Input: {test_input}

Refined prompt:"""
        
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
        
        print("\n📝 Testing with input:")
        print(f"   '{test_input}'")
        print("\n⏳ Calling Gemini API...")
        
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=contents,
            config=config,
        )
        
        refined = response.text.strip()
        
        print("\n✨ Refined output:")
        print(f"   '{refined}'")
        
        if refined and len(refined) > 0:
            print("\n✅ Gemini LLM integration working!")
            return True
        else:
            print("\n❌ Empty response from Gemini")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install google-genai python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gemini LLM Integration")
    print("=" * 60)
    
    success = test_gemini_refinement()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed")
    print("=" * 60)
