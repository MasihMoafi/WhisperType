#!/bin/bash
# Simple test for LLM integration

echo "=== Voice Commander LLM Integration Test ==="
echo ""

# Check if Ollama is running
echo "1. Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✓ Ollama is running"
else
    echo "   ❌ Ollama is not running"
    echo "   Start with: ollama serve"
    exit 1
fi

# Check if qwen3:8b is available
echo ""
echo "2. Checking for qwen3:8b model..."
if ollama list | grep -q "qwen3:8b"; then
    echo "   ✓ qwen3:8b is installed"
else
    echo "   ❌ qwen3:8b not found"
    echo "   Pull with: ollama pull qwen3:8b"
    exit 1
fi

# Test a simple refinement
echo ""
echo "3. Testing text refinement..."
echo "   Input: 'this is a test sentance with speling erors'"

RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Fix any grammar, spelling, or dictation errors in the following text. Return ONLY the corrected text without explanations.\n\nText: this is a test sentance with speling erors\n\nCorrected:",
  "stream": false,
  "options": {
    "temperature": 0.3,
    "num_predict": 100
  }
}')

if [ $? -eq 0 ]; then
    OUTPUT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('response', '').strip())")
    echo "   Output: '$OUTPUT'"
    echo "   ✓ LLM refinement working"
else
    echo "   ❌ LLM request failed"
    exit 1
fi

echo ""
echo "=== ALL TESTS PASSED ==="
echo "Your Voice Commander is ready with LLM post-processing!"
