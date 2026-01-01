#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List available Gemini models"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load .env file
project_root = Path(__file__).parent
env_file = project_root / ".env"
load_dotenv(env_file)

print("=" * 60)
print("  Danh sach Gemini Models kha dung")
print("=" * 60)
print()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("✗ GEMINI_API_KEY chua duoc set")
    sys.exit(1)

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("Dang lay danh sach models...")
    print()
    
    # List models
    models = client.models.list()
    
    print(f"Co {len(models.models) if hasattr(models, 'models') else 'N/A'} models kha dung:")
    print()
    
    for model in (models.models if hasattr(models, 'models') else models):
        name = model.name if hasattr(model, 'name') else str(model)
        print(f"  - {name}")
        if hasattr(model, 'description') and model.description:
            desc = model.description[:80] + "..." if len(model.description) > 80 else model.description
            print(f"    {desc}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"    Methods: {', '.join(model.supported_generation_methods)}")
        print()
    
except Exception as e:
    print(f"✗ Loi: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
