#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick check script for Gemini API configuration
Run: python check_gemini.py
"""
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
print("  Gemini API Configuration Check")
print("=" * 60)
print()

# Check 1: .env file exists
print("✓ Check 1: File .env")
if env_file.exists():
    print(f"  ✓ File .env ton tai: {env_file}")
else:
    print(f"  ✗ File .env KHONG ton tai: {env_file}")
    print(f"  → Chay: setup_gemini.bat (Windows) hoac ./setup_gemini.sh (Linux/Mac)")
print()

# Check 2: GEMINI_API_KEY in environment
print("✓ Check 2: Bien moi truong GEMINI_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"  ✓ GEMINI_API_KEY da duoc set")
    print(f"  → Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    print(f"  → Length: {len(api_key)} characters")
else:
    print(f"  ✗ GEMINI_API_KEY CHUA duoc set")
    print(f"  → Kiem tra file .env hoac bien moi truong")
print()

# Check 3: google-genai package
print("✓ Check 3: Package google-genai")
try:
    from google import genai
    print(f"  ✓ Package da duoc cai dat")
    print(f"  → Version: {genai.__version__ if hasattr(genai, '__version__') else 'Unknown'}")
except ImportError:
    print(f"  ✗ Package CHUA duoc cai dat")
    print(f"  → Chay: pip install google-genai")
    sys.exit(1)
print()

# Check 4: Try to initialize Gemini
print("✓ Check 4: Khoi tao Gemini service")
if api_key:
    try:
        client = genai.Client(api_key=api_key)
        print(f"  ✓ Khoi tao thanh cong")
        print(f"  → Model: gemini-2.5-flash (latest stable)")
    except Exception as e:
        print(f"  ✗ Khoi tao THAT BAI")
        print(f"  → Loi: {str(e)}")
        print(f"  → Kiem tra lai API key")
else:
    print(f"  ⊘ Bo qua (khong co API key)")
print()

# Check 5: Test API call (optional)
if api_key and "--test-api" in sys.argv:
    print("✓ Check 5: Test API call (--test-api)")
    try:
        print("  → Dang goi API Gemini...")
        import asyncio
        
        async def test_call():
            response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents="Xin chao, day la test"
            )
            return response.text
        
        result = asyncio.run(test_call())
        print(f"  ✓ API hoat dong binh thuong")
        print(f"  → Response: {result[:50]}...")
    except Exception as e:
        print(f"  ✗ API call THAT BAI")
        print(f"  → Loi: {str(e)}")
        print(f"  → Kiem tra:")
        print(f"    - API key hop le?")
        print(f"    - Da vuot quota?")
        print(f"    - Ket noi internet?")
    print()

# Summary
print("=" * 60)
print("  TONG KET")
print("=" * 60)
print()

if api_key:
    print("✅ Cau hinh CO VE OK!")
    print()
    print("BUOC TIEP THEO:")
    print("  1. Restart backend (neu dang chay):")
    print("     Ctrl+C -> uv run python backend/run.py")
    print()
    print("  2. Kiem tra log khi startup:")
    print("     Nen thay: '✓ Gemini AI service ENABLED'")
    print()
    print("  3. Thuc hien danh gia tu frontend")
    print()
    if "--test-api" not in sys.argv:
        print("💡 Tip: Chay 'python check_gemini.py --test-api' de test API call")
else:
    print("⚠️  Chua cau hinh GEMINI_API_KEY")
    print()
    print("CACH KHAC PHUC:")
    print("  1. Chay setup script:")
    print("     Windows: setup_gemini.bat")
    print("     Linux/Mac: ./setup_gemini.sh")
    print()
    print("  2. Hoac tao file .env thu cong:")
    print("     echo 'GEMINI_API_KEY=your_key' > .env")
    print()
    print("Xem huong dan chi tiet: GEMINI_SETUP.md")

print()
print("=" * 60)
