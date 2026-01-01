"""
Quick verification script to check if setup is correct
"""
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_imports():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Backend packages installed")
        return True
    except ImportError as e:
        print(f"❌ Missing backend package: {e}")
        return False

def check_structure():
    """Check if project structure is correct"""
    required_dirs = [
        "backend",
        "frontend",
        "contracts",
        "scripts",
        "analytics",
        "data",
        "docs",
    ]
    missing = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)
    
    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False
    print("✅ Project structure correct")
    return True

def check_contracts():
    """Check if contracts can be imported"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from contracts import AssessmentRequest, AssessmentResponse
        print("✅ Contracts importable")
        return True
    except ImportError as e:
        print(f"❌ Contracts import failed: {e}")
        return False

def main():
    print("Verifying setup...\n")
    results = [
        check_python_version(),
        check_structure(),
        check_imports(),
        check_contracts(),
    ]
    
    print("\n" + "="*50)
    if all(results):
        print("✅ All checks passed! Setup is correct.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

