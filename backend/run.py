"""
Run FastAPI backend server
Supports both traditional venv and UV environments
"""
import uvicorn
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Set PYTHONPATH for uvicorn subprocess (needed for reload mode)
if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = str(project_root)
else:
    # Append if already exists
    existing_path = os.environ.get("PYTHONPATH", "")
    if str(project_root) not in existing_path:
        os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_path}"

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Change to project root directory so uvicorn can find the backend module
    os.chdir(project_root)

    # Use string import for reload mode to work properly with UV
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(project_root / "backend")] if reload else None,
    )

