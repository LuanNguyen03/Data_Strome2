"""
Legacy OLAP build script - redirects to backend.scripts.olap_build for consistency.
This file is kept for backward compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import and run the main OLAP build script
from backend.scripts.olap_build import main

if __name__ == "__main__":
    main()

