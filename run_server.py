#!/usr/bin/env python3
"""
Run the Wulfram2 server emulator.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from wulfram.server import main

if __name__ == "__main__":
    main()
