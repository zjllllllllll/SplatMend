"""Standard-library-only inert worker for local crash-recovery tests."""
import sys
import time
from pathlib import Path

folder = Path(sys.argv[1])
deadline = time.monotonic() + 40
while time.monotonic() < deadline and not (folder / "release-worker").exists():
    time.sleep(.05)
