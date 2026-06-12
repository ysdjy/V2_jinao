"""Put the project root on sys.path so `import franka_d435_foundationpose` works.

Every script imports this first. The project root is the directory that contains
both this `scripts/` folder and the `franka_d435_foundationpose/` package.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
