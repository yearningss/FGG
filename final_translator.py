"""
FGG (Fast Global Translator) Main Launcher.
Backwards compatible entry point for FGG 2.0.
"""

from __future__ import annotations

import sys
from fgg.cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
