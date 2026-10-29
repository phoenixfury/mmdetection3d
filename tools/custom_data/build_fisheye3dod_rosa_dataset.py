#!/usr/bin/env python3
"""Create synchronized Fisheye3DOD and ROSA exports in one command.

This is the public entry point for the dual-format converter.  The underlying
builder is kept importable under its original name so existing automation does
not break.
"""

from build_improved_detzero_rosa_dataset import main


if __name__ == "__main__":
    main()
