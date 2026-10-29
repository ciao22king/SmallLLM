"""
Entry point so the CLI can be run with: python -m smallllm
"""

import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
