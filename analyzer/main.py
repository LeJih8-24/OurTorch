#!/usr/bin/env python3
import sys
from typing import List, Optional

from analyzer.cli import parse_args
from analyzer.utils import exit_error

try:
    from analyzer.pipeline import run
except ImportError as exc:  # e.g. NumPy not installed: still exit with 84
    exit_error(f"Cannot load the mytorch library: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as exc:
        # Corrupted network, invalid FEN, missing label, unwritable SAVEFILE...
        exit_error(str(exc))
    except MemoryError:
        exit_error("Not enough memory to process the chessboard file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
