#!/usr/bin/env python3
import sys
from typing import List, Optional

from analyzer.cli import parse_args


def main(argv: Optional[List[str]] = None) -> int:
    _args = parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
