#!/usr/bin/env python3
import os
import sys
from typing import Dict, List, Optional, Tuple

from analyzer.utils import exit_error

try:
    from mytorch.config import build_from_config, load_config
    from mytorch.serialization import save_network
except ImportError as exc:  # e.g. NumPy not installed: still exit with 84
    exit_error(f"Cannot load the mytorch library: {exc}")

HELP_TEXT = """USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i   Configuration file (JSON .conf) describing the network to generate
    nb_i            Number of networks to generate from this configuration (positive integer)

    For each configuration, creates <name>_1.nn ... <name>_<nb>.nn in the current
    directory, where <name> is the configuration file name without its extension.
    If the configuration has a seed s, network k is created with seed s + k - 1:
    the networks differ from each other, and the same command always produces
    the same files."""


def parse_args(argv: List[str]) -> List[Tuple[str, int]]:
    """Return [(config_path, count), ...] or exit with 84."""
    if "-h" in argv or "--help" in argv:
        print(HELP_TEXT)
        sys.exit(0)
    if not argv or len(argv) % 2 != 0:
        exit_error("Expected pairs of arguments: config_file nb [config_file nb ...]. See --help.")
    jobs = []
    for i in range(0, len(argv), 2):
        path, count = argv[i], argv[i + 1]
        if not count.isdigit() or int(count) <= 0:
            exit_error(f"Invalid number of networks '{count}' for '{path}': expected a positive integer.")
        jobs.append((path, int(count)))
    return jobs


def output_names(jobs: List[Tuple[str, int]]) -> List[List[str]]:
    """File names for each job. Numbering continues if the same name appears twice."""
    used: Dict[str, int] = {}
    names = []
    for path, count in jobs:
        base = os.path.splitext(os.path.basename(path))[0]
        start = used.get(base, 0)
        names.append([f"{base}_{start + k}.nn" for k in range(1, count + 1)])
        used[base] = start + count
    return names


def main(argv: Optional[List[str]] = None) -> int:
    jobs = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        # Validate every configuration before writing anything
        configs = [load_config(path) for path, _ in jobs]
        for config, names in zip(configs, output_names(jobs)):
            base_seed = config["seed"]
            for k, name in enumerate(names):
                config["seed"] = None if base_seed is None else base_seed + k
                save_network(build_from_config(config), name)
                print(name)
    except ValueError as exc:
        exit_error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
