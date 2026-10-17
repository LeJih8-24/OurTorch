from dataclasses import dataclass
import os
import sys
from typing import List, Optional

from analyzer.utils import exit_error

HELP_TEXT = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train     Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.
    --save      Save neural network into SAVEFILE. Only works in train mode.
    LOADFILE    File containing an artificial neural network
    CHESSFILE   File containing chessboards"""


@dataclass
class AnalyzerArgs:
    mode: str
    loadfile: str
    chessfile: str
    savefile: Optional[str] = None


def _validate_readable_non_empty_file(filepath: str, label: str) -> None:
    if not os.path.exists(filepath):
        exit_error(f"{label} '{filepath}' does not exist.")
    if not os.path.isfile(filepath):
        exit_error(f"{label} '{filepath}' is not a regular file.")
    if not os.access(filepath, os.R_OK):
        exit_error(f"{label} '{filepath}' is not readable.")
    try:
        if os.path.getsize(filepath) == 0:
            exit_error(f"{label} '{filepath}' is empty.")
    except OSError as exc:
        exit_error(f"Cannot access {label} '{filepath}': {exc}")


def parse_args(argv: Optional[List[str]] = None) -> AnalyzerArgs:
    if argv is None:
        argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        print(HELP_TEXT)
        sys.exit(0)

    mode: Optional[str] = None
    savefile: Optional[str] = None
    positional: List[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--train":
            if mode is not None:
                exit_error("Cannot specify multiple modes (--train / --predict).")
            mode = "train"
            i += 1
        elif arg == "--predict":
            if mode is not None:
                exit_error("Cannot specify multiple modes (--train / --predict).")
            mode = "predict"
            i += 1
        elif arg == "--save":
            if savefile is not None:
                exit_error("--save specified multiple times.")
            if i + 1 >= len(argv):
                exit_error("--save requires a SAVEFILE argument.")
            savefile = argv[i + 1]
            if not savefile or savefile.startswith("-"):
                exit_error("--save requires a valid SAVEFILE argument.")
            i += 2
        elif arg.startswith("-"):
            exit_error(f"Unknown option: {arg}")
        else:
            positional.append(arg)
            i += 1

    if mode is None:
        exit_error("Exactly one mode (--train or --predict) must be provided.")

    if savefile is not None and mode != "train":
        exit_error("--save can only be used with --train mode.")

    if len(positional) != 2:
        exit_error("Both LOADFILE and CHESSFILE must be provided.")

    loadfile, chessfile = positional[0], positional[1]

    _validate_readable_non_empty_file(loadfile, "LOADFILE")
    _validate_readable_non_empty_file(chessfile, "CHESSFILE")

    return AnalyzerArgs(
        mode=mode,
        loadfile=loadfile,
        chessfile=chessfile,
        savefile=savefile,
    )
