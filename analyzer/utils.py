import sys


def exit_error(message: str) -> None:
    """Print an error message to stderr and exit with code 84."""
    sys.stderr.write(f"Error: {message}\n")
    sys.exit(84)
