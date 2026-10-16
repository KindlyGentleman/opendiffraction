"""Script: Generate production dataset release."""

import sys
from opendiffraction.cli import main

if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "starter"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    sys.exit(main(["generate", "--preset", preset, "--output", out_dir, "--export-cxi", "--export-tasks"]))
