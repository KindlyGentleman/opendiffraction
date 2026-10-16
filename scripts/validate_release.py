"""Script: Execute complete scientific validation audit."""

import sys
from opendiffraction.cli import main

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    reports_dir = sys.argv[2] if len(sys.argv) > 2 else "./reports"
    sys.exit(main(["validate", "--dataset", dataset_dir, "--reports-dir", reports_dir]))
