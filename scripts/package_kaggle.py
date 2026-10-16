"""Script: Package dataset artifacts for Kaggle release."""

import sys
from opendiffraction.cli import main

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    pkg_dir = sys.argv[2] if len(sys.argv) > 2 else "./kaggle_release"
    username = sys.argv[3] if len(sys.argv) > 3 else "opendiffraction_team"
    sys.exit(main(["package", "--dataset", dataset_dir, "--output", pkg_dir, "--username", username]))
