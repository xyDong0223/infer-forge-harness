"""Location of versioned repository resources, separate from runtime storage."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
