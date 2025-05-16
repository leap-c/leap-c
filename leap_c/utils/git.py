import os
from pathlib import Path


def log_git_hash_and_diff(filename: Path):
    """Log the git hash and diff of the current commit to a file."""
    try:
        git_hash = (
            os.popen("git rev-parse HEAD").read().strip()
            if os.path.exists(".git")
            else "No git repository"
        )
        git_diff = (
            os.popen("git diff").read().strip()
            if os.path.exists(".git")
            else "No git repository"
        )

        with open(filename, "w") as f:
            f.write(f"Git hash: {git_hash}\n")
            f.write(f"Git diff:\n{git_diff}\n")
    except Exception as e:
        print(f"Error logging git hash and diff: {e}")
