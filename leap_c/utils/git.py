import subprocess
from pathlib import Path


def log_git_hash_and_diff(filename: Path, repo_path: Path = Path(".")) -> None:
    """Log the git hash and diff of the current commit to a file.

    Args:
        filename: File to write the log.
        repo_path: Path to the git repository (default: current dir).
    """
    try:
        git_dir = repo_path / ".git"
        if git_dir.exists():
            git_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            git_diff = subprocess.run(
                ["git", "diff"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        else:
            git_hash = "No git repository"
            git_diff = "No git repository"

        with open(filename, "w") as f:
            f.write(f"Git hash: {git_hash}\n")
            f.write(f"Git diff:\n{git_diff}\n")

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
    except Exception as e:
        print(f"Error logging git hash and diff: {e}")
