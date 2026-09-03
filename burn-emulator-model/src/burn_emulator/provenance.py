import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _pkg_root() -> Path:
    return Path(__file__).resolve().parent


def architecture_file(class_path: str, pkg_root: Path | None = None) -> Path:
    root = Path(pkg_root) if pkg_root is not None else _pkg_root()
    parts = class_path.split(".")[:-1]  # drop the class name
    if parts and parts[0] == root.name:
        parts = parts[1:]
    base = root.joinpath(*parts)
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def model_code_sha(class_path: str, pkg_root: Path | None = None) -> str:
    return hashlib.sha256(architecture_file(class_path, pkg_root).read_bytes()).hexdigest()


def git_head(repo: Path | None = None) -> tuple[str | None, bool]:
    root = str(Path(repo) if repo is not None else _pkg_root())
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            != ""
        )
        return (sha or None), dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


if __name__ == "__main__":
    sha, dirty = git_head()
    meta = {"model_repo_sha": sha, "model_repo_dirty": dirty}
    if len(sys.argv) > 1:  # optional dotted class_path to fingerprint
        meta["model_class_path"] = sys.argv[1]
        meta["model_code_sha256"] = model_code_sha(sys.argv[1])
    print(json.dumps(meta, indent=2))
