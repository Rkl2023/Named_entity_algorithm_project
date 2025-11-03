#!/usr/bin/env python3
"""Setup helper for the Scientific Named Entity Explorer.

- Creates a virtual environment at .venv (if missing)
- Installs pinned dependencies from requirements.txt
- Verifies Streamlit installation
- Prints launch instructions
"""

from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command and stream its output."""
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    venv_path = project_root / ".venv"
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        raise SystemExit("requirements.txt not found. Please keep install.py in the project root.")

    if not venv_path.exists():
        print("Creating virtual environment at .venv ...")
        run([sys.executable, "-m", "venv", str(venv_path)])
    else:
        print("Virtual environment already exists at .venv")

    if platform.system() == "Windows":
        venv_python = venv_path / "Scripts" / "python.exe"
        activate_hint = ".venv\\Scripts\\activate"
    else:
        venv_python = venv_path / "bin" / "python"
        activate_hint = "source .venv/bin/activate"

    if not venv_python.exists():
        raise SystemExit(f"Could not locate python in virtual environment: {venv_python}")

    print("Upgrading pip ...")
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing dependencies from requirements.txt ...")
    run([str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)])

    print("Verifying Streamlit installation ...")
    run([str(venv_python), "-m", "streamlit", "--version"])

    print("\nSetup complete ✅")
    print("Next steps:")
    print(f"  1. Activate the virtual environment: {activate_hint}")
    print("  2. Launch the app: streamlit run app.py")
    print("\nEnjoy exploring your scientific abstracts!")


if __name__ == "__main__":
    main()
