"""
Deprecated wrapper: output now goes to figures/final_figures/
Run: python generate_all_thesis_figures.py
"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "generate_all_thesis_figures.py")], cwd=root, check=False)
