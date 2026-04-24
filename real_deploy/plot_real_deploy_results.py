"""
Regenerate figures and Markdown from the last saved JSON (or a given JSON path).

  python real_deploy/plot_real_deploy_results.py
  python real_deploy/plot_real_deploy_results.py figures/real_deploy_last.json

Same as: python -m real_deploy.thesis_export
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_deploy.thesis_export import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
