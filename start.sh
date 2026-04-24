#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Starting Multi-Device DNN Collaborative Inference Visualization Platform..."
echo ""
python3 app.py
