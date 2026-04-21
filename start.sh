#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── One-time setup ─────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Checking system dependencies..."

    # C-extension packages must come from pacman, not pip, on MSYS2
    if ! python3.12 -c "import numpy" 2>/dev/null; then
        echo "Installing numpy (pacman)..."
        pacman -S --noconfirm --needed mingw-w64-x86_64-python-numpy
    fi
    if ! python3.12 -c "import pandas" 2>/dev/null; then
        echo "Installing pandas (pacman)..."
        pacman -S --noconfirm --needed mingw-w64-x86_64-python-pandas
    fi

    echo "Creating virtual environment..."
    # --system-site-packages lets the venv see pacman-installed numpy/pandas
    python3.12 -m venv --system-site-packages .venv
fi

source .venv/bin/activate

echo "Building dragonsci..."
maturin develop

echo ""
echo "Launching demo..."
python demo.py
