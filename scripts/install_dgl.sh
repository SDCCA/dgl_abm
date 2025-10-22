#!/usr/bin/env bash
set -o pipefail

ENV_NAME="${1:-dgl_abm_mac}"

echo "Installing DGL"

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo "macOS arm64 detected — attempting wheel installation of pytorch and dgl"
        conda install -y -n "$ENV_NAME" -c pytorch -c conda-forge "pytorch==2.0.0" || true
        if ! conda run -n "$ENV_NAME" --no-capture-output pip install "dgl==2.0.0" -f https://data.dgl.ai/wheels/repo.html; then
            echo "Warning: DGL pip wheel not available for this platform/Python, automatic installation of DGL-ABM cannot continue. You can try downloading a wheel for your Python+PyTorch and installing it manually."
            exit 1
        fi
    else
        echo "install_dgl.sh triggered but ARCH=$ARCH detected"
    fi
else
    echo "install_dgl.sh triggered but OS=$OS detected"
fi

