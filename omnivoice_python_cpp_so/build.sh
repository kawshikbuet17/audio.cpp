#!/usr/bin/env bash
set -e

# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIOCPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_CMAKE="$AUDIOCPP_ROOT/CMakeLists.txt"

cd "$AUDIOCPP_ROOT"

python3 -m pip install -U pybind11 numpy

# Add this binding to the main audio.cpp build only once.
CMAKE_LINE="add_subdirectory(omnivoice_python_cpp_so)"

if ! grep -Fxq "$CMAKE_LINE" "$ROOT_CMAKE"; then
    echo "" >> "$ROOT_CMAKE"
    echo "$CMAKE_LINE" >> "$ROOT_CMAKE"
fi

PYBIND11_DIR="$(python3 -m pybind11 --cmakedir)"

# Reconfigure the existing CUDA build.
cmake -S . -B build \
    -DENGINE_ENABLE_CUDA=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dpybind11_DIR="$PYBIND11_DIR"

# Build only the Python extension and required dependencies.
cmake --build build \
    --parallel \
    --target omnivoice_cpp audiocpp_cli audiocpp_server

echo
echo "Built module:"
find "$SCRIPT_DIR" -maxdepth 1 -name "omnivoice_cpp*.so" -print
