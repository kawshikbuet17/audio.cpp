#!/usr/bin/env bash
set -e

# Run from anywhere. The script resolves the audio.cpp repository root.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIOCPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_CMAKE="$AUDIOCPP_ROOT/CMakeLists.txt"
BUILD_DIR="$AUDIOCPP_ROOT/build-py312"

cd "$AUDIOCPP_ROOT"

PYTHON_BIN="$(command -v python3)"

echo "Using Python:"
echo "  executable: $PYTHON_BIN"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install -U pip pybind11 numpy

# Add this binding to the main audio.cpp build only once.
CMAKE_LINE="add_subdirectory(omnivoice_python_cpp_so)"

if ! grep -Fxq "$CMAKE_LINE" "$ROOT_CMAKE"; then
    echo "" >> "$ROOT_CMAKE"
    echo "$CMAKE_LINE" >> "$ROOT_CMAKE"
fi

PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"

echo "Python extension suffix:"
"$PYTHON_BIN" -c \
    'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'

echo "pybind11 CMake directory:"
echo "  $PYBIND11_DIR"

cmake -S . -B "$BUILD_DIR" \
    -DENGINE_ENABLE_CUDA=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build "$BUILD_DIR" \
    --parallel \
    --target omnivoice_cpp audiocpp_cli audiocpp_server

echo
echo "Built module:"
find "$SCRIPT_DIR" "$BUILD_DIR" \
    -name "omnivoice_cpp*.so" \
    -print