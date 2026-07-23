#!/usr/bin/env bash
set -e

DEVICE=0

# Run from the audio.cpp repository root.
# Keep this terminal open.

CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_server \
    --config omnivoice_python_cpp_simple/server.json
