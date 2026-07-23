# Simple OmniVoice C++ and Python Test

Keep this folder directly inside the `audio.cpp` repository:

```text
audio.cpp/
├── build/
├── models/OmniVoice/
├── ref_audios/ref_audio_02.wav
└── omnivoice_python_cpp_simple/
```

Run all commands from the `audio.cpp` repository root.

## Build

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON

cmake --build build \
  --parallel \
  --target audiocpp_cli audiocpp_server
```

## Test direct C++

```bash
chmod +x omnivoice_python_cpp_simple/*.sh
./omnivoice_python_cpp_simple/run_cpp.sh
```

Output:

```text
omnivoice_python_cpp_simple_simple/outputs/cpp_output.wav
```

## Test from Python

Terminal 1:

```bash
./omnivoice_python_cpp_simple/run_server.sh
```

Wait until the server is ready.

Terminal 2:

```bash
python3 omnivoice_python_cpp_simple/test_python.py
```

Output:

```text
omnivoice_python_cpp_simple/outputs/python_output.wav
```

Press `Ctrl+C` in terminal 1 when finished.

## Notes

- `run_cpp.sh` uses `audiocpp_cli` directly.
- `test_python.py` sends an HTTP request to `audiocpp_server`.
- In both cases, model inference runs in C++.
- The Python file uses only the standard library.
- Change weight types for direct C++ in `run_cpp.sh`.
- Change weight types for Python/server testing in `server.json`, then restart the server.
