# Simple OmniVoice Python `.so` Test

Keep this folder directly inside the `audio.cpp` repository:

```text
audio.cpp/
├── CMakeLists.txt
├── build/
├── models/OmniVoice/
├── ref_audios/ref_audio_02.wav
└── omnivoice_python_cpp_so/
```

The flow is:

```text
Python
  → import omnivoice_cpp.so
  → audio.cpp C++ model/session
  → NumPy float32 audio
```

There is no server and no subprocess in `test_so.py`.

## Files

```text
binding.cpp      pybind11 C++ binding
CMakeLists.txt   Builds the Python extension
build.sh         Builds omnivoice_cpp*.so
run_cpp.sh       Direct C++ CLI baseline
test_so.py       Imports the .so and runs inference
outputs/         Generated WAV files
```

## 1. Extract

Place the folder inside `audio.cpp` with this exact name:

```text
omnivoice_python_cpp_so
```

## 2. Build the `.so`

```bash
chmod +x omnivoice_python_cpp_so/*.sh

./omnivoice_python_cpp_so/build.sh
```

The output will look similar to:

```text
omnivoice_python_cpp_so/omnivoice_cpp.cpython-310-x86_64-linux-gnu.so
```

`build.sh` adds this line to the end of the main `CMakeLists.txt`:

```cmake
add_subdirectory(omnivoice_python_cpp_so)
```

It adds the line only once.

## 3. Test direct C++

```bash
./omnivoice_python_cpp_so/run_cpp.sh
```

Output:

```text
omnivoice_python_cpp_so/outputs/cpp_output.wav
```

## 4. Test through the `.so`

```bash
python3 omnivoice_python_cpp_so/test_so.py
```

Output:

```text
omnivoice_python_cpp_so/outputs/python_so_output.wav
```

The model is initialized here:

```python
model = omnivoice_cpp.OmniVoice(...)
```

It is reused here:

```python
result = model.generate(...)
```

Calling `model.generate()` again does not reload the model.

## Returned result

```python
{
    "samples": numpy_float32_array,
    "sample_rate": 24000,
    "channels": 1,
    "duration": 4.2
}
```

The samples are interleaved when `channels > 1`.

## Change settings

For the `.so` test, edit `test_so.py`:

```python
generator_weight_type="f16"
audio_tokenizer_weight_type="f16"
steps=20
guidance_scale=2.0
speed=1.2
```

For the direct C++ baseline, edit `run_cpp.sh`.

## Important build note

The extension must be built with the same Python version that imports it.

For example, a Python 3.10 extension generally cannot be imported by Python 3.11.

For Triton later, build the `.so` inside the same Triton Python-backend environment or stub/container version that will load it.
