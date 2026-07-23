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

There is no server and no subprocess in `test_so.py` or `test_so_2.py`.

## Files

```text
binding.cpp      pybind11 C++ binding
CMakeLists.txt   Builds the Python extension
build.sh         Builds omnivoice_cpp*.so
run_cpp.sh       Direct C++ CLI baseline
test_so.py       Imports the .so and runs single inference
test_so_2.py     Single and pseudo-multi inference test
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

## 5. Extended test through the `.so`

```bash
python3 omnivoice_python_cpp_so/test_so_2.py
```

`test_so_2.py` includes:

```python
single_infer(...)
pseudo_multi_infer(...)
multi_infer(...)
```

### Single inference

Generates one audio using the loaded `.so` model:

```python
result = tester.single_infer(
    text=TEXTS[0],
    return_type="both",
)
```

### Pseudo-multi inference

Generates multiple texts sequentially while reusing the same loaded model:

```python
results = tester.pseudo_multi_infer(
    texts=TEXTS,
    return_type="model",
)
```

The flow is:

```text
Load model once
  → generate text 1
  → generate text 2
  → generate text 3
```

This is sequential generation, not native batch inference.

### Multi inference

`multi_infer()` is prepared for native batch support:

```python
results = tester.multi_infer(
    texts=TEXTS,
)
```

The current `.so` does not expose `generate_batch()`, so this test is skipped until native batch support is added to `binding.cpp`.

### Return types

`test_so_2.py` supports three return types.

#### Raw

```python
return_type="raw"
```

Returns the direct `.so` result:

```python
{
    "samples": numpy_float32_array,
    "sample_rate": 24000,
    "channels": 1,
    "duration": 4.2
}
```

#### Model-compatible

```python
return_type="model"
```

Returns the same format used by the Triton `model.py` (according to a voicebot project):

```python
{
    "audio": "base64_encoded_wav",
    "duration": 4.2
}
```

The audio is converted to PCM16 WAV and resampled to 48 kHz.

#### Both

```python
return_type="both"
```

Returns both formats:

```python
{
    "raw": {
        "samples": numpy_float32_array,
        "sample_rate": 24000,
        "channels": 1,
        "duration": 4.2
    },
    "model": {
        "audio": "base64_encoded_wav",
        "duration": 4.2
    }
}
```

Outputs are written under:

```text
omnivoice_python_cpp_so/outputs/test_so_2/
```

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

For the extended `.so` test, edit `test_so_2.py`:

```python
LANGUAGE = "Bengali"
SEED = 42
STEPS = 20
GUIDANCE_SCALE = 2.0
SPEED = 1.2
```

For the direct C++ baseline, edit `run_cpp.sh`.

## Important build note

The extension must be built with the same Python version that imports it.

For example, a Python 3.10 extension generally cannot be imported by Python 3.11.

For Triton later, build the `.so` inside the same Triton Python-backend environment or stub/container version that will load it.
