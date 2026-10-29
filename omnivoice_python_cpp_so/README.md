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

### CUDA architecture

`build.sh` sets:

```
-DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;89-virtual"
```

One `.so` then runs on the Tesla T4 (`75`), RTX 3090 (`86`), RTX 4090 (`89`) and RTX 5090:

- `XX-real` is machine code for that GPU, so it starts without a JIT step and uses ggml's fast paths.
- `89-virtual` is PTX. GPUs newer than the 4090, such as the RTX 5090, JIT-compile it at first load. The driver caches the result in `~/.nv/ComputeCache`; in Docker, mount that folder and set `CUDA_CACHE_MAXSIZE=2147483648` so the JIT runs only once.
- A GPU older than the T4 fails with "no kernel image".

Without this setting, the root `CMakeLists.txt` picks a portable list for the installed CUDA version (on CUDA 12.6: `50-virtual;61-virtual;70-virtual;75-real;80-virtual;86-real;89-real;90-virtual`). It runs the same on the GPUs above, but the build is about twice as heavy and the `.so` is bigger. Older audio.cpp code (before upstream #280) had no such list and built for nvcc's default, `52`.

To target other GPUs, change the list:

| GPU | Value |
|---|---|
| Tesla T4 | `75` |
| RTX 3090 | `86` |
| RTX 4090 | `89` |
| RTX 5090 | `120` |
| Several GPUs | semicolon-separated list, e.g. `"75;89"` |

`120` (RTX 5090) needs CUDA 12.8 or newer. This machine has CUDA 12.6, which cannot build for it, so the 5090 uses the `89-virtual` PTX. With CUDA 12.8+, add `120-real`.

`build.sh` runs `cmake --build` with `--parallel "${JOBS:-16}"`. Each CUDA compile job can take several GB of RAM, so on a shared machine lower it, e.g. `JOBS=8 ./omnivoice_python_cpp_so/build.sh`. Never use `--parallel` without a number: with Makefiles it starts unlimited jobs and can freeze the machine.

The build type is `RelWithDebInfo`, which adds about 1 GB of debug info. `build.sh` removes it with `strip --strip-debug`; the code is unchanged.

To find a GPU's compute capability:

```bash
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
```

After changing the value, rebuild from scratch:

```bash
rm -rf build-py312
./omnivoice_python_cpp_so/build.sh
```

To check the built `.so`, list its GPU code. It should show `sm_75`, `sm_86` and `sm_89` machine code plus `sm_89` PTX:

```bash
cuobjdump --list-elf omnivoice_python_cpp_so/omnivoice_cpp*.so | grep -o 'sm_[0-9]*' | sort | uniq -c
cuobjdump --list-ptx omnivoice_python_cpp_so/omnivoice_cpp*.so | grep -o 'sm_[0-9]*' | sort | uniq -c
```

To make sure the RTX 4090 (GPU 0 on this machine) is used at runtime:

```bash
CUDA_VISIBLE_DEVICES=0 python3 omnivoice_python_cpp_so/test_so.py
```

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
