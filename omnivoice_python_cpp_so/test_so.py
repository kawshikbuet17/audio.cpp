#!/usr/bin/env python3

import time
import wave
from pathlib import Path

import numpy as np

import omnivoice_cpp


MODEL_PATH = "models/OmniVoice"
REF_AUDIO = "ref_audios/ref_audio_02.wav"

REF_TEXT = (
    "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, "
    "তার কি সর্বোচ্চ সীমা আছে?"
)

TEXT = (
    "আমি একটা টিটিএস মডেল, নাম OmniVoice। "
    "কিভাবে আপনাকে সাহায্য করতে পারি?"
)

OUTPUT = Path(
    "omnivoice_python_cpp_so/outputs/python_so_output.wav"
)


def save_wav(path, samples, sample_rate, channels):
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)

    pcm16 = (samples * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


print("Loading OmniVoice C++ model...")

load_start = time.perf_counter()

model = omnivoice_cpp.OmniVoice(
    model_path=MODEL_PATH,
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    backend="cuda",
    device=0,
    threads=4,
    generator_weight_type="f16",
    audio_tokenizer_weight_type="f16",
)

load_time = time.perf_counter() - load_start

print(f"Model loaded in {load_time:.3f} seconds")
print("Running inference...")

infer_start = time.perf_counter()

result = model.generate(
    text=TEXT,
    language="Bengali",
    seed=42,
    steps=20,
    guidance_scale=2.0,
    speed=1.2,
)

infer_time = time.perf_counter() - infer_start

save_wav(
    OUTPUT,
    result["samples"],
    result["sample_rate"],
    result["channels"],
)

print(f"Inference time: {infer_time:.3f} seconds")
print(f"Audio duration: {result['duration']:.3f} seconds")
print(f"Sample rate: {result['sample_rate']}")
print(f"Channels: {result['channels']}")
print(f"Output: {OUTPUT}")

# The same model object can be reused:
#
# second_result = model.generate(
#     text="এটি দ্বিতীয় অনুরোধ।",
#     language="Bengali",
#     steps=20,
# )
