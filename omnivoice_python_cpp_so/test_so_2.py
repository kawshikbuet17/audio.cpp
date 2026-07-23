#!/usr/bin/env python3

import base64
import io
import time
import wave
from pathlib import Path

import numpy as np

import omnivoice_cpp


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "models/OmniVoice"
REF_AUDIO = "ref_audios/ref_audio_02.wav"

REF_TEXT = (
    "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, "
    "তার কি সর্বোচ্চ সীমা আছে?"
)

TEXTS = [
    "আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?",
    "হ্যালো, আপনাকে স্বাগতম। আমি কিভাবে সহযোগিতা করতে পারি?",
    "অনেক ধন্যবাদ। আপনার দিনটি শুভ হোক।",
]

OUTPUT_DIR = Path("omnivoice_python_cpp_so/outputs/test_so_2")

LANGUAGE = "Bengali"
SEED = 42
STEPS = 20
GUIDANCE_SCALE = 2.0
SPEED = 1.2

SOURCE_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 48000


# =============================================================================
# Test wrapper
# =============================================================================

class OmniVoiceSoTester:
    def __init__(self):
        print("Loading OmniVoice C++ model...")

        start = time.perf_counter()

        self.model = omnivoice_cpp.OmniVoice(
            model_path=MODEL_PATH,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            backend="cuda",
            device=0,
            threads=4,
            generator_weight_type="f16",
            audio_tokenizer_weight_type="f16",
        )

        print(f"Model loaded in {time.perf_counter() - start:.3f} seconds")

    @staticmethod
    def _resample(samples, source_rate, target_rate, channels):
        """Simple NumPy linear resampling for local testing."""
        samples = np.asarray(samples, dtype=np.float32)

        if source_rate == target_rate:
            return samples

        if channels <= 0:
            raise ValueError("channels must be positive")

        if len(samples) % channels != 0:
            raise ValueError("sample count is not divisible by channels")

        frames = samples.reshape(-1, channels)

        if len(frames) == 0:
            return samples

        target_frame_count = max(
            1,
            round(len(frames) * target_rate / source_rate),
        )

        if len(frames) == 1:
            output = np.repeat(frames, target_frame_count, axis=0)
            return output.reshape(-1)

        source_positions = np.arange(len(frames), dtype=np.float64)
        target_positions = np.linspace(
            0,
            len(frames) - 1,
            target_frame_count,
            dtype=np.float64,
        )

        output = np.empty(
            (target_frame_count, channels),
            dtype=np.float32,
        )

        for channel in range(channels):
            output[:, channel] = np.interp(
                target_positions,
                source_positions,
                frames[:, channel],
            )

        return output.reshape(-1)

    @staticmethod
    def _wav_bytes(samples, sample_rate, channels):
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)

        pcm16 = (samples * 32767.0).astype(np.int16)

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        return buffer.getvalue()

    @staticmethod
    def _write_bytes(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _raw_to_model_result(self, raw_result):
        """
        Convert the .so result to the same response shape used by model.py:

        {
            "audio": "<base64 encoded WAV>",
            "duration": 4.25
        }
        """
        source_rate = int(raw_result["sample_rate"])
        channels = int(raw_result["channels"])

        samples = self._resample(
            raw_result["samples"],
            source_rate,
            TARGET_SAMPLE_RATE,
            channels,
        )

        wav_data = self._wav_bytes(
            samples,
            TARGET_SAMPLE_RATE,
            channels,
        )

        frame_count = len(samples) // channels

        return {
            "audio": base64.b64encode(wav_data).decode("utf-8"),
            "duration": frame_count / TARGET_SAMPLE_RATE,
        }

    def _format_result(self, raw_result, return_type, output_path=None):
        """
        return_type:
          raw   -> current .so return format
          model -> model.py-compatible Base64 return format
          both  -> return both formats
        """
        if return_type not in {"raw", "model", "both"}:
            raise ValueError(
                "return_type must be 'raw', 'model', or 'both'"
            )

        model_result = None

        if return_type in {"model", "both"}:
            model_result = self._raw_to_model_result(raw_result)

        if output_path is not None:
            output_path = Path(output_path)

            if return_type == "raw":
                raw_wav = self._wav_bytes(
                    raw_result["samples"],
                    int(raw_result["sample_rate"]),
                    int(raw_result["channels"]),
                )
                self._write_bytes(output_path, raw_wav)

            elif return_type == "model":
                model_wav = base64.b64decode(model_result["audio"])
                self._write_bytes(output_path, model_wav)

            else:
                raw_path = output_path.with_name(
                    f"{output_path.stem}_raw{output_path.suffix}"
                )
                model_path = output_path.with_name(
                    f"{output_path.stem}_model{output_path.suffix}"
                )

                raw_wav = self._wav_bytes(
                    raw_result["samples"],
                    int(raw_result["sample_rate"]),
                    int(raw_result["channels"]),
                )
                model_wav = base64.b64decode(model_result["audio"])

                self._write_bytes(raw_path, raw_wav)
                self._write_bytes(model_path, model_wav)

        if return_type == "raw":
            return raw_result

        if return_type == "model":
            return model_result

        return {
            "raw": raw_result,
            "model": model_result,
        }

    def single_infer(
        self,
        text,
        return_type="model",
        output_path=None,
    ):
        """Generate one text."""
        start = time.perf_counter()

        raw_result = self.model.generate(
            text=text,
            language=LANGUAGE,
            seed=SEED,
            steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            speed=SPEED,
        )

        elapsed = time.perf_counter() - start

        print(
            f"[single] elapsed={elapsed:.3f}s | "
            f"audio={raw_result['duration']:.3f}s"
        )

        return self._format_result(
            raw_result,
            return_type,
            output_path,
        )

    def pseudo_multi_infer(
        self,
        texts,
        return_type="model",
        output_dir=None,
    ):
        """
        Sequential multi inference.

        The same loaded C++ model/session is reused, but each text is generated
        through a separate model.generate() call.
        """
        results = []
        total_start = time.perf_counter()

        for index, text in enumerate(texts, start=1):
            output_path = None

            if output_dir is not None:
                output_path = (
                    Path(output_dir)
                    / f"pseudo_multi_{index:02d}.wav"
                )

            result = self.single_infer(
                text=text,
                return_type=return_type,
                output_path=output_path,
            )
            results.append(result)

        print(
            f"[pseudo-multi] count={len(texts)} | "
            f"elapsed={time.perf_counter() - total_start:.3f}s"
        )

        return results

    def multi_infer(
        self,
        texts,
        return_type="model",
        output_dir=None,
    ):
        """
        True/native multi inference.

        This requires the .so binding to expose:

            model.generate_batch(...)

        The current binding only exposes model.generate(), so this method will
        clearly report that native multi inference is not available yet.
        """
        if not hasattr(self.model, "generate_batch"):
            raise NotImplementedError(
                "Current omnivoice_cpp.so does not expose generate_batch(). "
                "Use pseudo_multi_infer() for sequential generation, or add "
                "native batch support in binding.cpp."
            )

        start = time.perf_counter()

        raw_results = self.model.generate_batch(
            texts=texts,
            language=LANGUAGE,
            seed=SEED,
            steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            speed=SPEED,
        )

        results = []

        for index, raw_result in enumerate(raw_results, start=1):
            output_path = None

            if output_dir is not None:
                output_path = (
                    Path(output_dir)
                    / f"multi_{index:02d}.wav"
                )

            results.append(
                self._format_result(
                    raw_result,
                    return_type,
                    output_path,
                )
            )

        print(
            f"[multi] count={len(texts)} | "
            f"elapsed={time.perf_counter() - start:.3f}s"
        )

        return results


# =============================================================================
# Tests
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tester = OmniVoiceSoTester()

    print("\n=== Single inference: both return formats ===")

    single_result = tester.single_infer(
        text=TEXTS[0],
        return_type="both",
        output_path=OUTPUT_DIR / "single.wav",
    )

    print("Raw result keys:", list(single_result["raw"].keys()))
    print("Model result keys:", list(single_result["model"].keys()))
    print(
        "Model Base64 length:",
        len(single_result["model"]["audio"]),
    )

    print("\n=== Pseudo multi inference: sequential ===")

    pseudo_results = tester.pseudo_multi_infer(
        texts=TEXTS,
        return_type="model",
        output_dir=OUTPUT_DIR / "pseudo_multi",
    )

    print(f"Pseudo multi returned {len(pseudo_results)} result(s)")

    print("\n=== Native multi inference ===")

    try:
        multi_results = tester.multi_infer(
            texts=TEXTS,
            return_type="model",
            output_dir=OUTPUT_DIR / "multi",
        )
        print(f"Native multi returned {len(multi_results)} result(s)")

    except NotImplementedError as error:
        print(f"SKIPPED: {error}")

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()