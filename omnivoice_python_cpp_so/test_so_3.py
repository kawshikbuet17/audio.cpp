#!/usr/bin/env python3
"""
A/B test: Python OmniVoice vs audio.cpp OmniVoice (.so) on the same inputs.

Variants:
  python       Same call as OmniVoice/test_omnivoice_full_benchmark.sh
               (OmniVoice.from_pretrained + model.generate, clone mode,
               no language, fp16).
  cpp_triton   Same call as the Triton model_repository_test_cpp run_test.sh
               (omnivoice_cpp.OmniVoice, language="bn", target text NFC
               normalized by fix_pronunciation, reference text raw).
  cpp_matched  omnivoice_cpp with the inputs the Python tokenizer actually
               sees: NFC on both target and reference text, no language.

The Python tokenizer applies NFC (tokenizer.json "normalizer"); the C++
tokenizer does not. Comparing cpp_triton and cpp_matched shows the effect.

The two engines need different Python versions (omnivoice lives in the
`omnivoice` conda env, Python 3.10; the .so is built for Python 3.12),
so each variant runs in its own subprocess.

Run from the audio.cpp repository root:
    python3 omnivoice_python_cpp_so/test_so_3.py
    python3 omnivoice_python_cpp_so/test_so_3.py --variants python cpp_matched --runs 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    "/home/ipt/kawshik/tts-testing/OmniVoice/models/"
    "OmniVoice-LoRA-31-Aug-2026"
)
REF_AUDIO = "ref_audios/ref_audio_03.wav"
REF_TEXT_FILE = "ref_audios/ref_text_03.txt"

PYTHON_BIN = "/root/miniconda3/envs/omnivoice/bin/python"
CPP_PYTHON_BIN = "/root/miniconda3/bin/python3"

OUTPUT_DIR = "omnivoice_python_cpp_so/outputs/test_so_3"

ALL_VARIANTS = ["python", "cpp_triton", "cpp_matched"]

WARMUP_TEXT = "আমি বাংলায় কথা বলছি।"

# Copied byte-for-byte from OmniVoice/test_omnivoice_full_benchmark.py.
TEXTS = [
    {
        "label": "bn_very_short_01",
        "text": "অথবা আপনি সরাসরি আমাদের কাস্টমার কেয়ার নাম্বার শূন্য নয় ছয় এক এক এক দুই তিন এক দুই তিন নম্বরে কল দিয়েও জানতে পারেন।",
    },
    {
        "label": "bn_very_short_02",
        "text": "যদি নম্বরটি বিকাশ অ্যাকাউন্ট না হয়, তবে আপনি অ্যাপ থেকে সেন্ড মানি অপশনে গিয়ে লেনদেনটি বাতিল করতে পারবেন।",
    },
    {
        "label": "bn_very_short_03",
        "text": "আপনি প্রথমে প্রাপকের বিকাশ নম্বরটি নিশ্চিত করবেন, তারপর রেমিট্যান্স পাঠানোর মাধ্যমটি নির্বাচন করবেন এবং সবশেষে প্রয়োজনীয় তথ্য দিয়ে লেনদেনটি সম্পন্ন করবেন।",
    },
]


def clean_unicode(text):
    # Same as clean_unicode() in the Triton fix_pronunciation.py.
    text = unicodedata.normalize("NFC", text)
    return text.replace("‌", "").replace("‍", "")


def save_wav(path, samples, sample_rate, channels=1):
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)

    pcm16 = (samples * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
#  Workers (run inside the subprocess for one variant)
# ─────────────────────────────────────────────────────────────────────────────


def load_python_engine(args, ref_text):
    import torch
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.float16,
    )

    def generate(text, seed):
        # Same as test_omnivoice_full_benchmark.py: concat + normalize.
        torch.manual_seed(seed)

        with torch.inference_mode():
            audio_obj = model.generate(
                text=text,
                num_step=args.num_step,
                guidance_scale=args.guidance_scale,
                speed=args.speed,
                ref_audio=args.ref_audio,
                ref_text=ref_text,
            )

        torch.cuda.synchronize()

        if isinstance(audio_obj, (list, tuple)):
            audio = np.concatenate(
                [np.squeeze(np.asarray(a, dtype=np.float32)) for a in audio_obj]
            )
        else:
            audio = np.squeeze(np.asarray(audio_obj, dtype=np.float32))

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        return audio, 24000

    return generate


def load_cpp_engine(args, ref_text, language):
    sys.path.insert(0, str(SCRIPT_DIR))
    import omnivoice_cpp

    model = omnivoice_cpp.OmniVoice(
        model_path=args.model,
        ref_audio=args.ref_audio,
        ref_text=ref_text,
        backend="cuda",
        device=0,
        threads=4,
        generator_weight_type="f16",
        audio_tokenizer_weight_type="f16",
    )

    def generate(text, seed):
        result = model.generate(
            text=text,
            language=language,
            seed=seed,
            steps=args.num_step,
            guidance_scale=args.guidance_scale,
            speed=args.speed,
        )
        return np.asarray(result["samples"], dtype=np.float32), int(
            result["sample_rate"]
        )

    return generate


def run_worker(args):
    variant = args.worker
    raw_ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()

    if variant == "python":
        # Python tokenizer applies NFC itself, so pass text as-is.
        prepare_text = lambda text: text
        generate = load_python_engine(args, raw_ref_text)

    elif variant == "cpp_triton":
        # Triton: fix_pronunciation() NFC-normalizes the target text only.
        prepare_text = clean_unicode
        generate = load_cpp_engine(args, raw_ref_text, "bn")

    elif variant == "cpp_matched":
        prepare_text = clean_unicode
        generate = load_cpp_engine(args, clean_unicode(raw_ref_text), "none")

    else:
        raise ValueError(f"Unknown variant: {variant}")

    out_dir = Path(args.output_dir) / variant

    generate(prepare_text(WARMUP_TEXT), args.seed)

    results = []

    for item in TEXTS:
        for run in range(args.runs):
            seed = args.seed + run

            start = time.perf_counter()
            audio, sample_rate = generate(prepare_text(item["text"]), seed)
            elapsed = time.perf_counter() - start

            path = out_dir / f"{item['label']}_seed{seed}.wav"
            save_wav(path, audio, sample_rate)

            duration = len(audio) / sample_rate
            results.append(
                {
                    "label": item["label"],
                    "seed": seed,
                    "duration": duration,
                    "elapsed": elapsed,
                    "path": str(path),
                }
            )

            print(
                f"[{variant}] {item['label']} seed={seed} "
                f"audio={duration:.2f}s gen={elapsed:.2f}s",
                flush=True,
            )

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────


def run_variant(args, variant):
    python_bin = args.python_bin if variant == "python" else args.cpp_python_bin

    command = [
        python_bin,
        str(Path(__file__).resolve()),
        "--worker", variant,
        "--model", args.model,
        "--ref-audio", args.ref_audio,
        "--ref-text-file", args.ref_text_file,
        "--num-step", str(args.num_step),
        "--guidance-scale", str(args.guidance_scale),
        "--speed", str(args.speed),
        "--seed", str(args.seed),
        "--runs", str(args.runs),
        "--output-dir", args.output_dir,
    ]

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"\n=== {variant} ({python_bin}) ===", flush=True)
    subprocess.run(command, env=env, check=True)

    results_path = Path(args.output_dir) / variant / "results.json"
    return json.loads(results_path.read_text(encoding="utf-8"))


def print_summary(all_results):
    variants = list(all_results)
    keys = [
        (r["label"], r["seed"])
        for r in all_results[variants[0]]
    ]

    lookup = {
        variant: {(r["label"], r["seed"]): r for r in results}
        for variant, results in all_results.items()
    }

    print("\nAudio duration in seconds (shorter = faster speech):\n")
    header = f"{'text':<18} {'seed':>4}  " + "  ".join(
        f"{v:>12}" for v in variants
    )
    print(header)
    print("-" * len(header))

    for key in keys:
        row = f"{key[0]:<18} {key[1]:>4}  " + "  ".join(
            f"{lookup[v][key]['duration']:>12.2f}" for v in variants
        )
        print(row)

    print("\nAverage generation time in seconds:\n")
    for variant in variants:
        times = [r["elapsed"] for r in all_results[variant]]
        print(f"  {variant:<12} {sum(times) / len(times):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--ref-audio", default=REF_AUDIO)
    parser.add_argument("--ref-text-file", default=REF_TEXT_FILE)
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per text, seeds seed..seed+runs-1")
    parser.add_argument("--gpu", default="0",
                        help="nvidia-smi index; 0 = RTX 4090, 1 = Tesla T4")
    parser.add_argument("--python-bin", default=PYTHON_BIN)
    parser.add_argument("--cpp-python-bin", default=CPP_PYTHON_BIN)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    all_results = {
        variant: run_variant(args, variant)
        for variant in args.variants
    }

    print_summary(all_results)
    print(f"\nWAV files: {args.output_dir}/<variant>/")


if __name__ == "__main__":
    main()
