#!/usr/bin/env python3
"""
Batch-size and multiple-instance benchmark: Python OmniVoice vs audio.cpp
OmniVoice (.so).

Phase 1, batch sizes: for each text size (5/10/15/20 words) and batch size
(1..20), the same text is repeated batch-size times and generated as one
batch. Each point is run --repeats times and the median wall time is
reported.

Phase 2, multiple instances: N engine processes on the same GPU, each with
its own model copy (like Triton instance_group count=N), get N concurrent
requests, one each. N grows until an instance no longer fits.

Engines:
  python           OmniVoice.from_pretrained(dtype=fp16), native batching:
                   model.generate(text=[text] * batch) runs the batch in one pass.
  python_fi        python + apply_flashinfer(model) (OmniVoice commit 28bc088).
  python_fi_graph  python_fi + CUDA graphs (enable_cuda_graph=True).
  cpp              omnivoice_cpp.OmniVoice(generator f16, audio tokenizer f16).
                   The binding has no batch API, so a batch of N is N sequential
                   generate() calls (what the Triton model.py does today).

Both engines encode the reference once and reuse it (Python through
create_voice_clone_prompt, C++ through its reference prompt cache), so the
timings are generation only.

The engines need different Python versions (omnivoice lives in the
`omnivoice` conda env, Python 3.10; the .so is built for Python 3.12),
so each engine runs in its own subprocess.

Live output (appended / rewritten after every measured point):
  <output-dir>/benchmark_batch.log
  <output-dir>/benchmark_batch_summary.md

Run from the audio.cpp repository root. To stop the other services on the
RTX 4090 during the run and restart them afterwards, use
run_benchmark_batch.sh instead of calling this script directly.

    python3 omnivoice_python_cpp_so/benchmark_batch.py
    python3 omnivoice_python_cpp_so/benchmark_batch.py --batch-sizes 1-4 --word-counts 5 --repeats 1
"""

import argparse
import json
import multiprocessing
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from datetime import datetime
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

OUTPUT_DIR = "omnivoice_python_cpp_so/outputs/benchmark_batch"
LOG_NAME = "benchmark_batch.log"
SUMMARY_NAME = "benchmark_batch_summary.md"
RESULTS_NAME = "results.jsonl"
INSTANCES_NAME = "instances.jsonl"

# The untimed utilization run is repeated until it lasts this long.
UTIL_MIN_SECONDS = 1.5

ALL_ENGINES = ["python", "python_fi", "python_fi_graph", "cpp"]
ENGINE_NAMES = {
    "python": "Python",
    "python_fi": "Python + FlashInfer",
    "python_fi_graph": "Python + FlashInfer + CUDA graphs",
    "cpp": "C++",
}
ENGINE_DESCRIPTIONS = {
    "python": "OmniVoice fp16, native batching, `generate(text=[text] * batch)`: the "
    "generator runs the whole batch together; audio decode and post-processing still "
    "loop per item. No language.",
    "python_fi": "same as Python, plus `apply_flashinfer(model)` (OmniVoice commit "
    "28bc088): cond/uncond packed into one row with FlashInfer ragged attention, fused "
    "RMSNorm/RoPE/QKV/MLP kernels, KV cache off.",
    "python_fi_graph": "same as Python + FlashInfer, plus `enable_cuda_graph=True`: one "
    "CUDA graph per packed shape, replayed for every step. A new shape (batch size or "
    "text length) is captured on first use; the per-point warmup does that here, and the "
    "graph cache is cleared before each point.",
    "cpp": "`.so` generator f16, audio tokenizer f16 (codebook f32, the binding default), "
    "`language=\"bn\"`, NFC-normalized text. **No batch API: a batch of N is N "
    "sequential calls.**",
}

# Exactly 5, 10, 15 and 20 whitespace-separated words (checked at startup).
TEXTS = {
    5: "আপনার লেনদেন সফলভাবে সম্পন্ন হয়েছে।",
    10: "আপনার বিকাশ অ্যাকাউন্টে পাঁচশো টাকা জমা হয়েছে, ব্যালেন্স দেখে নিন।",
    15: (
        "যদি নম্বরটি বিকাশ অ্যাকাউন্ট না হয়, তবে আপনি অ্যাপ থেকে "
        "লেনদেনটি সহজেই বাতিল করতে পারবেন।"
    ),
    20: (
        "আপনি প্রথমে প্রাপকের বিকাশ নম্বরটি নিশ্চিত করবেন, তারপর টাকার "
        "পরিমাণ লিখে পিন দিয়ে লেনদেনটি সম্পন্ন করবেন এবং কনফার্মেশন "
        "মেসেজ দেখবেন।"
    ),
}


def clean_unicode(text):
    # Same as clean_unicode() in the Triton fix_pronunciation.py. The C++
    # tokenizer does not apply the tokenizer.json NFC normalizer itself.
    text = unicodedata.normalize("NFC", text)
    return text.replace("‌", "").replace("‍", "")


def parse_batch_sizes(spec):
    sizes = []
    for part in spec.split(","):
        if "-" in part:
            low, high = part.split("-")
            sizes.extend(range(int(low), int(high) + 1))
        else:
            sizes.append(int(part))
    return sorted(set(sizes))


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(args, source, message):
    line = f"[{now()}] [{source}] {message}"
    print(line, flush=True)
    with open(Path(args.output_dir) / LOG_NAME, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_wav(path, samples, sample_rate):
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def gpu_memory_mb():
    # Memory nvidia-smi reports for this process. For Python this includes
    # PyTorch's allocator cache, so it is the peak reserved so far, not the
    # memory a single batch needs (see peak_allocated_mb for that).
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None

    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            return int(parts[1])
    return None


class GpuUtilSampler:
    """Samples GPU utilization (%) with nvidia-smi every interval_ms.

    nvidia-smi's utilization.gpu is the share of time at least one kernel
    was running, so 100% means busy, not that every SM is saturated.
    """

    def __init__(self, gpu, interval_ms=100):
        self.gpu = gpu
        self.interval_ms = interval_ms
        self.samples = []

    def __enter__(self):
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "-i", self.gpu,
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits",
                 "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.process = None
            return self

        def read():
            for line in self.process.stdout:
                line = line.strip()
                if line.isdigit():
                    self.samples.append(int(line))

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

        # nvidia-smi startup (NVML init) is CPU- and driver-heavy and slowed
        # the first timed repeat at small, launch-bound batches. Wait for the
        # first sample so startup is over before timing starts, then drop it
        # (it was taken while the GPU was idle).
        deadline = time.monotonic() + 10
        while not self.samples and time.monotonic() < deadline:
            time.sleep(0.02)
        self.samples.clear()
        return self

    def __exit__(self, *exc):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.thread.join(timeout=5)

    def mean(self):
        return round(statistics.mean(self.samples)) if self.samples else None

    def max(self):
        return max(self.samples) if self.samples else None


def parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(","):
        if "-" in part:
            low, high = part.split("-")
            cpus.update(range(int(low), int(high) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def gpu_numa_cpus(gpu):
    """(numa_node, cpus) of the socket the GPU is attached to, or None.

    At small batches the run is bound by the CPU launching GPU kernels, so a
    worker scheduled on the other socket is measurably slower (seen: 0.95 s
    vs 1.19 s for the same point). Pinning keeps runs comparable.
    """
    try:
        bus_id = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        # nvidia-smi prints 00000000:03:00.0, sysfs uses 0000:03:00.0.
        domain, rest = bus_id.split(":", 1)
        device = Path(f"/sys/bus/pci/devices/{domain[-4:]}:{rest}".lower())
        node = int((device / "numa_node").read_text())
        if node < 0:
            return None
        cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return node, parse_cpulist(cpulist)


# ─────────────────────────────────────────────────────────────────────────────
#  Engines. Each returns a dict with:
#    generate_batch(texts, seed) -> list of (audio, sample_rate)
#    warmup_point(text, batch)   untimed run before each measured point
#    reset_peak()                called before each point
#    peak_allocated_mb()         peak memory used by the point, or None
#    is_oom(error), free_memory()
# ─────────────────────────────────────────────────────────────────────────────


def load_python_engine(args, ref_text, flashinfer=None):
    import torch
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.float16,
    )

    if flashinfer is not None:
        from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

        apply_flashinfer(model, enable_cuda_graph=(flashinfer == "graph"))

    prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=ref_text,
    )

    def generate_batch(texts, seed):
        torch.manual_seed(seed)

        with torch.inference_mode():
            audios = model.generate(
                text=texts,
                voice_clone_prompt=prompt,
                num_step=args.num_step,
                guidance_scale=args.guidance_scale,
                speed=args.speed,
            )

        torch.cuda.synchronize()

        return [
            (np.squeeze(np.asarray(audio, dtype=np.float32)), 24000)
            for audio in audios
        ]

    def warmup_point(text, batch):
        # Every batch size is a new tensor shape; run it once untimed.
        generate_batch([text] * batch, args.seed)

    def reset_peak():
        # Drop CUDA graphs captured for earlier points (one per shape), then
        # release cached blocks so nvidia-smi reflects this point, and
        # restart the peak counter.
        graph_cache = getattr(model, "_fi_graph_cache", None)
        if graph_cache:
            graph_cache.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def peak_allocated_mb():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024)

    return {
        "generate_batch": generate_batch,
        "warmup_point": warmup_point,
        "reset_peak": reset_peak,
        "peak_allocated_mb": peak_allocated_mb,
        "is_oom": lambda error: isinstance(error, torch.cuda.OutOfMemoryError),
        "free_memory": torch.cuda.empty_cache,
    }


def load_cpp_engine(args, ref_text):
    sys.path.insert(0, str(SCRIPT_DIR))
    import omnivoice_cpp

    model = omnivoice_cpp.OmniVoice(
        model_path=args.model,
        ref_audio=args.ref_audio,
        ref_text=clean_unicode(ref_text),
        backend="cuda",
        device=0,
        threads=4,
        generator_weight_type="f16",
        audio_tokenizer_weight_type="f16",
    )

    def generate_batch(texts, seed):
        # No batch API in the binding: one call per item, like model.py.
        outputs = []
        for text in texts:
            result = model.generate(
                text=clean_unicode(text),
                language="bn",
                seed=seed,
                steps=args.num_step,
                guidance_scale=args.guidance_scale,
                speed=args.speed,
            )
            outputs.append(
                (
                    np.asarray(result["samples"], dtype=np.float32),
                    int(result["sample_rate"]),
                )
            )
        return outputs

    def warmup_point(text, batch):
        # Sequential calls: every item has the batch-1 shape, so one call
        # warms the point up.
        generate_batch([text], args.seed)

    return {
        "generate_batch": generate_batch,
        "warmup_point": warmup_point,
        "reset_peak": lambda: None,
        "peak_allocated_mb": lambda: None,
        "is_oom": lambda error: "out of memory" in str(error).lower(),
        "free_memory": lambda: None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Worker (runs inside the subprocess for one engine)
# ─────────────────────────────────────────────────────────────────────────────


LOG_LEGEND = (
    "columns: inference time = wall time to generate the whole batch (cpp: all "
    "N sequential calls together), median "
    "of the repeats (min..max); audio duration = longest audio in the batch; "
    "RTF = inference time / longest audio duration in the batch (lower is "
    "better); GPU memory = peak MiB used by the batch (python: torch peak "
    "allocated, cpp: nvidia-smi for the process); GPU utilization = mean "
    "(and max) nvidia-smi utilization sampled every 100 ms during one extra "
    "untimed run of the batch"
)


def format_row(row):
    head = f" 1 instance  | batch {row['batch']:2d} | {row['words']:2d} words | "
    if row["status"] != "ok":
        return head + row["status"]

    gpu = row["peak_allocated_mb"] or row["gpu_mem_mb"]
    return (
        head
        + f"inference time {row['median']:6.3f} s ({row['min']:.3f}..{row['max']:.3f}) | "
        f"audio duration {row['audio_seconds_max']:5.2f} s | "
        f"RTF {row['rtf']:.3f} | "
        f"GPU memory {gpu} MiB | "
        f"GPU utilization {row.get('gpu_util_mean')}% (max {row.get('gpu_util_max')}%)"
    )


def run_worker(args):
    engine = args.worker
    source = f"{engine}"
    out_dir = Path(args.output_dir)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    log(
        args,
        source,
        f"loading engine (pid={os.getpid()}, "
        f"{len(os.sched_getaffinity(0))} CPUs allowed)",
    )
    start = time.perf_counter()

    e = load_engine(args, engine)

    log(
        args,
        source,
        f"engine loaded in {time.perf_counter() - start:.1f} s, "
        f"GPU {gpu_memory_mb()} MiB (nvidia-smi)",
    )

    # Global warmup: every text at batch 1, --warmup times. Each measured
    # point also gets --point-warmup untimed runs of its own shape below.
    start = time.perf_counter()
    for _ in range(args.warmup):
        for words in args.word_counts:
            e["generate_batch"]([TEXTS[words]], args.seed)
    log(
        args,
        source,
        f"warmup done ({args.warmup} x {len(args.word_counts)} texts, "
        f"{time.perf_counter() - start:.1f} s), "
        f"GPU {gpu_memory_mb()} MiB (nvidia-smi)",
    )
    log(args, source, LOG_LEGEND)

    for words in args.word_counts:
        text = TEXTS[words]
        skip_rest = False

        for batch in batch_sizes:
            update_summary(args, f"running {engine}, {words} words, batch {batch}")

            row = {
                "engine": engine,
                "words": words,
                "batch": batch,
                "status": "ok",
                "times": [],
                "peak_allocated_mb": None,
            }

            if skip_rest:
                row["status"] = "skipped (OOM at smaller batch)"
            else:
                try:
                    e["reset_peak"]()

                    for _ in range(args.point_warmup):
                        e["warmup_point"](text, batch)

                    for repeat in range(args.repeats):
                        t0 = time.perf_counter()
                        outputs = e["generate_batch"]([text] * batch, args.seed + repeat)
                        row["times"].append(time.perf_counter() - t0)

                    # Utilization comes from one extra, untimed run: polling
                    # nvidia-smi contends with kernel launches in the driver
                    # and slowed the timed runs by up to ~35% at small batches.
                    sampler = GpuUtilSampler(args.gpu)
                    with sampler:
                        # Repeat short runs so the 100 ms sampler gets enough samples.
                        t0 = time.perf_counter()
                        while True:
                            e["generate_batch"]([text] * batch, args.seed)
                            if time.perf_counter() - t0 >= UTIL_MIN_SECONDS:
                                break
                    row["gpu_util_mean"] = sampler.mean()
                    row["gpu_util_max"] = sampler.max()

                    if args.save_audio:
                        audio, sample_rate = outputs[0]
                        save_wav(
                            out_dir / "audio" / engine
                            / f"w{words:02d}_b{batch:02d}.wav",
                            audio,
                            sample_rate,
                        )

                    durations = [len(a) / sr for a, sr in outputs]
                    row["audio_seconds_per_item"] = statistics.mean(durations)
                    row["audio_seconds_total"] = sum(durations)
                    row["audio_seconds_max"] = max(durations)
                    row["peak_allocated_mb"] = e["peak_allocated_mb"]()

                except Exception as error:  # noqa: BLE001 - logged and recorded
                    if not e["is_oom"](error):
                        raise
                    e["free_memory"]()
                    row["status"] = "OOM"
                    skip_rest = True

            row["gpu_mem_mb"] = gpu_memory_mb()

            if row["times"]:
                row["median"] = statistics.median(row["times"])
                row["min"] = min(row["times"])
                row["max"] = max(row["times"])
                row["per_item"] = row["median"] / batch
                row["rtf"] = row["median"] / row["audio_seconds_max"]

            with open(out_dir / RESULTS_NAME, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            log(args, source, format_row(row))

    log(args, source, "done")


# ─────────────────────────────────────────────────────────────────────────────
#  Multiple instances: N engine processes on the same GPU, each with its own
#  copy of the model (like Triton instance_group count=N). N concurrent
#  requests, one per instance; inference time = wall time from sending all N
#  until all N are done.
# ─────────────────────────────────────────────────────────────────────────────


def load_engine(args, engine):
    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    if engine == "python":
        return load_python_engine(args, ref_text)
    if engine == "python_fi":
        return load_python_engine(args, ref_text, flashinfer="plain")
    if engine == "python_fi_graph":
        return load_python_engine(args, ref_text, flashinfer="graph")
    if engine == "cpp":
        return load_cpp_engine(args, ref_text)
    raise ValueError(f"Unknown engine: {engine}")


def instance_main(args, engine, index, in_queue, out_queue):
    """One engine instance (spawned process): load, warm up, serve requests."""
    try:
        e = load_engine(args, engine)
        for _ in range(args.warmup):
            for words in args.word_counts:
                e["generate_batch"]([TEXTS[words]], args.seed)
    except Exception as error:  # noqa: BLE001 - reported to the coordinator
        out_queue.put(("load_failed", index, repr(error)))
        return

    out_queue.put(("ready", index, os.getpid()))

    while True:
        request = in_queue.get()
        if request is None:
            return
        text, seed = request
        try:
            outputs = e["generate_batch"]([text], seed)
            audio, sample_rate = outputs[0]
            out_queue.put(("done", index, len(audio) / sample_rate))
        except Exception as error:  # noqa: BLE001 - reported to the coordinator
            out_queue.put(("error", index, repr(error)))


def gpu_memory_used_mb(gpu):
    # Whole-GPU memory in use: every instance plus its CUDA context.
    try:
        output = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return int(output)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run_instances(args):
    engine = args.worker_instances
    source = f"{engine} instances"
    out_dir = Path(args.output_dir)
    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()
    instances = []  # (process, in_queue)
    max_supported = 0
    stop_reason = f"reached --max-instances {args.max_instances}"

    def wait_ready(process, index, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = out_queue.get(timeout=1)
            except queue.Empty:
                if not process.is_alive():
                    return f"process exited with code {process.exitcode}"
                continue
            if message[0] == "ready" and message[1] == index:
                return None
            if message[0] == "load_failed" and message[1] == index:
                return message[2]
        return "timed out while loading"

    def run_round(text, seed):
        """Send one request to every instance; wall time until all finish."""
        t0 = time.perf_counter()
        for _, in_queue in instances:
            in_queue.put((text, seed))
        durations, errors = [], []
        while len(durations) + len(errors) < len(instances):
            try:
                message = out_queue.get(timeout=1)
            except queue.Empty:
                dead = [p for p, _ in instances if not p.is_alive()]
                if dead:
                    errors.append(f"instance process died (exit {dead[0].exitcode})")
                    break
                continue
            if message[0] == "done":
                durations.append(message[2])
            elif message[0] == "error":
                errors.append(message[2])
        return time.perf_counter() - t0, durations, errors

    log(args, source, f"coordinator pid={os.getpid()}, max {args.max_instances} instances")

    try:
        for n in range(1, args.max_instances + 1):
            in_queue = ctx.Queue()
            process = ctx.Process(
                target=instance_main,
                args=(args, engine, n - 1, in_queue, out_queue),
                daemon=True,
            )
            start = time.perf_counter()
            process.start()
            instances.append((process, in_queue))

            error = wait_ready(process, n - 1)
            if error is not None:
                stop_reason = f"instance {n} failed to load: {error[:200]}"
                log(args, source, stop_reason)
                process.terminate()
                instances.pop()
                break

            log(
                args,
                source,
                f"instance {n} loaded in {time.perf_counter() - start:.1f} s "
                f"(pid={process.pid}), GPU memory {gpu_memory_used_mb(args.gpu)} MiB total",
            )

            failed = False
            for words in args.word_counts:
                update_summary(args, f"running {engine}, {n} instances, {words} words")
                text = TEXTS[words]
                row = {
                    "engine": engine,
                    "instances": n,
                    "words": words,
                    "status": "ok",
                    "times": [],
                }

                run_round(text, args.seed)  # untimed warmup of the point
                for repeat in range(args.repeats):
                    elapsed, durations, errors = run_round(text, args.seed + repeat)
                    if errors:
                        row["status"] = "OOM" if "out of memory" in errors[0].lower() else "error"
                        row["error"] = errors[0][:300]
                        break
                    row["times"].append(elapsed)

                if row["status"] == "ok":
                    sampler = GpuUtilSampler(args.gpu)
                    with sampler:
                        # Repeat short rounds so the 100 ms sampler gets enough samples.
                        t0 = time.perf_counter()
                        while True:
                            _, _, errors = run_round(text, args.seed)
                            if errors or time.perf_counter() - t0 >= UTIL_MIN_SECONDS:
                                break
                    row["gpu_util_mean"] = sampler.mean()
                    row["gpu_util_max"] = sampler.max()
                    row["gpu_mem_total_mb"] = gpu_memory_used_mb(args.gpu)
                    row["median"] = statistics.median(row["times"])
                    row["min"] = min(row["times"])
                    row["max"] = max(row["times"])
                    row["audio_seconds_max"] = max(durations)
                    row["rtf"] = row["median"] / row["audio_seconds_max"]

                with open(out_dir / INSTANCES_NAME, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

                log(args, source, format_instance_row(row))

                if row["status"] != "ok":
                    failed = True
                    break

            if failed:
                stop_reason = f"{n} instances failed while running ({row['status']})"
                break
            max_supported = n
    finally:
        for process, in_queue in instances:
            try:
                in_queue.put(None)
            except (OSError, ValueError):
                pass
        for process, _ in instances:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()

    with open(out_dir / INSTANCES_NAME, "a", encoding="utf-8") as f:
        f.write(json.dumps({"engine": engine, "max_supported": max_supported,
                            "stop_reason": stop_reason}) + "\n")
    log(args, source, f"done: max supported instances = {max_supported} ({stop_reason})")


def format_instance_row(row):
    label = "instance " if row["instances"] == 1 else "instances"
    head = f"{row['instances']:2d} {label} | batch  1 | {row['words']:2d} words | "
    if row["status"] != "ok":
        return head + f"{row['status']}: {row.get('error', '')}"
    return (
        head
        + f"inference time {row['median']:6.3f} s ({row['min']:.3f}..{row['max']:.3f}) | "
        f"audio duration {row['audio_seconds_max']:5.2f} s | "
        f"RTF {row['rtf']:.3f} | "
        f"GPU memory {row['gpu_mem_total_mb']} MiB total | "
        f"GPU utilization {row.get('gpu_util_mean')}% (max {row.get('gpu_util_max')}%)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Summary markdown (rewritten after every point, readable while running)
# ─────────────────────────────────────────────────────────────────────────────


def load_rows(args):
    path = Path(args.output_dir) / RESULTS_NAME
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def update_summary(args, status):
    """Write the summary md: short notes and one table with a row per point."""
    out_dir = Path(args.output_dir)
    engines = list(args.engines)
    engine_order = {engine: i for i, engine in enumerate(engines)}

    points = []
    for row in read_jsonl(out_dir / RESULTS_NAME):
        points.append(dict(row, phase="batch sizes", instances=1,
                           batch_per_instance=row["batch"],
                           gpu_memory=row.get("peak_allocated_mb") or row.get("gpu_mem_mb")))
    limits = {}
    for row in read_jsonl(out_dir / INSTANCES_NAME):
        if "max_supported" in row:
            limits[row["engine"]] = row
            continue
        points.append(dict(row, phase="instances", batch_per_instance=1,
                           gpu_memory=row.get("gpu_mem_total_mb")))
    points = [p for p in points if p["engine"] in engine_order]
    points.sort(key=lambda p: (
        p["words"],
        engine_order[p["engine"]],
        p["phase"] != "batch sizes",
        p["instances"],
        p["batch_per_instance"],
    ))

    limit_text = "; ".join(
        f"{ENGINE_NAMES[e]}: {limits[e]['max_supported']} ({limits[e]['stop_reason']})"
        for e in engines if e in limits
    ) or "not measured yet"

    lines = [
        "# OmniVoice batch-size and multiple-instance benchmark",
        "",
        f"**Status:** {status}  ",
        f"**Updated:** {now()}",
        "",
        "## Notes",
        "",
        f"- GPU: nvidia-smi index {args.gpu} (RTX 4090 = 0), other 4090 services stopped "
        f"during the run. CPU: {args.cpu_affinity}.",
        f"- Model: `{args.model}`, reference `{args.ref_audio}` + `{args.ref_text_file}` "
        "(encoded once, reused).",
        f"- num_step={args.num_step}, guidance_scale={args.guidance_scale}, "
        f"speed={args.speed}. Each point: {args.point_warmup} untimed warmup run, then "
        f"{args.repeats} timed runs (median reported); {args.warmup} warmup runs of every "
        "text at startup.",
        "- **Phase \"batch sizes\"**: one instance (one process, one model copy) gets one "
        "batch of N copies of the text. C++ has no batch API, so its batch of N is N "
        "sequential calls.",
        "- **Phase \"instances\"**: N instances (N processes on the GPU, each with its own "
        "model copy, like Triton `instance_group count: N`) each get one request at the "
        "same time. Instances are added until one no longer fits.",
        f"- **Max instances supported on this GPU**: {limit_text}.",
        "- **Concurrent requests** = instances × batch size per instance.",
        "- **Inference time** = wall time until every request of the point is done (also "
        "the latency of each of those requests). Min–max is over the timed runs.",
        "- **RTF** = inference time / longest audio duration (lower is better).",
        "- **GPU memory**: batch-size phase = memory of the one process (Python engines: "
        "PyTorch peak allocated; C++: nvidia-smi); instances phase = whole GPU, all "
        "instances.",
        "- **GPU utilization** = mean nvidia-smi `utilization.gpu` during extra untimed "
        f"runs of at least {UTIL_MIN_SECONDS:g} s (share of time a kernel was running).",
        "",
        "Engines:",
        "",
    ]
    for engine in engines:
        lines.append(f"- **{ENGINE_NAMES[engine]}**: {ENGINE_DESCRIPTIONS[engine]}")
    lines += ["", "Texts (the same text in every request of a point):", ""]
    for words in args.word_counts:
        lines.append(f"- {words} words: {TEXTS[words]}")

    columns = [
        "Text size (words)",
        "Engine",
        "Phase",
        "Instances",
        "Batch size per instance",
        "Concurrent requests",
        "Inference time (s)",
        "Inference time min–max (s)",
        "Longest audio duration (s)",
        "RTF",
        "GPU memory (MiB)",
        "GPU utilization (%)",
        "Status",
    ]
    lines += [
        "",
        "## Results",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]

    def show(value, digits):
        if value is None:
            return "–"
        if isinstance(value, int) or digits == 0:
            return str(int(round(value)))
        return f"{value:.{digits}f}"

    for p in points:
        ok = p["status"] == "ok"
        cells = [
            str(p["words"]),
            ENGINE_NAMES[p["engine"]],
            p["phase"],
            str(p["instances"]),
            str(p["batch_per_instance"]),
            str(p["instances"] * p["batch_per_instance"]),
            show(p.get("median"), 3) if ok else "–",
            f"{p['min']:.3f}–{p['max']:.3f}" if ok else "–",
            show(p.get("audio_seconds_max"), 2) if ok else "–",
            show(p.get("rtf"), 4) if ok else "–",
            show(p.get("gpu_memory"), 0) if ok else "–",
            show(p.get("gpu_util_mean"), 0) if ok else "–",
            "ok" if ok else p["status"],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"Raw data: `{RESULTS_NAME}`, `{INSTANCES_NAME}`. Log: `{LOG_NAME}`.",
        "",
    ]

    path = out_dir / SUMMARY_NAME
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────


def worker_command(args, engine, mode="--worker"):
    python_bin = args.python_bin if engine.startswith("python") else args.cpp_python_bin

    command = [
        python_bin,
        str(Path(__file__).resolve()),
        mode, engine,
        "--engines", *args.engines,
        "--model", args.model,
        "--ref-audio", args.ref_audio,
        "--ref-text-file", args.ref_text_file,
        "--batch-sizes", args.batch_sizes,
        "--word-counts", *[str(w) for w in args.word_counts],
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--point-warmup", str(args.point_warmup),
        "--num-step", str(args.num_step),
        "--guidance-scale", str(args.guidance_scale),
        "--speed", str(args.speed),
        "--seed", str(args.seed),
        "--gpu", args.gpu,
        "--budgets", *[str(b) for b in args.budgets],
        "--output-dir", args.output_dir,
        "--cpu-affinity", args.cpu_affinity,
        "--max-instances", str(args.max_instances),
    ]
    if not args.save_audio:
        command.append("--no-save-audio")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="+", default=ALL_ENGINES,
                        choices=ALL_ENGINES)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--ref-audio", default=REF_AUDIO)
    parser.add_argument("--ref-text-file", default=REF_TEXT_FILE)
    parser.add_argument("--batch-sizes", default="1-20",
                        help="e.g. 1-20 or 1,2,4,8")
    parser.add_argument("--word-counts", nargs="+", type=int,
                        default=sorted(TEXTS), choices=sorted(TEXTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed runs of every text at batch 1 at startup")
    parser.add_argument("--point-warmup", type=int, default=1,
                        help="Untimed runs before each measured point")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0",
                        help="nvidia-smi index; 0 = RTX 4090, 1 = Tesla T4")
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0, 3.0, 5.0],
                        help="Not used any more (the summary is one table); kept so older commands still run")
    parser.add_argument("--python-bin", default=PYTHON_BIN)
    parser.add_argument("--cpp-python-bin", default=CPP_PYTHON_BIN)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-save-audio", dest="save_audio",
                        action="store_false",
                        help="Do not save item 0 of each batch as a WAV")
    parser.add_argument("--no-pin", dest="pin", action="store_false",
                        help="Do not pin workers to the GPU's NUMA node CPUs")
    parser.add_argument("--cpu-affinity", default="not pinned",
                        help=argparse.SUPPRESS)
    parser.add_argument("--summary-only", action="store_true",
                        help="Only rebuild the summary md from results.jsonl")
    parser.add_argument("--summary-status", default="complete",
                        help="Status line written by --summary-only")
    parser.add_argument("--max-instances", type=int, default=12,
                        help="Upper limit for the multiple-instances phase "
                             "(it stops earlier when an instance does not fit); 0 = skip")
    parser.add_argument("--skip-batch", action="store_true",
                        help="Skip the batch-size phase (run only instances)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-instances", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.word_counts = sorted(args.word_counts)

    for words, text in TEXTS.items():
        assert len(text.split()) == words, (words, len(text.split()))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.worker:
        run_worker(args)
        return

    if args.worker_instances:
        run_instances(args)
        return

    if args.summary_only:
        update_summary(args, args.summary_status)
        return

    (Path(args.output_dir) / RESULTS_NAME).unlink(missing_ok=True)
    (Path(args.output_dir) / INSTANCES_NAME).unlink(missing_ok=True)

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    pin_cpus = None
    numa = gpu_numa_cpus(args.gpu) if args.pin else None
    if numa is not None:
        node, pin_cpus = numa
        args.cpu_affinity = (
            f"workers pinned to NUMA node {node} (the GPU's socket), "
            f"CPUs {','.join(str(c) for c in sorted(pin_cpus))}"
        )
    elif args.pin:
        args.cpu_affinity = "not pinned (GPU NUMA node unknown)"

    log(
        args,
        "driver",
        f"start engines={args.engines} words={args.word_counts} "
        f"batches={args.batch_sizes} repeats={args.repeats} gpu={args.gpu}",
    )
    log(args, "driver", f"CPU: {args.cpu_affinity}")
    update_summary(args, "starting")

    phases = []
    if not args.skip_batch:
        phases += [(engine, "--worker", "batch sizes") for engine in args.engines]
    if args.max_instances > 0:
        phases += [(engine, "--worker-instances", "instances") for engine in args.engines]

    failed = []
    for engine, mode, title in phases:
        log(args, "driver", f"=== {engine}: {title} ===")
        result = subprocess.run(
            worker_command(args, engine, mode),
            env=env,
            preexec_fn=(lambda: os.sched_setaffinity(0, pin_cpus)) if pin_cpus else None,
        )
        if result.returncode != 0:
            failed.append(f"{engine} {title}")
            log(args, "driver", f"{engine} {title} failed (exit {result.returncode})")

    status = (
        "complete"
        if not failed
        else f"finished with errors ({', '.join(failed)} failed, see log)"
    )
    update_summary(args, status)
    log(args, "driver", status)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
