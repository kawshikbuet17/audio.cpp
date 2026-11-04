# OmniVoice batch-size and multiple-instance benchmark

**Status:** complete  
**Updated:** 2026-11-04 17:50:48

## Notes

- GPU: nvidia-smi index 0 (RTX 4090 = 0), other 4090 services stopped during the run. CPU: workers pinned to NUMA node 0 (the GPU's socket), CPUs 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38.
- Model: `/home/ipt/kawshik/tts-testing/OmniVoice/models/OmniVoice-LoRA-31-Aug-2026`, reference `ref_audios/ref_audio_03.wav` + `ref_audios/ref_text_03.txt` (encoded once, reused).
- num_step=16, guidance_scale=2.0, speed=1.0. Each point: 1 untimed warmup run, then 3 timed runs (median reported); 3 warmup runs of every text at startup.
- **Phase "batch sizes"**: one instance (one process, one model copy) gets one batch of N copies of the text. C++ has no batch API, so its batch of N is N sequential calls.
- **Phase "instances"**: N instances (N processes on the GPU, each with its own model copy, like Triton `instance_group count: N`) each get one request at the same time. Instances are added until one no longer fits.
- **Max instances supported on this GPU**: Python: 5 (instance 6 failed to load: OutOfMemoryError('CUDA out of memory. Tried to allocate 36.00 MiB. GPU 0 has a total capacity of 23.55 GiB of which 29.19 MiB is free. Process 1128037 has 4.22 GiB memory in use. Process 1129337 has 4); Python + FlashInfer: 4 (instance 5 failed to load: OutOfMemoryError('CUDA out of memory. Tried to allocate 34.00 MiB. GPU 0 has a total capacity of 23.55 GiB of which 12.50 MiB is free. Process 1135658 has 4.87 GiB memory in use. Process 1136640 has 4); Python + FlashInfer + CUDA graphs: 5 (instance 6 failed to load: OutOfMemoryError('CUDA out of memory. Tried to allocate 34.00 MiB. GPU 0 has a total capacity of 23.55 GiB of which 21.19 MiB is free. Process 1140283 has 3.96 GiB memory in use. Process 1141143 has 3); C++: 9 (instance 10 failed to load: RuntimeError('failed to allocate omnivoice.generator.weights backend weight buffer')).
- **Concurrent requests** = instances × batch size per instance.
- **Inference time** = wall time until every request of the point is done (also the latency of each of those requests). Min–max is over the timed runs.
- **RTF** = inference time / longest audio duration (lower is better).
- **GPU memory**: batch-size phase = memory of the one process (Python engines: PyTorch peak allocated; C++: nvidia-smi); instances phase = whole GPU, all instances.
- **GPU utilization** = mean nvidia-smi `utilization.gpu` during extra untimed runs of at least 1.5 s (share of time a kernel was running).

Engines:

- **Python**: OmniVoice fp16, native batching, `generate(text=[text] * batch)`: the generator runs the whole batch together; audio decode and post-processing still loop per item. No language.
- **Python + FlashInfer**: same as Python, plus `apply_flashinfer(model)` (OmniVoice commit 28bc088): cond/uncond packed into one row with FlashInfer ragged attention, fused RMSNorm/RoPE/QKV/MLP kernels, KV cache off.
- **Python + FlashInfer + CUDA graphs**: same as Python + FlashInfer, plus `enable_cuda_graph=True`: one CUDA graph per packed shape, replayed for every step. A new shape (batch size or text length) is captured on first use; the per-point warmup does that here, and the graph cache is cleared before each point.
- **C++**: `.so` generator f16, audio tokenizer f16 (codebook f32, the binding default), `language="bn"`, NFC-normalized text. **No batch API: a batch of N is N sequential calls.**

Texts (the same text in every request of a point):

- 5 words: আপনার লেনদেন সফলভাবে সম্পন্ন হয়েছে।
- 10 words: আপনার বিকাশ অ্যাকাউন্টে পাঁচশো টাকা জমা হয়েছে, ব্যালেন্স দেখে নিন।
- 15 words: যদি নম্বরটি বিকাশ অ্যাকাউন্ট না হয়, তবে আপনি অ্যাপ থেকে লেনদেনটি সহজেই বাতিল করতে পারবেন।
- 20 words: আপনি প্রথমে প্রাপকের বিকাশ নম্বরটি নিশ্চিত করবেন, তারপর টাকার পরিমাণ লিখে পিন দিয়ে লেনদেনটি সম্পন্ন করবেন এবং কনফার্মেশন মেসেজ দেখবেন।

## Results

| Text size (words) | Engine | Phase | Instances | Batch size per instance | Concurrent requests | Inference time (s) | Inference time min–max (s) | Longest audio duration (s) | RTF | GPU memory (MiB) | GPU utilization (%) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Python | batch sizes | 1 | 1 | 1 | 0.875 | 0.874–0.878 | 3.16 | 0.2769 | 2061 | 22 | ok |
| 5 | Python | batch sizes | 1 | 2 | 2 | 0.918 | 0.899–1.085 | 3.16 | 0.2905 | 2176 | 34 | ok |
| 5 | Python | batch sizes | 1 | 3 | 3 | 0.945 | 0.938–1.007 | 3.16 | 0.2992 | 2289 | 39 | ok |
| 5 | Python | batch sizes | 1 | 4 | 4 | 0.998 | 0.965–1.051 | 3.16 | 0.3157 | 2411 | 45 | ok |
| 5 | Python | batch sizes | 1 | 5 | 5 | 0.998 | 0.985–1.108 | 3.16 | 0.3159 | 2519 | 66 | ok |
| 5 | Python | batch sizes | 1 | 6 | 6 | 1.021 | 1.020–1.032 | 3.16 | 0.3230 | 2634 | 61 | ok |
| 5 | Python | batch sizes | 1 | 7 | 7 | 1.087 | 1.078–1.287 | 3.16 | 0.3438 | 2753 | 80 | ok |
| 5 | Python | batch sizes | 1 | 8 | 8 | 1.226 | 1.225–1.227 | 3.16 | 0.3881 | 2876 | 87 | ok |
| 5 | Python | batch sizes | 1 | 9 | 9 | 1.412 | 1.411–1.416 | 3.16 | 0.4468 | 2977 | 86 | ok |
| 5 | Python | batch sizes | 1 | 10 | 10 | 1.587 | 1.580–1.590 | 3.16 | 0.5021 | 3096 | 84 | ok |
| 5 | Python | batch sizes | 1 | 11 | 11 | 1.782 | 1.781–1.788 | 3.16 | 0.5641 | 3218 | 87 | ok |
| 5 | Python | batch sizes | 1 | 12 | 12 | 1.993 | 1.992–1.995 | 3.16 | 0.6308 | 3320 | 86 | ok |
| 5 | Python | batch sizes | 1 | 13 | 13 | 2.152 | 2.147–2.153 | 3.16 | 0.6809 | 3437 | 91 | ok |
| 5 | Python | batch sizes | 1 | 14 | 14 | 2.356 | 2.355–2.358 | 3.16 | 0.7456 | 3560 | 86 | ok |
| 5 | Python | batch sizes | 1 | 15 | 15 | 2.586 | 2.584–2.588 | 3.16 | 0.8184 | 3664 | 90 | ok |
| 5 | Python | batch sizes | 1 | 16 | 16 | 2.761 | 2.759–2.765 | 3.16 | 0.8737 | 3779 | 82 | ok |
| 5 | Python | batch sizes | 1 | 17 | 17 | 2.980 | 2.978–2.983 | 3.16 | 0.9430 | 3901 | 92 | ok |
| 5 | Python | batch sizes | 1 | 18 | 18 | 3.196 | 3.191–3.221 | 3.16 | 1.0114 | 4008 | 93 | ok |
| 5 | Python | batch sizes | 1 | 19 | 19 | 3.392 | 3.390–3.394 | 3.16 | 1.0735 | 4123 | 91 | ok |
| 5 | Python | batch sizes | 1 | 20 | 20 | 3.617 | 3.617–3.625 | 3.16 | 1.1446 | 4243 | 90 | ok |
| 5 | Python | instances | 1 | 1 | 1 | 0.917 | 0.901–0.926 | 3.16 | 0.2903 | 4330 | 22 | ok |
| 5 | Python | instances | 2 | 1 | 2 | 0.968 | 0.967–1.039 | 3.16 | 0.3063 | 8655 | 90 | ok |
| 5 | Python | instances | 3 | 1 | 3 | 1.375 | 1.141–1.469 | 3.16 | 0.4351 | 12980 | 94 | ok |
| 5 | Python | instances | 4 | 1 | 4 | 1.288 | 1.146–1.424 | 3.16 | 0.4075 | 17306 | 94 | ok |
| 5 | Python | instances | 5 | 1 | 5 | 1.125 | 1.062–1.159 | 3.16 | 0.3560 | 21631 | 95 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 1 | 1 | 0.442 | 0.433–0.443 | 3.16 | 0.1398 | 2682 | 18 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 2 | 2 | 0.583 | 0.522–0.587 | 3.16 | 0.1845 | 2682 | 29 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 3 | 3 | 0.618 | 0.613–0.669 | 3.16 | 0.1956 | 2689 | 30 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 4 | 4 | 0.574 | 0.527–0.651 | 3.16 | 0.1815 | 2707 | 40 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 5 | 5 | 0.657 | 0.636–0.671 | 3.16 | 0.2080 | 2719 | 37 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 6 | 6 | 0.687 | 0.648–0.787 | 3.16 | 0.2174 | 2736 | 40 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 7 | 7 | 0.601 | 0.594–0.633 | 3.16 | 0.1901 | 2752 | 53 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 8 | 8 | 0.740 | 0.632–0.749 | 3.16 | 0.2341 | 2767 | 64 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 9 | 9 | 0.693 | 0.659–0.789 | 3.16 | 0.2194 | 2781 | 75 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 10 | 10 | 0.731 | 0.729–0.732 | 3.16 | 0.2314 | 2798 | 76 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 11 | 11 | 0.784 | 0.783–0.834 | 3.16 | 0.2481 | 2813 | 73 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 12 | 12 | 0.850 | 0.849–0.855 | 3.16 | 0.2690 | 2830 | 70 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 13 | 13 | 0.925 | 0.924–1.026 | 3.16 | 0.2926 | 2844 | 82 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 14 | 14 | 0.988 | 0.986–0.994 | 3.16 | 0.3128 | 2860 | 70 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 15 | 15 | 1.072 | 1.070–1.075 | 3.16 | 0.3394 | 2875 | 71 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 16 | 16 | 1.132 | 1.128–1.133 | 3.16 | 0.3581 | 2893 | 69 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 17 | 17 | 1.191 | 1.189–1.201 | 3.16 | 0.3771 | 2906 | 68 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 18 | 18 | 1.277 | 1.273–1.278 | 3.16 | 0.4042 | 2921 | 73 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 19 | 19 | 1.340 | 1.335–1.341 | 3.16 | 0.4240 | 2938 | 78 | ok |
| 5 | Python + FlashInfer | batch sizes | 1 | 20 | 20 | 1.413 | 1.411–1.416 | 3.16 | 0.4472 | 2952 | 78 | ok |
| 5 | Python + FlashInfer | instances | 1 | 1 | 1 | 0.442 | 0.420–0.488 | 3.16 | 0.1397 | 4994 | 20 | ok |
| 5 | Python + FlashInfer | instances | 2 | 1 | 2 | 0.466 | 0.441–0.489 | 3.16 | 0.1474 | 9983 | 90 | ok |
| 5 | Python + FlashInfer | instances | 3 | 1 | 3 | 0.455 | 0.453–0.484 | 3.16 | 0.1439 | 14972 | 89 | ok |
| 5 | Python + FlashInfer | instances | 4 | 1 | 4 | 0.533 | 0.532–0.617 | 3.16 | 0.1685 | 19962 | 93 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 1 | 1 | 0.101 | 0.101–0.102 | 3.16 | 0.0320 | 2826 | 71 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 2 | 2 | 0.173 | 0.172–0.173 | 3.16 | 0.0547 | 2834 | 71 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 3 | 3 | 0.235 | 0.233–0.250 | 3.16 | 0.0744 | 2842 | 66 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 4 | 4 | 0.308 | 0.305–0.308 | 3.16 | 0.0974 | 2853 | 68 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 5 | 5 | 0.377 | 0.375–0.381 | 3.16 | 0.1192 | 2879 | 68 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 6 | 6 | 0.438 | 0.419–0.438 | 3.16 | 0.1385 | 2901 | 70 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 7 | 7 | 0.485 | 0.484–0.487 | 3.16 | 0.1535 | 2925 | 68 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 8 | 8 | 0.573 | 0.568–0.594 | 3.16 | 0.1813 | 2950 | 76 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 9 | 9 | 0.658 | 0.652–0.686 | 3.16 | 0.2083 | 2974 | 72 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 10 | 10 | 0.721 | 0.720–0.727 | 3.16 | 0.2282 | 3002 | 78 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 11 | 11 | 0.770 | 0.767–0.803 | 3.16 | 0.2438 | 3025 | 69 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 12 | 12 | 0.837 | 0.835–0.841 | 3.16 | 0.2650 | 3048 | 74 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 13 | 13 | 0.910 | 0.909–0.914 | 3.16 | 0.2879 | 3072 | 74 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 14 | 14 | 1.007 | 0.991–1.030 | 3.16 | 0.3187 | 3096 | 73 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 15 | 15 | 1.062 | 1.054–1.062 | 3.16 | 0.3361 | 3121 | 71 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 16 | 16 | 1.127 | 1.125–1.144 | 3.16 | 0.3567 | 3149 | 71 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 17 | 17 | 1.186 | 1.184–1.201 | 3.16 | 0.3752 | 3172 | 69 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 18 | 18 | 1.275 | 1.270–1.277 | 3.16 | 0.4035 | 3196 | 77 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 19 | 19 | 1.333 | 1.332–1.340 | 3.16 | 0.4220 | 3218 | 74 | ok |
| 5 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 20 | 20 | 1.416 | 1.416–1.420 | 3.16 | 0.4480 | 3243 | 73 | ok |
| 5 | Python + FlashInfer + CUDA graphs | instances | 1 | 1 | 1 | 0.098 | 0.098–0.098 | 3.16 | 0.0309 | 4062 | 70 | ok |
| 5 | Python + FlashInfer + CUDA graphs | instances | 2 | 1 | 2 | 0.178 | 0.177–0.195 | 3.16 | 0.0564 | 8119 | 85 | ok |
| 5 | Python + FlashInfer + CUDA graphs | instances | 3 | 1 | 3 | 0.251 | 0.251–0.255 | 3.16 | 0.0795 | 12176 | 90 | ok |
| 5 | Python + FlashInfer + CUDA graphs | instances | 4 | 1 | 4 | 0.333 | 0.332–0.334 | 3.16 | 0.1055 | 16234 | 90 | ok |
| 5 | Python + FlashInfer + CUDA graphs | instances | 5 | 1 | 5 | 0.415 | 0.409–0.425 | 3.16 | 0.1313 | 20291 | 91 | ok |
| 5 | C++ | batch sizes | 1 | 1 | 1 | 0.308 | 0.308–0.308 | 3.15 | 0.0978 | 2462 | 62 | ok |
| 5 | C++ | batch sizes | 1 | 2 | 2 | 0.607 | 0.604–0.610 | 3.15 | 0.1927 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 3 | 3 | 0.912 | 0.907–0.919 | 3.15 | 0.2895 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 4 | 4 | 1.220 | 1.206–1.223 | 3.15 | 0.3872 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 5 | 5 | 1.553 | 1.520–1.571 | 3.15 | 0.4929 | 2462 | 61 | ok |
| 5 | C++ | batch sizes | 1 | 6 | 6 | 1.823 | 1.812–1.833 | 3.15 | 0.5786 | 2462 | 60 | ok |
| 5 | C++ | batch sizes | 1 | 7 | 7 | 2.134 | 2.123–2.141 | 3.15 | 0.6775 | 2462 | 61 | ok |
| 5 | C++ | batch sizes | 1 | 8 | 8 | 2.447 | 2.436–2.451 | 3.15 | 0.7770 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 9 | 9 | 2.745 | 2.727–2.752 | 3.15 | 0.8714 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 10 | 10 | 3.045 | 3.037–3.056 | 3.15 | 0.9667 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 11 | 11 | 3.344 | 3.342–3.355 | 3.15 | 1.0614 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 12 | 12 | 3.683 | 3.667–3.765 | 3.15 | 1.1693 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 13 | 13 | 3.960 | 3.956–4.011 | 3.15 | 1.2573 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 14 | 14 | 4.265 | 4.237–4.270 | 3.15 | 1.3540 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 15 | 15 | 4.565 | 4.564–4.577 | 3.15 | 1.4492 | 2462 | 62 | ok |
| 5 | C++ | batch sizes | 1 | 16 | 16 | 4.866 | 4.849–4.891 | 3.15 | 1.5446 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 17 | 17 | 5.162 | 5.151–5.199 | 3.15 | 1.6386 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 18 | 18 | 5.457 | 5.451–5.466 | 3.15 | 1.7324 | 2462 | 64 | ok |
| 5 | C++ | batch sizes | 1 | 19 | 19 | 5.757 | 5.755–5.768 | 3.15 | 1.8276 | 2462 | 63 | ok |
| 5 | C++ | batch sizes | 1 | 20 | 20 | 6.081 | 6.034–6.098 | 3.15 | 1.9305 | 2462 | 64 | ok |
| 5 | C++ | instances | 1 | 1 | 1 | 0.304 | 0.302–0.307 | 3.15 | 0.0964 | 2472 | 62 | ok |
| 5 | C++ | instances | 2 | 1 | 2 | 0.472 | 0.471–0.479 | 3.15 | 0.1498 | 4939 | 85 | ok |
| 5 | C++ | instances | 3 | 1 | 3 | 0.714 | 0.712–0.717 | 3.15 | 0.2267 | 7406 | 90 | ok |
| 5 | C++ | instances | 4 | 1 | 4 | 0.945 | 0.945–0.961 | 3.15 | 0.3001 | 9874 | 91 | ok |
| 5 | C++ | instances | 5 | 1 | 5 | 1.174 | 1.172–1.181 | 3.15 | 0.3727 | 12341 | 92 | ok |
| 5 | C++ | instances | 6 | 1 | 6 | 1.400 | 1.397–1.408 | 3.15 | 0.4445 | 14808 | 96 | ok |
| 5 | C++ | instances | 7 | 1 | 7 | 1.626 | 1.624–1.630 | 3.15 | 0.5163 | 17276 | 91 | ok |
| 5 | C++ | instances | 8 | 1 | 8 | 1.855 | 1.853–1.864 | 3.15 | 0.5890 | 19743 | 91 | ok |
| 5 | C++ | instances | 9 | 1 | 9 | 2.083 | 2.082–2.090 | 3.15 | 0.6614 | 22212 | 94 | ok |
| 10 | Python | batch sizes | 1 | 1 | 1 | 0.889 | 0.883–0.898 | 4.80 | 0.1851 | 2088 | 21 | ok |
| 10 | Python | batch sizes | 1 | 2 | 2 | 0.950 | 0.921–0.950 | 4.80 | 0.1978 | 2233 | 37 | ok |
| 10 | Python | batch sizes | 1 | 3 | 3 | 0.970 | 0.944–0.970 | 4.80 | 0.2021 | 2369 | 51 | ok |
| 10 | Python | batch sizes | 1 | 4 | 4 | 1.005 | 0.999–1.133 | 4.80 | 0.2093 | 2502 | 62 | ok |
| 10 | Python | batch sizes | 1 | 5 | 5 | 1.046 | 1.041–1.053 | 4.80 | 0.2180 | 2640 | 72 | ok |
| 10 | Python | batch sizes | 1 | 6 | 6 | 1.124 | 1.120–1.204 | 4.80 | 0.2341 | 2793 | 73 | ok |
| 10 | Python | batch sizes | 1 | 7 | 7 | 1.332 | 1.331–1.354 | 4.80 | 0.2775 | 2925 | 87 | ok |
| 10 | Python | batch sizes | 1 | 8 | 8 | 1.580 | 1.572–1.586 | 4.80 | 0.3293 | 3060 | 90 | ok |
| 10 | Python | batch sizes | 1 | 9 | 9 | 1.816 | 1.809–1.817 | 4.80 | 0.3784 | 3196 | 86 | ok |
| 10 | Python | batch sizes | 1 | 10 | 10 | 2.050 | 2.049–2.060 | 4.80 | 0.4271 | 3346 | 89 | ok |
| 10 | Python | batch sizes | 1 | 11 | 11 | 2.249 | 2.246–2.250 | 4.80 | 0.4686 | 3478 | 90 | ok |
| 10 | Python | batch sizes | 1 | 12 | 12 | 2.500 | 2.497–2.500 | 4.80 | 0.5208 | 3614 | 87 | ok |
| 10 | Python | batch sizes | 1 | 13 | 13 | 2.758 | 2.754–2.758 | 4.80 | 0.5746 | 3756 | 89 | ok |
| 10 | Python | batch sizes | 1 | 14 | 14 | 3.037 | 3.024–3.083 | 4.80 | 0.6327 | 3902 | 92 | ok |
| 10 | Python | batch sizes | 1 | 15 | 15 | 3.290 | 3.289–3.318 | 4.80 | 0.6854 | 4034 | 88 | ok |
| 10 | Python | batch sizes | 1 | 16 | 16 | 3.571 | 3.571–3.573 | 4.80 | 0.7440 | 4169 | 90 | ok |
| 10 | Python | batch sizes | 1 | 17 | 17 | 3.807 | 3.801–3.810 | 4.80 | 0.7931 | 4313 | 90 | ok |
| 10 | Python | batch sizes | 1 | 18 | 18 | 4.130 | 4.128–4.131 | 4.80 | 0.8604 | 4448 | 92 | ok |
| 10 | Python | batch sizes | 1 | 19 | 19 | 4.386 | 4.382–4.389 | 4.80 | 0.9138 | 4588 | 91 | ok |
| 10 | Python | batch sizes | 1 | 20 | 20 | 4.636 | 4.632–4.637 | 4.80 | 0.9658 | 4725 | 88 | ok |
| 10 | Python | instances | 1 | 1 | 1 | 0.923 | 0.921–1.113 | 4.80 | 0.1923 | 4330 | 20 | ok |
| 10 | Python | instances | 2 | 1 | 2 | 1.040 | 1.035–1.049 | 4.80 | 0.2167 | 8655 | 90 | ok |
| 10 | Python | instances | 3 | 1 | 3 | 1.112 | 1.105–1.139 | 4.80 | 0.2316 | 12980 | 87 | ok |
| 10 | Python | instances | 4 | 1 | 4 | 1.078 | 1.011–1.145 | 4.80 | 0.2245 | 17306 | 95 | ok |
| 10 | Python | instances | 5 | 1 | 5 | 1.163 | 1.139–1.395 | 4.80 | 0.2422 | 21631 | 95 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 1 | 1 | 0.531 | 0.459–0.546 | 4.80 | 0.1105 | 2705 | 23 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 2 | 2 | 0.617 | 0.604–0.646 | 4.80 | 0.1285 | 2705 | 27 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 3 | 3 | 0.531 | 0.518–0.583 | 4.80 | 0.1107 | 2709 | 40 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 4 | 4 | 0.557 | 0.547–0.568 | 4.80 | 0.1160 | 2728 | 47 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 5 | 5 | 0.634 | 0.557–0.650 | 4.80 | 0.1320 | 2750 | 61 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 6 | 6 | 0.655 | 0.611–0.737 | 4.80 | 0.1364 | 2773 | 58 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 7 | 7 | 0.648 | 0.648–0.653 | 4.80 | 0.1351 | 2795 | 79 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 8 | 8 | 0.733 | 0.729–0.735 | 4.80 | 0.1527 | 2815 | 75 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 9 | 9 | 0.831 | 0.819–0.863 | 4.80 | 0.1730 | 2836 | 73 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 10 | 10 | 0.897 | 0.895–0.912 | 4.80 | 0.1868 | 2858 | 72 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 11 | 11 | 0.994 | 0.983–0.999 | 4.80 | 0.2070 | 2881 | 76 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 12 | 12 | 1.074 | 1.069–1.077 | 4.80 | 0.2236 | 2903 | 77 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 13 | 13 | 1.152 | 1.150–1.158 | 4.80 | 0.2400 | 2924 | 78 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 14 | 14 | 1.254 | 1.254–1.254 | 4.80 | 0.2613 | 2945 | 71 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 15 | 15 | 1.402 | 1.337–1.517 | 4.80 | 0.2921 | 2967 | 74 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 16 | 16 | 1.437 | 1.437–1.440 | 4.80 | 0.2994 | 2990 | 75 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 17 | 17 | 1.532 | 1.531–1.537 | 4.80 | 0.3191 | 3011 | 79 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 18 | 18 | 1.635 | 1.626–1.636 | 4.80 | 0.3406 | 3032 | 79 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 19 | 19 | 1.806 | 1.785–1.813 | 4.80 | 0.3763 | 3053 | 74 | ok |
| 10 | Python + FlashInfer | batch sizes | 1 | 20 | 20 | 1.818 | 1.809–1.827 | 4.80 | 0.3787 | 3075 | 78 | ok |
| 10 | Python + FlashInfer | instances | 1 | 1 | 1 | 0.476 | 0.435–0.536 | 4.80 | 0.0992 | 4994 | 24 | ok |
| 10 | Python + FlashInfer | instances | 2 | 1 | 2 | 0.456 | 0.456–0.506 | 4.80 | 0.0950 | 9983 | 90 | ok |
| 10 | Python + FlashInfer | instances | 3 | 1 | 3 | 0.509 | 0.477–0.509 | 4.80 | 0.1060 | 14972 | 89 | ok |
| 10 | Python + FlashInfer | instances | 4 | 1 | 4 | 0.541 | 0.530–0.573 | 4.80 | 0.1128 | 19962 | 92 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 1 | 1 | 0.122 | 0.121–0.122 | 4.80 | 0.0254 | 2993 | 77 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 2 | 2 | 0.215 | 0.208–0.216 | 4.80 | 0.0449 | 3004 | 71 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 3 | 3 | 0.290 | 0.289–0.290 | 4.80 | 0.0603 | 3017 | 70 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 4 | 4 | 0.389 | 0.378–0.389 | 4.80 | 0.0810 | 3050 | 74 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 5 | 5 | 0.459 | 0.453–0.474 | 4.80 | 0.0957 | 3084 | 75 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 6 | 6 | 0.541 | 0.540–0.542 | 4.80 | 0.1128 | 3120 | 80 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 7 | 7 | 0.644 | 0.639–0.653 | 4.80 | 0.1341 | 3154 | 71 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 8 | 8 | 0.766 | 0.736–0.772 | 4.80 | 0.1596 | 3176 | 74 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 9 | 9 | 0.807 | 0.805–0.813 | 4.80 | 0.1681 | 3201 | 73 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 10 | 10 | 0.890 | 0.887–0.893 | 4.80 | 0.1855 | 3227 | 74 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 11 | 11 | 0.984 | 0.982–1.017 | 4.80 | 0.2050 | 3255 | 79 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 12 | 12 | 1.071 | 1.066–1.075 | 4.80 | 0.2232 | 3281 | 81 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 13 | 13 | 1.151 | 1.149–1.154 | 4.80 | 0.2398 | 3303 | 72 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 14 | 14 | 1.256 | 1.250–1.319 | 4.80 | 0.2616 | 3328 | 73 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 15 | 15 | 1.332 | 1.326–1.406 | 4.80 | 0.2774 | 3353 | 80 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 16 | 16 | 1.427 | 1.426–1.430 | 4.80 | 0.2974 | 3381 | 78 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 17 | 17 | 1.509 | 1.506–1.511 | 4.80 | 0.3145 | 3407 | 74 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 18 | 18 | 1.616 | 1.616–1.628 | 4.80 | 0.3368 | 3430 | 79 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 19 | 19 | 1.688 | 1.686–1.688 | 4.80 | 0.3516 | 3454 | 78 | ok |
| 10 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 20 | 20 | 1.795 | 1.790–1.802 | 4.80 | 0.3740 | 3480 | 78 | ok |
| 10 | Python + FlashInfer + CUDA graphs | instances | 1 | 1 | 1 | 0.115 | 0.115–0.116 | 4.80 | 0.0239 | 4062 | 81 | ok |
| 10 | Python + FlashInfer + CUDA graphs | instances | 2 | 1 | 2 | 0.220 | 0.220–0.222 | 4.80 | 0.0459 | 8119 | 88 | ok |
| 10 | Python + FlashInfer + CUDA graphs | instances | 3 | 1 | 3 | 0.318 | 0.318–0.321 | 4.80 | 0.0662 | 12176 | 89 | ok |
| 10 | Python + FlashInfer + CUDA graphs | instances | 4 | 1 | 4 | 0.414 | 0.414–0.419 | 4.80 | 0.0863 | 16234 | 89 | ok |
| 10 | Python + FlashInfer + CUDA graphs | instances | 5 | 1 | 5 | 0.523 | 0.512–0.526 | 4.80 | 0.1089 | 20291 | 90 | ok |
| 10 | C++ | batch sizes | 1 | 1 | 1 | 0.381 | 0.378–0.381 | 4.57 | 0.0833 | 2474 | 59 | ok |
| 10 | C++ | batch sizes | 1 | 2 | 2 | 0.749 | 0.746–0.750 | 4.57 | 0.1640 | 2474 | 58 | ok |
| 10 | C++ | batch sizes | 1 | 3 | 3 | 1.120 | 1.119–1.123 | 4.57 | 0.2451 | 2474 | 60 | ok |
| 10 | C++ | batch sizes | 1 | 4 | 4 | 1.497 | 1.494–1.525 | 4.57 | 0.3276 | 2474 | 59 | ok |
| 10 | C++ | batch sizes | 1 | 5 | 5 | 1.895 | 1.876–1.916 | 4.57 | 0.4147 | 2474 | 59 | ok |
| 10 | C++ | batch sizes | 1 | 6 | 6 | 2.299 | 2.261–2.376 | 4.57 | 0.5030 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 7 | 7 | 2.625 | 2.622–2.648 | 4.57 | 0.5744 | 2474 | 62 | ok |
| 10 | C++ | batch sizes | 1 | 8 | 8 | 3.024 | 2.996–3.051 | 4.57 | 0.6618 | 2474 | 60 | ok |
| 10 | C++ | batch sizes | 1 | 9 | 9 | 3.382 | 3.371–3.406 | 4.57 | 0.7401 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 10 | 10 | 3.787 | 3.764–3.843 | 4.57 | 0.8287 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 11 | 11 | 4.130 | 4.118–4.132 | 4.57 | 0.9036 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 12 | 12 | 4.523 | 4.521–4.524 | 4.57 | 0.9897 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 13 | 13 | 4.874 | 4.864–4.890 | 4.57 | 1.0664 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 14 | 14 | 5.288 | 5.254–5.300 | 4.57 | 1.1572 | 2474 | 62 | ok |
| 10 | C++ | batch sizes | 1 | 15 | 15 | 5.634 | 5.624–5.684 | 4.57 | 1.2329 | 2474 | 62 | ok |
| 10 | C++ | batch sizes | 1 | 16 | 16 | 6.038 | 6.020–6.105 | 4.57 | 1.3213 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 17 | 17 | 6.452 | 6.433–6.691 | 4.57 | 1.4118 | 2474 | 61 | ok |
| 10 | C++ | batch sizes | 1 | 18 | 18 | 6.859 | 6.832–6.866 | 4.57 | 1.5009 | 2474 | 62 | ok |
| 10 | C++ | batch sizes | 1 | 19 | 19 | 7.186 | 7.172–7.210 | 4.57 | 1.5724 | 2474 | 62 | ok |
| 10 | C++ | batch sizes | 1 | 20 | 20 | 7.541 | 7.519–7.567 | 4.57 | 1.6501 | 2474 | 62 | ok |
| 10 | C++ | instances | 1 | 1 | 1 | 0.372 | 0.371–0.377 | 4.57 | 0.0815 | 2484 | 59 | ok |
| 10 | C++ | instances | 2 | 1 | 2 | 0.552 | 0.550–0.555 | 4.57 | 0.1207 | 4963 | 86 | ok |
| 10 | C++ | instances | 3 | 1 | 3 | 0.839 | 0.832–0.845 | 4.57 | 0.1835 | 7442 | 89 | ok |
| 10 | C++ | instances | 4 | 1 | 4 | 1.102 | 1.097–1.104 | 4.57 | 0.2410 | 9922 | 92 | ok |
| 10 | C++ | instances | 5 | 1 | 5 | 1.370 | 1.369–1.378 | 4.57 | 0.2998 | 12401 | 94 | ok |
| 10 | C++ | instances | 6 | 1 | 6 | 1.641 | 1.633–1.645 | 4.57 | 0.3590 | 14880 | 93 | ok |
| 10 | C++ | instances | 7 | 1 | 7 | 1.900 | 1.896–1.904 | 4.57 | 0.4158 | 17360 | 93 | ok |
| 10 | C++ | instances | 8 | 1 | 8 | 2.167 | 2.167–2.176 | 4.57 | 0.4743 | 19839 | 94 | ok |
| 10 | C++ | instances | 9 | 1 | 9 | 2.429 | 2.427–2.431 | 4.57 | 0.5315 | 22320 | 94 | ok |
| 15 | Python | batch sizes | 1 | 1 | 1 | 0.926 | 0.918–1.019 | 7.08 | 0.1308 | 2116 | 25 | ok |
| 15 | Python | batch sizes | 1 | 2 | 2 | 0.962 | 0.952–0.978 | 7.08 | 0.1358 | 2284 | 38 | ok |
| 15 | Python | batch sizes | 1 | 3 | 3 | 1.096 | 1.016–1.149 | 7.08 | 0.1548 | 2454 | 54 | ok |
| 15 | Python | batch sizes | 1 | 4 | 4 | 1.194 | 1.050–1.275 | 7.08 | 0.1687 | 2621 | 69 | ok |
| 15 | Python | batch sizes | 1 | 5 | 5 | 1.148 | 1.119–1.160 | 7.08 | 0.1621 | 2775 | 84 | ok |
| 15 | Python | batch sizes | 1 | 6 | 6 | 1.370 | 1.365–1.380 | 7.08 | 0.1935 | 2938 | 88 | ok |
| 15 | Python | batch sizes | 1 | 7 | 7 | 1.647 | 1.647–1.671 | 7.08 | 0.2327 | 3104 | 82 | ok |
| 15 | Python | batch sizes | 1 | 8 | 8 | 1.923 | 1.918–1.927 | 7.08 | 0.2716 | 3269 | 90 | ok |
| 15 | Python | batch sizes | 1 | 9 | 9 | 2.210 | 2.203–2.213 | 7.08 | 0.3121 | 3438 | 90 | ok |
| 15 | Python | batch sizes | 1 | 10 | 10 | 2.512 | 2.511–2.539 | 7.08 | 0.3547 | 3606 | 88 | ok |
| 15 | Python | batch sizes | 1 | 11 | 11 | 2.839 | 2.828–2.846 | 7.08 | 0.4010 | 3766 | 89 | ok |
| 15 | Python | batch sizes | 1 | 12 | 12 | 3.116 | 3.111–3.188 | 7.08 | 0.4401 | 3944 | 88 | ok |
| 15 | Python | batch sizes | 1 | 13 | 13 | 3.442 | 3.433–3.456 | 7.08 | 0.4862 | 4098 | 91 | ok |
| 15 | Python | batch sizes | 1 | 14 | 14 | 3.762 | 3.757–3.765 | 7.08 | 0.5313 | 4275 | 92 | ok |
| 15 | Python | batch sizes | 1 | 15 | 15 | 4.088 | 4.080–4.090 | 7.08 | 0.5774 | 4444 | 85 | ok |
| 15 | Python | batch sizes | 1 | 16 | 16 | 4.431 | 4.422–4.434 | 7.08 | 0.6259 | 4593 | 93 | ok |
| 15 | Python | batch sizes | 1 | 17 | 17 | 4.733 | 4.724–4.760 | 7.08 | 0.6685 | 4760 | 92 | ok |
| 15 | Python | batch sizes | 1 | 18 | 18 | 5.077 | 5.056–5.098 | 7.08 | 0.7172 | 4928 | 92 | ok |
| 15 | Python | batch sizes | 1 | 19 | 19 | 5.401 | 5.380–5.481 | 7.08 | 0.7628 | 5097 | 92 | ok |
| 15 | Python | batch sizes | 1 | 20 | 20 | 5.695 | 5.687–5.718 | 7.08 | 0.8043 | 5265 | 93 | ok |
| 15 | Python | instances | 1 | 1 | 1 | 0.972 | 0.869–1.110 | 7.08 | 0.1373 | 4330 | 26 | ok |
| 15 | Python | instances | 2 | 1 | 2 | 1.085 | 1.039–1.095 | 7.08 | 0.1533 | 8655 | 92 | ok |
| 15 | Python | instances | 3 | 1 | 3 | 1.053 | 1.026–1.150 | 7.08 | 0.1487 | 12980 | 92 | ok |
| 15 | Python | instances | 4 | 1 | 4 | 1.081 | 1.074–1.096 | 7.08 | 0.1527 | 17306 | 93 | ok |
| 15 | Python | instances | 5 | 1 | 5 | 1.190 | 1.187–1.194 | 7.08 | 0.1680 | 21631 | 95 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 1 | 1 | 0.464 | 0.459–0.471 | 7.08 | 0.0656 | 2757 | 27 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 2 | 2 | 0.567 | 0.517–0.628 | 7.08 | 0.0801 | 2756 | 36 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 3 | 3 | 0.526 | 0.517–0.526 | 7.08 | 0.0742 | 2756 | 47 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 4 | 4 | 0.618 | 0.584–0.646 | 7.08 | 0.0873 | 2773 | 58 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 5 | 5 | 0.733 | 0.656–0.768 | 7.08 | 0.1036 | 2804 | 73 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 6 | 6 | 0.679 | 0.679–0.684 | 7.08 | 0.0960 | 2836 | 76 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 7 | 7 | 0.797 | 0.796–0.799 | 7.08 | 0.1125 | 2870 | 73 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 8 | 8 | 0.908 | 0.904–0.914 | 7.08 | 0.1283 | 2903 | 76 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 9 | 9 | 1.020 | 1.015–1.020 | 7.08 | 0.1441 | 2933 | 79 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 10 | 10 | 1.199 | 1.161–1.259 | 7.08 | 0.1694 | 2966 | 73 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 11 | 11 | 1.291 | 1.278–1.323 | 7.08 | 0.1824 | 3000 | 70 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 12 | 12 | 1.356 | 1.352–1.407 | 7.08 | 0.1916 | 3031 | 76 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 13 | 13 | 1.473 | 1.467–1.492 | 7.08 | 0.2081 | 3063 | 77 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 14 | 14 | 1.591 | 1.589–1.595 | 7.08 | 0.2247 | 3097 | 78 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 15 | 15 | 1.711 | 1.703–1.717 | 7.08 | 0.2417 | 3128 | 79 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 16 | 16 | 1.828 | 1.826–1.830 | 7.08 | 0.2582 | 3160 | 80 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 17 | 17 | 1.955 | 1.946–1.957 | 7.08 | 0.2761 | 3194 | 77 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 18 | 18 | 2.069 | 2.055–2.073 | 7.08 | 0.2923 | 3225 | 81 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 19 | 19 | 2.178 | 2.171–2.187 | 7.08 | 0.3077 | 3257 | 74 | ok |
| 15 | Python + FlashInfer | batch sizes | 1 | 20 | 20 | 2.288 | 2.282–2.295 | 7.08 | 0.3231 | 3291 | 79 | ok |
| 15 | Python + FlashInfer | instances | 1 | 1 | 1 | 0.432 | 0.430–0.432 | 7.08 | 0.0611 | 4994 | 27 | ok |
| 15 | Python + FlashInfer | instances | 2 | 1 | 2 | 0.536 | 0.512–0.572 | 7.08 | 0.0757 | 9983 | 87 | ok |
| 15 | Python + FlashInfer | instances | 3 | 1 | 3 | 0.573 | 0.543–0.573 | 7.08 | 0.0809 | 14972 | 89 | ok |
| 15 | Python + FlashInfer | instances | 4 | 1 | 4 | 0.523 | 0.522–0.532 | 7.08 | 0.0738 | 19962 | 91 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 1 | 1 | 0.146 | 0.146–0.147 | 7.08 | 0.0207 | 3095 | 74 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 2 | 2 | 0.247 | 0.246–0.248 | 7.08 | 0.0349 | 3100 | 71 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 3 | 3 | 0.343 | 0.342–0.344 | 7.08 | 0.0484 | 3105 | 71 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 4 | 4 | 0.480 | 0.480–0.481 | 7.08 | 0.0678 | 3128 | 71 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 5 | 5 | 0.581 | 0.574–0.609 | 7.08 | 0.0821 | 3163 | 75 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 6 | 6 | 0.682 | 0.675–0.684 | 7.08 | 0.0963 | 3201 | 75 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 7 | 7 | 0.794 | 0.793–0.801 | 7.08 | 0.1122 | 3240 | 74 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 8 | 8 | 0.900 | 0.896–0.903 | 7.08 | 0.1271 | 3280 | 75 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 9 | 9 | 1.011 | 1.011–1.018 | 7.08 | 0.1428 | 3314 | 79 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 10 | 10 | 1.120 | 1.115–1.126 | 7.08 | 0.1582 | 3353 | 77 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 11 | 11 | 1.243 | 1.242–1.260 | 7.08 | 0.1755 | 3393 | 77 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 12 | 12 | 1.350 | 1.345–1.351 | 7.08 | 0.1906 | 3428 | 82 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 13 | 13 | 1.480 | 1.465–1.496 | 7.08 | 0.2091 | 3466 | 71 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 14 | 14 | 1.586 | 1.578–1.587 | 7.08 | 0.2241 | 3506 | 77 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 15 | 15 | 1.695 | 1.689–1.700 | 7.08 | 0.2394 | 3542 | 78 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 16 | 16 | 1.825 | 1.810–1.854 | 7.08 | 0.2578 | 3579 | 78 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 17 | 17 | 1.934 | 1.924–1.937 | 7.08 | 0.2732 | 3619 | 80 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 18 | 18 | 2.048 | 2.040–2.049 | 7.08 | 0.2893 | 3655 | 81 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 19 | 19 | 2.186 | 2.180–2.195 | 7.08 | 0.3088 | 3692 | 73 | ok |
| 15 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 20 | 20 | 2.280 | 2.275–2.294 | 7.08 | 0.3221 | 3732 | 83 | ok |
| 15 | Python + FlashInfer + CUDA graphs | instances | 1 | 1 | 1 | 0.139 | 0.138–0.140 | 7.08 | 0.0196 | 4062 | 77 | ok |
| 15 | Python + FlashInfer + CUDA graphs | instances | 2 | 1 | 2 | 0.268 | 0.266–0.269 | 7.08 | 0.0379 | 8119 | 88 | ok |
| 15 | Python + FlashInfer + CUDA graphs | instances | 3 | 1 | 3 | 0.390 | 0.390–0.409 | 7.08 | 0.0551 | 12176 | 93 | ok |
| 15 | Python + FlashInfer + CUDA graphs | instances | 4 | 1 | 4 | 0.510 | 0.510–0.511 | 7.08 | 0.0720 | 16234 | 89 | ok |
| 15 | Python + FlashInfer + CUDA graphs | instances | 5 | 1 | 5 | 0.631 | 0.630–0.636 | 7.08 | 0.0892 | 20291 | 93 | ok |
| 15 | C++ | batch sizes | 1 | 1 | 1 | 0.462 | 0.458–0.466 | 7.08 | 0.0653 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 2 | 2 | 0.936 | 0.920–0.963 | 7.08 | 0.1323 | 2492 | 55 | ok |
| 15 | C++ | batch sizes | 1 | 3 | 3 | 1.374 | 1.373–1.392 | 7.08 | 0.1940 | 2492 | 55 | ok |
| 15 | C++ | batch sizes | 1 | 4 | 4 | 1.833 | 1.833–1.903 | 7.08 | 0.2590 | 2492 | 54 | ok |
| 15 | C++ | batch sizes | 1 | 5 | 5 | 2.309 | 2.301–2.347 | 7.08 | 0.3262 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 6 | 6 | 2.671 | 2.660–2.842 | 7.08 | 0.3773 | 2492 | 55 | ok |
| 15 | C++ | batch sizes | 1 | 7 | 7 | 3.212 | 3.206–3.216 | 7.08 | 0.4537 | 2492 | 55 | ok |
| 15 | C++ | batch sizes | 1 | 8 | 8 | 3.658 | 3.655–3.731 | 7.08 | 0.5167 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 9 | 9 | 4.128 | 4.123–4.185 | 7.08 | 0.5831 | 2492 | 55 | ok |
| 15 | C++ | batch sizes | 1 | 10 | 10 | 4.584 | 4.581–4.590 | 7.08 | 0.6475 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 11 | 11 | 5.052 | 5.042–5.117 | 7.08 | 0.7135 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 12 | 12 | 5.514 | 5.497–5.565 | 7.08 | 0.7789 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 13 | 13 | 5.922 | 5.902–5.947 | 7.08 | 0.8364 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 14 | 14 | 6.424 | 6.420–6.436 | 7.08 | 0.9074 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 15 | 15 | 6.890 | 6.884–6.898 | 7.08 | 0.9732 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 16 | 16 | 7.356 | 7.340–7.408 | 7.08 | 1.0389 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 17 | 17 | 7.816 | 7.775–7.821 | 7.08 | 1.1039 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 18 | 18 | 8.263 | 8.234–8.282 | 7.08 | 1.1671 | 2492 | 57 | ok |
| 15 | C++ | batch sizes | 1 | 19 | 19 | 8.713 | 8.699–8.755 | 7.08 | 1.2306 | 2492 | 56 | ok |
| 15 | C++ | batch sizes | 1 | 20 | 20 | 9.188 | 9.159–9.232 | 7.08 | 1.2977 | 2492 | 57 | ok |
| 15 | C++ | instances | 1 | 1 | 1 | 0.446 | 0.444–0.456 | 7.08 | 0.0630 | 2502 | 55 | ok |
| 15 | C++ | instances | 2 | 1 | 2 | 0.604 | 0.602–0.609 | 7.08 | 0.0853 | 4999 | 84 | ok |
| 15 | C++ | instances | 3 | 1 | 3 | 0.928 | 0.926–0.930 | 7.08 | 0.1311 | 7496 | 89 | ok |
| 15 | C++ | instances | 4 | 1 | 4 | 1.225 | 1.222–1.243 | 7.08 | 0.1730 | 9994 | 92 | ok |
| 15 | C++ | instances | 5 | 1 | 5 | 1.516 | 1.514–1.532 | 7.08 | 0.2142 | 12491 | 91 | ok |
| 15 | C++ | instances | 6 | 1 | 6 | 1.823 | 1.818–1.825 | 7.08 | 0.2574 | 14988 | 90 | ok |
| 15 | C++ | instances | 7 | 1 | 7 | 2.116 | 2.107–2.117 | 7.08 | 0.2989 | 17486 | 91 | ok |
| 15 | C++ | instances | 8 | 1 | 8 | 2.408 | 2.403–2.412 | 7.08 | 0.3401 | 19983 | 94 | ok |
| 15 | C++ | instances | 9 | 1 | 9 | 2.698 | 2.696–2.701 | 7.08 | 0.3811 | 22482 | 95 | ok |
| 20 | Python | batch sizes | 1 | 1 | 1 | 1.037 | 0.984–1.205 | 10.92 | 0.0950 | 2159 | 32 | ok |
| 20 | Python | batch sizes | 1 | 2 | 2 | 1.089 | 1.059–1.198 | 10.92 | 0.0997 | 2372 | 50 | ok |
| 20 | Python | batch sizes | 1 | 3 | 3 | 1.079 | 1.063–1.136 | 10.92 | 0.0988 | 2586 | 65 | ok |
| 20 | Python | batch sizes | 1 | 4 | 4 | 1.197 | 1.196–1.199 | 10.92 | 0.1096 | 2800 | 61 | ok |
| 20 | Python | batch sizes | 1 | 5 | 5 | 1.556 | 1.555–1.561 | 10.92 | 0.1425 | 3013 | 85 | ok |
| 20 | Python | batch sizes | 1 | 6 | 6 | 1.911 | 1.911–1.932 | 10.92 | 0.1750 | 3226 | 84 | ok |
| 20 | Python | batch sizes | 1 | 7 | 7 | 2.265 | 2.264–2.295 | 10.92 | 0.2074 | 3440 | 92 | ok |
| 20 | Python | batch sizes | 1 | 8 | 8 | 2.659 | 2.652–2.660 | 10.92 | 0.2435 | 3653 | 87 | ok |
| 20 | Python | batch sizes | 1 | 9 | 9 | 3.096 | 3.081–3.142 | 10.92 | 0.2835 | 3866 | 90 | ok |
| 20 | Python | batch sizes | 1 | 10 | 10 | 3.468 | 3.467–3.472 | 10.92 | 0.3176 | 4079 | 91 | ok |
| 20 | Python | batch sizes | 1 | 11 | 11 | 3.889 | 3.882–3.893 | 10.92 | 0.3561 | 4293 | 92 | ok |
| 20 | Python | batch sizes | 1 | 12 | 12 | 4.344 | 4.342–4.345 | 10.92 | 0.3978 | 4505 | 90 | ok |
| 20 | Python | batch sizes | 1 | 13 | 13 | 4.763 | 4.760–4.824 | 10.92 | 0.4361 | 4718 | 92 | ok |
| 20 | Python | batch sizes | 1 | 14 | 14 | 5.174 | 5.165–5.176 | 10.92 | 0.4738 | 4931 | 93 | ok |
| 20 | Python | batch sizes | 1 | 15 | 15 | 5.593 | 5.592–5.643 | 10.92 | 0.5122 | 5146 | 92 | ok |
| 20 | Python | batch sizes | 1 | 16 | 16 | 5.964 | 5.956–6.035 | 10.92 | 0.5462 | 5359 | 92 | ok |
| 20 | Python | batch sizes | 1 | 17 | 17 | 6.431 | 6.398–6.434 | 10.92 | 0.5890 | 5572 | 90 | ok |
| 20 | Python | batch sizes | 1 | 18 | 18 | 6.793 | 6.792–6.798 | 10.92 | 0.6221 | 5784 | 92 | ok |
| 20 | Python | batch sizes | 1 | 19 | 19 | 7.166 | 7.166–7.197 | 10.92 | 0.6562 | 5998 | 92 | ok |
| 20 | Python | batch sizes | 1 | 20 | 20 | 7.554 | 7.552–7.560 | 10.92 | 0.6918 | 6211 | 91 | ok |
| 20 | Python | instances | 1 | 1 | 1 | 0.931 | 0.921–0.934 | 10.92 | 0.0853 | 4330 | 33 | ok |
| 20 | Python | instances | 2 | 1 | 2 | 1.005 | 1.004–1.078 | 10.92 | 0.0921 | 8655 | 90 | ok |
| 20 | Python | instances | 3 | 1 | 3 | 1.089 | 1.063–1.096 | 10.92 | 0.0997 | 12980 | 91 | ok |
| 20 | Python | instances | 4 | 1 | 4 | 1.311 | 1.310–1.315 | 10.92 | 0.1201 | 17306 | 95 | ok |
| 20 | Python | instances | 5 | 1 | 5 | 1.631 | 1.631–1.634 | 10.92 | 0.1494 | 21631 | 92 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 1 | 1 | 0.553 | 0.477–0.560 | 10.92 | 0.0506 | 2790 | 32 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 2 | 2 | 0.508 | 0.506–0.515 | 10.92 | 0.0465 | 2790 | 45 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 3 | 3 | 0.664 | 0.603–0.675 | 10.91 | 0.0609 | 2795 | 40 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 4 | 4 | 0.661 | 0.636–0.710 | 10.92 | 0.0605 | 2846 | 76 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 5 | 5 | 0.790 | 0.790–0.825 | 10.92 | 0.0723 | 2894 | 75 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 6 | 6 | 0.942 | 0.935–0.949 | 10.92 | 0.0863 | 2944 | 68 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 7 | 7 | 1.105 | 1.093–1.140 | 10.92 | 0.1012 | 2995 | 77 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 8 | 8 | 1.256 | 1.251–1.259 | 10.92 | 0.1150 | 3048 | 79 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 9 | 9 | 1.402 | 1.402–1.403 | 10.92 | 0.1284 | 3097 | 80 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 10 | 10 | 1.574 | 1.567–1.589 | 10.92 | 0.1441 | 3147 | 82 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 11 | 11 | 1.733 | 1.725–1.737 | 10.92 | 0.1587 | 3196 | 85 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 12 | 12 | 1.874 | 1.867–1.875 | 10.92 | 0.1716 | 3248 | 80 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 13 | 13 | 2.051 | 2.046–2.069 | 10.92 | 0.1878 | 3297 | 82 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 14 | 14 | 2.208 | 2.205–2.252 | 10.92 | 0.2022 | 3350 | 80 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 15 | 15 | 2.383 | 2.382–2.420 | 10.92 | 0.2183 | 3399 | 80 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 16 | 16 | 2.569 | 2.559–2.610 | 10.92 | 0.2353 | 3449 | 79 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 17 | 17 | 2.683 | 2.681–2.686 | 10.92 | 0.2457 | 3500 | 86 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 18 | 18 | 2.852 | 2.851–2.860 | 10.92 | 0.2612 | 3550 | 82 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 19 | 19 | 3.024 | 3.019–3.025 | 10.92 | 0.2769 | 3602 | 76 | ok |
| 20 | Python + FlashInfer | batch sizes | 1 | 20 | 20 | 3.181 | 3.177–3.191 | 10.92 | 0.2913 | 3652 | 81 | ok |
| 20 | Python + FlashInfer | instances | 1 | 1 | 1 | 0.497 | 0.436–0.513 | 10.92 | 0.0455 | 4994 | 34 | ok |
| 20 | Python + FlashInfer | instances | 2 | 1 | 2 | 0.526 | 0.520–0.528 | 10.92 | 0.0481 | 9983 | 83 | ok |
| 20 | Python + FlashInfer | instances | 3 | 1 | 3 | 0.540 | 0.527–0.772 | 10.92 | 0.0495 | 14972 | 78 | ok |
| 20 | Python + FlashInfer | instances | 4 | 1 | 4 | 0.651 | 0.650–0.652 | 10.92 | 0.0596 | 19962 | 91 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 1 | 1 | 0.172 | 0.172–0.172 | 10.92 | 0.0157 | 3131 | 71 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 2 | 2 | 0.315 | 0.312–0.315 | 10.92 | 0.0288 | 3139 | 74 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 3 | 3 | 0.507 | 0.506–0.507 | 10.91 | 0.0465 | 3154 | 75 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 4 | 4 | 0.640 | 0.635–0.641 | 10.92 | 0.0586 | 3212 | 77 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 5 | 5 | 0.792 | 0.785–0.811 | 10.92 | 0.0725 | 3269 | 77 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 6 | 6 | 0.933 | 0.929–0.951 | 10.92 | 0.0854 | 3327 | 74 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 7 | 7 | 1.093 | 1.088–1.179 | 10.92 | 0.1001 | 3387 | 83 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 8 | 8 | 1.251 | 1.245–1.260 | 10.92 | 0.1146 | 3449 | 74 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 9 | 9 | 1.437 | 1.402–1.457 | 10.92 | 0.1316 | 3506 | 80 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 10 | 10 | 1.556 | 1.552–1.557 | 10.92 | 0.1425 | 3564 | 80 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 11 | 11 | 1.711 | 1.709–1.750 | 10.92 | 0.1567 | 3622 | 80 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 12 | 12 | 1.874 | 1.868–1.878 | 10.92 | 0.1716 | 3681 | 85 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 13 | 13 | 2.043 | 2.031–2.049 | 10.92 | 0.1871 | 3739 | 82 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 14 | 14 | 2.197 | 2.187–2.199 | 10.92 | 0.2012 | 3801 | 80 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 15 | 15 | 2.374 | 2.359–2.393 | 10.92 | 0.2174 | 3859 | 82 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 16 | 16 | 2.526 | 2.524–2.531 | 10.92 | 0.2314 | 3916 | 78 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 17 | 17 | 2.682 | 2.676–2.688 | 10.92 | 0.2456 | 3976 | 78 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 18 | 18 | 2.848 | 2.847–2.849 | 10.92 | 0.2608 | 4034 | 82 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 19 | 19 | 3.023 | 3.022–3.039 | 10.92 | 0.2768 | 4096 | 81 | ok |
| 20 | Python + FlashInfer + CUDA graphs | batch sizes | 1 | 20 | 20 | 3.173 | 3.173–3.189 | 10.92 | 0.2906 | 4154 | 82 | ok |
| 20 | Python + FlashInfer + CUDA graphs | instances | 1 | 1 | 1 | 0.171 | 0.170–0.171 | 10.92 | 0.0156 | 4062 | 76 | ok |
| 20 | Python + FlashInfer + CUDA graphs | instances | 2 | 1 | 2 | 0.334 | 0.332–0.335 | 10.92 | 0.0306 | 8119 | 86 | ok |
| 20 | Python + FlashInfer + CUDA graphs | instances | 3 | 1 | 3 | 0.482 | 0.482–0.482 | 10.92 | 0.0441 | 12176 | 89 | ok |
| 20 | Python + FlashInfer + CUDA graphs | instances | 4 | 1 | 4 | 0.626 | 0.626–0.627 | 10.92 | 0.0573 | 16234 | 90 | ok |
| 20 | Python + FlashInfer + CUDA graphs | instances | 5 | 1 | 5 | 0.779 | 0.776–0.779 | 10.92 | 0.0713 | 20291 | 90 | ok |
| 20 | C++ | batch sizes | 1 | 1 | 1 | 0.687 | 0.686–0.692 | 10.91 | 0.0630 | 2526 | 59 | ok |
| 20 | C++ | batch sizes | 1 | 2 | 2 | 1.381 | 1.372–1.419 | 10.91 | 0.1266 | 2526 | 60 | ok |
| 20 | C++ | batch sizes | 1 | 3 | 3 | 2.078 | 2.056–2.132 | 10.91 | 0.1905 | 2526 | 59 | ok |
| 20 | C++ | batch sizes | 1 | 4 | 4 | 2.798 | 2.749–2.809 | 10.91 | 0.2565 | 2526 | 59 | ok |
| 20 | C++ | batch sizes | 1 | 5 | 5 | 3.461 | 3.432–3.524 | 10.91 | 0.3172 | 2526 | 60 | ok |
| 20 | C++ | batch sizes | 1 | 6 | 6 | 4.125 | 4.115–4.133 | 10.91 | 0.3781 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 7 | 7 | 4.807 | 4.801–4.810 | 10.91 | 0.4406 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 8 | 8 | 5.530 | 5.489–5.561 | 10.91 | 0.5069 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 9 | 9 | 6.258 | 6.207–6.284 | 10.91 | 0.5736 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 10 | 10 | 6.948 | 6.895–6.985 | 10.91 | 0.6368 | 2526 | 60 | ok |
| 20 | C++ | batch sizes | 1 | 11 | 11 | 7.558 | 7.536–7.581 | 10.91 | 0.6928 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 12 | 12 | 8.267 | 8.259–8.352 | 10.91 | 0.7577 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 13 | 13 | 8.960 | 8.947–8.973 | 10.91 | 0.8212 | 2526 | 62 | ok |
| 20 | C++ | batch sizes | 1 | 14 | 14 | 9.645 | 9.621–9.659 | 10.91 | 0.8841 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 15 | 15 | 10.398 | 10.348–10.406 | 10.91 | 0.9531 | 2526 | 62 | ok |
| 20 | C++ | batch sizes | 1 | 16 | 16 | 11.051 | 11.047–11.129 | 10.91 | 1.0129 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 17 | 17 | 11.724 | 11.677–11.821 | 10.91 | 1.0746 | 2526 | 62 | ok |
| 20 | C++ | batch sizes | 1 | 18 | 18 | 12.420 | 12.416–12.634 | 10.91 | 1.1384 | 2526 | 61 | ok |
| 20 | C++ | batch sizes | 1 | 19 | 19 | 13.119 | 13.079–13.170 | 10.91 | 1.2025 | 2526 | 62 | ok |
| 20 | C++ | batch sizes | 1 | 20 | 20 | 13.816 | 13.784–13.863 | 10.91 | 1.2663 | 2526 | 61 | ok |
| 20 | C++ | instances | 1 | 1 | 1 | 0.689 | 0.688–0.696 | 10.91 | 0.0631 | 2536 | 64 | ok |
| 20 | C++ | instances | 2 | 1 | 2 | 0.962 | 0.961–0.971 | 10.91 | 0.0881 | 5067 | 89 | ok |
| 20 | C++ | instances | 3 | 1 | 3 | 1.467 | 1.454–1.473 | 10.91 | 0.1345 | 7598 | 94 | ok |
| 20 | C++ | instances | 4 | 1 | 4 | 1.929 | 1.920–1.949 | 10.91 | 0.1768 | 10130 | 94 | ok |
| 20 | C++ | instances | 5 | 1 | 5 | 2.418 | 2.416–2.419 | 10.91 | 0.2217 | 12661 | 94 | ok |
| 20 | C++ | instances | 6 | 1 | 6 | 2.895 | 2.881–2.898 | 10.91 | 0.2654 | 15192 | 95 | ok |
| 20 | C++ | instances | 7 | 1 | 7 | 3.360 | 3.359–3.370 | 10.91 | 0.3080 | 17724 | 96 | ok |
| 20 | C++ | instances | 8 | 1 | 8 | 3.835 | 3.834–3.836 | 10.91 | 0.3515 | 20255 | 97 | ok |
| 20 | C++ | instances | 9 | 1 | 9 | 4.296 | 4.294–4.302 | 10.91 | 0.3937 | 22788 | 97 | ok |

Raw data: `results.jsonl`, `instances.jsonl`. Log: `benchmark_batch.log`.
