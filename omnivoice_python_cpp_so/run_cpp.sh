#!/usr/bin/env bash
set -e

DEVICE=0

# Direct C++ baseline using the same main settings as test_so.py.
# Run from the audio.cpp repository root.

CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
    --task tts \
    --family omnivoice \
    --model models/OmniVoice \
    --backend cuda \
    --device $DEVICE \
    --text "আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?" \
    --voice-ref ref_audios/ref_audio_02.wav \
    --reference-text "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?" \
    --language Bengali \
    --seed 42 \
    --num-inference-steps 20 \
    --guidance-scale 2.0 \
    --request-option speed=1.2 \
    --session-option omnivoice.generator_weight_type=f16 \
    --session-option omnivoice.audio_tokenizer_weight_type=f16 \
    --out omnivoice_python_cpp_so/outputs/cpp_output.wav \
    --log

echo
echo "C++ output: omnivoice_python_cpp_so/outputs/cpp_output.wav"
