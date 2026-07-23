#!/usr/bin/env bash
set -e

# Run from the audio.cpp repository root.

DEVICE=0
MODEL="models/OmniVoice"

TEXT="আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?"
REF_WAV="ref_audios/ref_audio_02.wav"
REF_TEXT="আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?"

LANGUAGE="Bengali"
SEED=42
STEPS=20
GUIDANCE_SCALE=2.0
SPEED=1.2

GENERATOR_WEIGHT_TYPE="f16"
AUDIO_TOKENIZER_WEIGHT_TYPE="f16"

OUTPUT="omnivoice_python_cpp_simple/outputs/cpp_output.wav"

mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
    --task tts \
    --family omnivoice \
    --model "$MODEL" \
    --backend cuda \
    --device "$DEVICE" \
    --text "$TEXT" \
    --voice-ref "$REF_WAV" \
    --reference-text "$REF_TEXT" \
    --language "$LANGUAGE" \
    --seed "$SEED" \
    --num-inference-steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --request-option "speed=$SPEED" \
    --session-option "omnivoice.generator_weight_type=$GENERATOR_WEIGHT_TYPE" \
    --session-option "omnivoice.audio_tokenizer_weight_type=$AUDIO_TOKENIZER_WEIGHT_TYPE" \
    --out "$OUTPUT" \
    --log

echo
echo "C++ output: $OUTPUT"
