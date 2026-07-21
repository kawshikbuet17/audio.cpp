#!/bin/bash
set -e

DEVICE="0"
LANG="Bengali"
SEED="42"
STEPS_LIST=("1" "2" "4" "8" "16" "32" "64" "128")
WEIGHT_TYPES=("native" "f32" "f16" "bf16" "q8_0")
REF_WAV="./ref_audios/ref_audio_02.wav"
REF_TEXT="আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?"
TEXT="আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?"
OUT_DIR="./outputs/single_infer"
LOG_FILE="benchmark_audiocpp_omnivoice_clone.log"

mkdir -p "$OUT_DIR"

# Redirect ALL output to global log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Started: $(date)"
echo "Running all audio.cpp OmniVoice combinations"
echo "Device: cuda:$DEVICE | Language: $LANG | Seed: $SEED"
echo "Global Log: $LOG_FILE"
echo "=========================================="
echo ""

TOTAL=$(( ${#STEPS_LIST[@]} * ${#WEIGHT_TYPES[@]} ))
COUNT=0

for steps in "${STEPS_LIST[@]}"; do
    for wt in "${WEIGHT_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        OUT_NAME="audiocpp_omnivoice_clone_${wt}_step${steps}.wav"

        echo "[$COUNT/$TOTAL] weight_type=$wt | steps=$steps -> $OUT_NAME"
        echo "  Started at: $(date +%H:%M:%S)"

        CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
            --task tts \
            --family omnivoice \
            --model models/OmniVoice \
            --backend cuda \
            --device "$DEVICE" \
            --text "$TEXT" \
            --voice-ref "$REF_WAV" \
            --reference-text "$REF_TEXT" \
            --language "$LANG" \
            --seed "$SEED" \
            --num-inference-steps "$steps" \
            --load-option weight_type="$wt" \
            --out "${OUT_DIR}/${OUT_NAME}" \
            --log

        echo "  Completed at: $(date +%H:%M:%S)"
        echo "  Output: ${OUT_DIR}/${OUT_NAME}"
        echo ""
    done
done

echo "=========================================="
echo "All $TOTAL combinations completed!"
echo "Finished: $(date)"
echo "Outputs in: $OUT_DIR"
echo "Global Log: $LOG_FILE"
echo "=========================================="