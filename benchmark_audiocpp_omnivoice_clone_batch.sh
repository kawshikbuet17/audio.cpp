#!/bin/bash
set -e

# Config
DEVICE="0"
LANG="Bengali"
SEED="42"
BATCH_FILE="prompts.txt"
REF_WAV="./ref_audios/ref_audio_02.wav"
REF_TEXT="আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?"
LOG_FILE="benchmark_audiocpp_omnivoice_clone_batch.log"

# Arrays
STEPS_LIST=("1" "2" "4" "8" "16" "32" "64" "128")
WEIGHT_TYPES=("f16" "f32" "bf16" "q8_0")

# Redirect ALL output to global log
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "=========================================="
echo "Started: $(date)"
echo "audio.cpp OmniVoice Batch Benchmark"
echo "Device: cuda:$DEVICE | Batch: $BATCH_FILE | Seed: $SEED"
echo "Global Log: $LOG_FILE"
echo "=========================================="
echo ""

TOTAL=$(( ${#STEPS_LIST[@]} * ${#WEIGHT_TYPES[@]} ))
COUNT=0

for steps in "${STEPS_LIST[@]}"; do
    for wt in "${WEIGHT_TYPES[@]}"; do
        COUNT=$((COUNT + 1))
        OUT_DIR="./outputs/batch_infer/audiocpp_omnivoice_clone_batch_${wt}_step${steps}"

        echo "[$COUNT/$TOTAL] weight_type=$wt | steps=$steps -> $OUT_DIR"
        echo "  Started at: $(date +%H:%M:%S)"

        mkdir -p "$OUT_DIR"

        CUDA_VISIBLE_DEVICES=$DEVICE build/bin/audiocpp_cli \
            --task tts \
            --family omnivoice \
            --model models/OmniVoice \
            --backend cuda \
            --device "$DEVICE" \
            --batch-text-file "$BATCH_FILE" \
            --voice-ref "$REF_WAV" \
            --reference-text "$REF_TEXT" \
            --language "$LANG" \
            --seed "$SEED" \
            --num-inference-steps "$steps" \
            --load-option weight_type="$wt" \
            --out-dir "$OUT_DIR" \
            --log

        echo "  Completed at: $(date +%H:%M:%S)"
        echo "  Output dir: $OUT_DIR"
        echo ""
    done
done

echo "=========================================="
echo "All $TOTAL batch combinations completed!"
echo "Finished: $(date)"
echo "Global Log: $LOG_FILE"
echo "=========================================="