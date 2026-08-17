conda create -n audiocpp312 python=3.12 -y
conda activate audiocpp312

pip install torch safetensors PyYAML

apt update
apt install -y cmake build-essential git

git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp

cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_cli

python tools/model_manager.py install omnivoice

python tools/model_manager.py info omnivoice

build/bin/audiocpp_cli --help --family omnivoice

build/bin/audiocpp_cli --help --model models/OmniVoice

# Voice Clone
CUDA_VISIBLE_DEVICES=0 build/bin/audiocpp_cli \
  --task tts \
  --family omnivoice \
  --model models/OmniVoice \
  --backend cuda \
  --text "আমি একটা টিটিএস মডেল, নাম OmniVoice। কিভাবে আপনাকে সাহায্য করতে পারি?" \
  --voice-ref ./ref_audios/ref_audio_02.wav \
  --reference-text "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?" \
  --language Bengali \
  --session-option num_inference_steps=16 \
  --load-option weight_type=f16 \
  --out omnivoice_f16_step16_output.wav \
  --log

CUDA_VISIBLE_DEVICES=0 build/bin/audiocpp_cli \
  --task tts \
  --family omnivoice \
  --model models/OmniVoice \
  --backend cuda \
  --batch-text-file prompts.txt \
  --voice-ref ./ref_audios/ref_audio_02.wav \
  --reference-text "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?" \
  --session-option num_inference_steps=32 \
  --load-option weight_type=f16 \
  --out-dir batch_output/ \
  --log

chmod +x benchmark_audiocpp_omnivoice_clone.sh
./benchmark_audiocpp_omnivoice_clone.sh

chmod +x benchmark_audiocpp_omnivoice_clone_batch.sh
./benchmark_audiocpp_omnivoice_clone_batch.sh



CUDA_VISIBLE_DEVICES=0 build/bin/audiocpp_cli \
  --task tts \
  --family omnivoice \
  --model models/OmniVoice \
  --backend cuda \
  --text "আমি একটা টিটিএস মডেল, নাম OmniVoice। এটা দ্বিতীয় লাইন। এটা তৃতীয় লাইন। এটা চতুর্থ লাইন।" \
  --voice-ref ./ref_audios/ref_audio_02.wav \
  --reference-text "আমি যে পরিমাণ তহবিল স্থানান্তর করতে পারি, তার কি সর্বোচ্চ সীমা আছে?" \
  --language Bengali \
  --session-option num_inference_steps=1 \
  --load-option weight_type=f16 \
  --out omnivoice_f16_step01_output.wav \
  --log
