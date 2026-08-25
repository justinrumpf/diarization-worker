# Diarization serverless worker (faster-whisper large-v3 + pyannote 3.1).
# Base matches examples/speech-diarization (CUDA 12.4 / torch 2.4.1).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Torch stack pinned to the base image's CUDA (must be first). torchvision is
#    pinned to the version that matches torch 2.4.1 — pyannote pulls torchvision in,
#    and a mismatched one causes "torchvision has no attribute 'extension'".
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Worker deps. (pyannote's chain — speechbrain / pytorch-metric-learning /
#    lightning — can pull a mismatched torch/torchvision from PyPI over the cu124
#    pins above, which is what causes "torchvision has no attribute 'extension'".)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# 3. pyannote 3.x needs use_auth_token support.
RUN pip install --no-cache-dir "huggingface_hub<0.24.0"

# 4. FINAL, AUTHORITATIVE torch stack: force the matched cu124 trio back with
#    --no-deps so the dependency resolver can't touch anything else. This must win.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 5. Restore cuDNN for CUDA 12.4 LAST (the torch reinstall above may re-pull a
#    different bundled cuDNN).
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

COPY handler.py .

# faster-whisper large-v3 + pyannote download on first job (cold start). To cut
# cold starts, attach a RunPod network volume mounted at /runpod-volume and set
# HF_HOME=/runpod-volume/hf so the model cache persists across worker starts.
CMD ["python3", "-u", "handler.py"]
