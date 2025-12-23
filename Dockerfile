# Use Python 3.10 slim image (same as your working setup)
FROM python:3.10-slim

# Prevent interactive prompts and set Python unbuffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    g++ \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app

# Environment Config for CPU-Only (prevents TensorFlow/Essentia from looking for GPU)
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install build dependencies FIRST (needed for madmom compilation)
RUN pip install --no-cache-dir numpy Cython

# Install testing/evaluation dependencies
RUN pip install --no-cache-dir \
    PyYAML>=6.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scikit-learn>=1.0.0 \
    joblib>=1.0.0

# Install audio analysis dependencies
RUN pip install --no-cache-dir \
    essentia-tensorflow>=2.1b6.dev0 \
    librosa>=0.9.0 \
    soundfile>=0.10.0

# Install madmom from git (requires Cython pre-installed)
RUN pip install --no-cache-dir git+https://github.com/CPJKU/madmom.git

# Copy the application code
COPY . /app/

# Install the neckenml-analyzer package with audio extras
RUN pip3 install -e ".[audio]" || pip3 install -e .

# Create models directory
RUN mkdir -p /root/.neckenml/models

# Create a startup script that creates the symlink if needed
RUN echo '#!/bin/bash\n\
# Create symlink for voice_instrumental model if the source file exists\n\
if [ -f /root/.neckenml/models/voice_instrumental-msd-musicnn-1.pb ] && [ ! -e /root/.neckenml/models/voice_instrumental-musicnn-msd-1.pb ]; then\n\
  ln -sf /root/.neckenml/models/voice_instrumental-msd-musicnn-1.pb /root/.neckenml/models/voice_instrumental-musicnn-msd-1.pb\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
