# Use the PyTorch CUDA-enabled runtime image as a base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for sound (libsndfile1, ffmpeg).
# 'git' is no longer strictly necessary if you're not cloning inside the Dockerfile,
# but it's good practice to keep it if your app might use it for other reasons.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install the remaining Python packages
# PyTorch, numpy are already in the base image.
RUN pip install --no-cache-dir \
    gradio \
    flask \
    scipy

# Copy the entire Chatterbox repository (the build context) into /app.
# Since your Dockerfile is in the root of the chatterbox repo, and you
# are likely building from that directory, '.' refers to the chatterbox repo itself.
COPY . /app

# Install ChatterboxTTS in editable mode from the copied source code at /app.
# This assumes that the 'setup.py' for ChatterboxTTS is directly in /app after the copy.
RUN pip install --no-cache-dir -e /app

# Expose the ports that the Flask app and Gradio app will run on
EXPOSE 7861
EXPOSE 7860

# Command to run the application
# Ensure 'gradio_tts_app.py' is also part of the chatterbox repo and copied to /app.
CMD ["python", "gradio_tts_app.py"]