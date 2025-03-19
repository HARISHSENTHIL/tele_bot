FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for H100 optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download LoRA adapters (if not mounted as a volume)
RUN mkdir -p flux-lora-adapters-00 && \
    if [ ! -f flux-lora-adapters-00/flux-lora-adapters-00.safetensors ]; then \
        wget -q -O flux-lora-adapters-00/flux-lora-adapters-00.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux-lora-adapters-00.safetensors || echo "Could not download LoRA adapters, will continue with base model"; \
    fi

# Copy the rest of the application
COPY . .

# Create directory for generated images
RUN mkdir -p generated_images

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"] 