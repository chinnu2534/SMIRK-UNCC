FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone SMIRK repo (original, no modifications)
RUN git clone https://github.com/georgeretsi/smirk.git .

# Install PyTorch3D prebuilt wheel for py310/cu117/torch2.0.1
# -f adds the wheel page as an extra source; pip still fetches deps (iopath, fvcore) from PyPI
RUN pip install --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html

# Install SMIRK requirements (skip torch/torchvision already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Install Gradio for the web UI (5.x is compatible with huggingface_hub>=0.26)
RUN pip install --no-cache-dir "gradio>=5.0,<6.0"

# Copy the Gradio wrapper app (only new file, original repo unchanged)
COPY app.py /app/app.py
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

# startup.sh downloads models at runtime using secrets, then launches app
CMD ["bash", "/app/startup.sh"]
