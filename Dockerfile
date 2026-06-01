ARG RUNPOD_BASE_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
FROM ${RUNPOD_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update --yes \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Keep the RunPod image's default /start.sh behavior for interactive SSH and
# Jupyter development. Run the analyzer from a terminal after the Pod starts.
