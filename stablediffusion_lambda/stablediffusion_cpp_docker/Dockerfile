# Define function directory
ARG FUNCTION_DIR="/function"

# Build Stage
FROM ubuntu:22.04 AS build
ENV DEBIAN_FRONTEND noninteractive

ARG FOUNDATIONMODEL="https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
ARG VAEMODEL="https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors"
ARG UPSCALER="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ARG TAESD="https://huggingface.co/madebyollin/taesdxl/resolve/main/diffusion_pytorch_model.safetensors"

# Install build dependencies
RUN apt-get update \
    apt-get install -y --no-install-recommends \
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3.10-venv \
    wget \
    git

# Clone the repository and perform the quantization
WORKDIR /opt
RUN git clone --recursive https://github.com/leejet/stable-diffusion.cpp
RUN mkdir /opt/stable-diffusion.cpp/models && mkdir /opt/stable-diffusion.cpp/vae && mkdir /opt/stable-diffusion.cpp/upscaler && mkdir /opt/stable-diffusion.cpp/taesd
WORKDIR /opt/stable-diffusion.cpp/models
RUN extension="${FOUNDATIONMODEL##*.}" && wget -O foundationmodel.${extension} ${FOUNDATIONMODEL}
WORKDIR /opt/stable-diffusion.cpp/vae
RUN extension="${VAEMODEL##*.}" && wget -O vaemodel.${extension} ${VAEMODEL}
WORKDIR /opt/stable-diffusion.cpp/upscaler
RUN extension="${UPSCALER##*.}" && wget -O upscaler.${extension} ${UPSCALER}
WORKDIR /opt/stable-diffusion.cpp/taesd
RUN extension="${TAESD##*.}" && wget -O taesd.${extension} ${TAESD}

#No more quantization needed, this works at the edge, though it will need to have a smaller model size due to 10GB limitations...

# Build the binary executable
WORKDIR /opt/stable-diffusion.cpp
RUN git submodule init && \
    git submodule update
RUN mkdir build
WORKDIR /opt/stable-diffusion.cpp/build
RUN cmake .. && \
    cmake --build . --config Release

# Final Stage
FROM --platform=arm64 ubuntu:22.04 AS final
ENV DEBIAN_FRONTEND noninteractive
ARG FUNCTION_DIR

# Copy necessary files from build stage
COPY --from=build /opt/stable-diffusion.cpp/models/* /opt/stable-diffusion.cpp/models/
COPY --from=build /opt/stable-diffusion.cpp/vae/* /opt/stable-diffusion.cpp/vae/
COPY --from=build /opt/stable-diffusion.cpp/upscaler/* /opt/stable-diffusion.cpp/upscaler/
COPY --from=build /opt/stable-diffusion.cpp/taesd/* /opt/stable-diffusion.cpp/taesd/
COPY --from=build /opt/stable-diffusion.cpp/build/bin/sd /opt/stable-diffusion.cpp/build/bin/sd

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3.10-venv \
    git\
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set up the runtime environment
RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}
COPY main.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    pip3 install awslambdaric
#I don't know if I want to keep this upscaling method. Maybe compare the two?
# Set up xbrz.py
RUN git clone https://github.com/ioistired/xbrz.py.git /xbrz.py  ##--target "${FUNCTION_DIR}"
WORKDIR /xbrz.py
RUN pip3 install pillow && \
    pip3 install .

# Set up environment variables
ENV MODELPATH /opt/stable-diffusion.cpp/models
ENV VAEPATH /opt/stable-diffusion.cpp/vae
ENV UPSCALEPATH /opt/stable-diffusion.cpp/upscaler
ENV TAEPATH /opt/stable-diffusion.cpp/taesd
ENV SDPATH /opt/stable-diffusion.cpp/build/bin/sd
ENV IMAGE_DIMENSION 512

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}
ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
CMD [ "main.handler" ]
