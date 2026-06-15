FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy source files
COPY *.cu ./

# Default command
CMD ["/bin/bash"]
