# ---- build the Rust web server ----
# Built on the same Ubuntu release as the runtime image so glibc matches.
FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.97.0
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
RUN cargo build --release --locked

# ---- runtime: Python compute env (RDKit, pandas, nextflow) + server binary ----
FROM ubuntu:22.04
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget unzip vim git-core

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc

COPY requirements.yml .
RUN mamba env create -f requirements.yml

# Ketcher structure editor (standalone build, served at /static/ketcher/)
ARG KETCHER_VERSION=3.18.0
RUN wget -q https://github.com/epam/ketcher/releases/download/v${KETCHER_VERSION}/ketcher-standalone-${KETCHER_VERSION}.zip -O /tmp/ketcher.zip \
    && mkdir -p /opt/ketcher && unzip -q /tmp/ketcher.zip -d /opt/ketcher && rm /tmp/ketcher.zip

COPY . /app
WORKDIR /app
RUN rm -rf /app/static/ketcher && ln -s /opt/ketcher /app/static/ketcher
COPY --from=build /src/target/release/structure-masst /usr/local/bin/structure-masst
