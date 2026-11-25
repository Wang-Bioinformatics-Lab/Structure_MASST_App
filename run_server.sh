#!/bin/bash
# Starts the StructureMASST web server (Rust). It launches the Python compute
# worker (bin/compute_worker.py) itself, using the conda env's python.

source activate StructureMASST_env
export STRUCTUREMASST_PYTHON="$(which python)"
export STRUCTUREMASST_NEXTFLOW="$(which nextflow)"
export PORT="${PORT:-5000}"
cd /app
exec structure-masst
