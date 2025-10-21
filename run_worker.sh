#!/bin/bash

source activate StructureMASST_env
celery -A tasks worker --loglevel=INFO --concurrency=8 --max-tasks-per-child=30